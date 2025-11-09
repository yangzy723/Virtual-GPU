#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "list.h"
#include "util.h"
#include "my_elf.h"
#include "resource-mg.h"
#include "gsched.h"

typedef CUresult (*orig___cuModuleLoadData_t)(CUmodule *module, const void *image);
typedef CUresult (*orig___cuModuleGetFunction_t)(CUfunction *hfunc, CUmodule hmod, const char *name);
typedef CUresult (*orig___cuModuleGetGlobal_t)(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);

list kernel_infos;

typedef struct {
	__u_int mem_data_len;
	char *mem_data_val;
} mem_data;

extern gsched_t sched_none;

gsched_t *sched;

__attribute__((constructor)) static void init(void) {
    cudaError_t cres;
    if ((cres = cudaSetDevice(0)) != cudaSuccess) {
        printf("cudaSetDevice failed: %d\n", cres);
    }
    cudaDeviceSynchronize();
    list_init(&kernel_infos, sizeof(kernel_info_t));
    sched = &sched_none;
    if (sched->init() != 0) {
        printf("initializing scheduler failed.\n");
        exit(1);
    }
    resource_mg_init(&rm_modules, 0);
    resource_mg_init(&rm_functions, 0);
    resource_mg_init(&rm_globals, 0);
}

static orig___cuModuleLoadData_t orig_cuModuleLoadData = NULL;
void **__cudaRegisterFatBinary(void *fatCubin)
{
    uint8_t *fatbin_data = NULL;
    size_t fatbin_size = 0;
    mem_data rpc_fat = { .mem_data_len = 0, .mem_data_val = NULL };

    if (elf2_get_fatbin_info((struct fat_header *)fatCubin, &kernel_infos, (uint8_t **)&rpc_fat.mem_data_val, &fatbin_size) != 0) {
        printf("error getting fatbin info\n");
        return NULL;
    }
    rpc_fat.mem_data_len = fatbin_size;
    if (orig_cuModuleLoadData == NULL)
        orig_cuModuleLoadData = get_sym("cuModuleLoadData");
    CUmodule module;
    void **result = (void**)calloc(1, 0x58);
    GSCHED_RETAIN;
    CUresult res = orig_cuModuleLoadData(&module, rpc_fat.mem_data_val);
    GSCHED_RELEASE;
    if (res != CUDA_SUCCESS) {
        printf("cuModuleLoadData failed: %d\n", res);
    }
    else{
        printf("cuModuleLoadData succeeded: %d\n", res);
    }
    resource_mg_add_sorted(&rm_modules, (void *)result, (void *)module);
    return result;
}

static orig___cuModuleGetGlobal_t orig_cuModuleGetGlobal = NULL;
void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress,
                       const char *deviceName, int ext, size_t size, int constant,
                       int global)
{
    size_t d_size = 0;
    CUdeviceptr dptr = 0;
    CUmodule module = resource_mg_get(&rm_modules, (void *)fatCubinHandle);
    if (orig_cuModuleGetGlobal == NULL)
        orig_cuModuleGetGlobal = get_sym("cuModuleGetGlobal");
    GSCHED_RETAIN;
    orig_cuModuleGetGlobal(&dptr, &d_size, module, deviceName);
    GSCHED_RELEASE;
    resource_mg_add_sorted(&rm_globals, (void *)hostVar, (void *)dptr);
}

static orig___cuModuleGetFunction_t orig_cuModuleGetFunction = NULL;
void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize)
{
    // 打印出来 deviceFun 和 deviceName 好像是一样的
    // printf("%s\n", deviceName);
    kernel_info_t *info = utils_search_info(&kernel_infos, (char *)deviceName);
    if (info == NULL) {
        printf("request to register unknown function: \"%s\"\n", deviceName);
        return;
    } else {
        // printf("request to register known function: \"%s\"\n", deviceName);
        CUfunction f;
        CUmodule module = resource_mg_get(&rm_modules, (void *)fatCubinHandle);
        if (orig_cuModuleGetFunction == NULL)
            orig_cuModuleGetFunction = get_sym("cuModuleGetFunction");
        GSCHED_RETAIN;
        if (orig_cuModuleGetFunction(&f, module, deviceName) != CUDA_SUCCESS) {
            printf("cuModuleGetFunction failed for %s\n", deviceName);
            GSCHED_RELEASE;
            return;
        }
        else{
            printf("cuModuleGetFunction succeeded for %s\n", deviceName);
        }
        GSCHED_RELEASE;
        resource_mg_add_sorted(&rm_functions, (void *)hostFun, (void *)(f));
        info->host_fun = (void *)hostFun;
    }
}

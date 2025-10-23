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

typedef CUresult (*orig___cuModuleLoadData_t)(CUmodule *module, const void *image);
typedef CUresult (*orig___cuModuleGetFunction_t)(CUfunction *hfunc, CUmodule hmod, const char *name);

list kernel_infos;

int cnt = 0;
static orig___cuModuleLoadData_t orig_cuModuleLoadData = NULL;
void **__cudaRegisterFatBinary(void *fatCubin)
{
    cnt++;
    printf("__cudaRegisterFatBinary(fatCubin=%p), called %d times\n", fatCubin, cnt);

    uint8_t *fatbin_data = NULL;
    size_t fatbin_size = 0;

    if (elf2_get_fatbin_info((struct fat_header *)fatCubin, &kernel_infos, &fatbin_data, &fatbin_size) != 0) {
        printf("error getting fatbin info\n");
        return NULL;
    }

    if (orig_cuModuleLoadData == NULL)
        orig_cuModuleLoadData = get_sym("cuModuleLoadData");
    CUmodule module;
    void **result = (void**)calloc(1, 0x58);
    orig_cuModuleLoadData(&module, fatbin_data);
    resource_mg_add_sorted(&rm_modules, (void *)result, (void *)module);
    return result;
}

static orig___cuModuleGetFunction_t orig_cuModuleGetFunction = NULL;
void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize)
{
    kernel_info_t *info = utils_search_info(&kernel_infos, (char *)deviceName);
    if (info == NULL) {
        // printf("request to register unknown function: \"%s\"", deviceName);
        return;
    } else {
        printf("request to register known function: \"%s\"", deviceName);
        CUfunction f;
        CUmodule module = resource_mg_get(&rm_modules, (void *)fatCubinHandle);
        if (orig_cuModuleGetFunction == NULL)
            orig_cuModuleGetFunction = get_sym("cuModuleGetFunction");
        orig_cuModuleGetFunction(&f, module, deviceName);
        resource_mg_add_sorted(&rm_functions, (void *)hostFun, (void *)(&f));
        info->host_fun = (void *)hostFun;
    }
}
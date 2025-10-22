#include <stdio.h>
#include <string.h>
#include <dlfcn.h>

#include "list.h"
#include "util.h"
#include "resource-mg.h"

int cnt = 0;

typedef void **(*orig___cuModuleLoadData_t)(CUmodule module, const void *fatCubin);
void **__cudaRegisterFatBinary(void *fatCubin)
{
    cnt++;
    printf("__cudaRegisterFatBinary(fatCubin=%p), called %d times\n", fatCubin, cnt);

    void **result = NULL;
    uint8_t *fatbin_data = NULL;
    size_t fatbin_size = 0;

    if (elf2_get_fatbin_info((struct fat_header *)fatCubin, &kernel_infos, &fatbin_data, &fatbin_size) != 0) {
        printf("error getting fatbin info\n");
        return NULL;
    }


    static orig___cuModuleLoadData_t orig_cuModuleLoadData = get_sym("cuModuleLoadData");
    CUmodule module;
    void **result = (void**)calloc(1, 0x58);
    orig_cuModuleLoadData(&module, fatbin_data);
    resource_mg_add_sorted(&rm_modules, (void *)result, (void *)module);
    return result;
}

void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize)
{
    kernel_info_t *info = utils_search_info(&kernel_infos, (char *)deviceName);
    if (info == NULL) {
        printf("request to register unknown function: \"%s\"", deviceName);
        return;
    } else {
        printf("request to register known function: \"%s\"", deviceName);
        CUfunction f;
        CUmodule module = resource_mg_get(&rm_modules, (void *)fatCubinHandle)
        cuModuleGetFunction(&f, module, deviceName);
        resource_mg_add_sorted(&rm_functions, (void *)hostFun, (void *)(&f));
        info->host_fun = (void *)hostFun;
    }
}
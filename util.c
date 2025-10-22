#include "util.h"

#include <stdio.h>
#include <dlfcn.h>
#include <string.h>

kernel_info_t* utils_search_info(list *_kernel_infos, const char *kernelname)
{
    kernel_info_t *info = NULL;
    if (_kernel_infos == NULL) {
        printf("list is NULL.\n");
        return NULL;
    }
    printf("searching for %s in %ld entries\n", kernelname, _kernel_infos->length);
    for (int i=0; i < _kernel_infos->length; ++i) {
        if (list_at(_kernel_infos, i, (void**)&info) != 0) {
            printf("no element at index %d\n", i);
        }
        if (strcmp(kernelname, info->name) == 0) {
            return info;
        }
    }
    return NULL;
}

void *get_sym(const char *name) {
    void *p = dlsym(RTLD_NEXT, name);
    if (p) return p;

    const char *libs[] = {
        "libcudart.so",
        "libcudart.so.12",
        "libcuda.so",
        "libcuda.so.12",
        NULL
    };

    for (int i = 0; libs[i]; ++i) {
        void *handle = dlopen(libs[i], RTLD_LAZY | RTLD_NOLOAD);
        if (handle) {
            p = dlsym(handle, name);
            if (p) return p;
        }
    }

    fprintf(stderr, "[HOOK] get_sym: cannot find symbol %s\n", name);
    return NULL;
}

void hexdump(const uint8_t* data, size_t size)
{
    size_t pos = 0;
    while (pos < size) {
        printf("%#05zx: ", pos);
        for (int i = 0; i < 16; i++) {
            if (pos + i < size) {
                printf("%02x", data[pos + i]);
            } else {
                printf("  ");
            }
            if (i % 4 == 3) {
                printf(" ");
            }
        }
        printf(" | ");
        for (int i = 0; i < 16; i++) {
            if (pos + i < size) {
                if (data[pos + i] >= 0x20 && data[pos + i] <= 0x7e) {
                    printf("%c", data[pos + i]);
                } else {
                    printf(".");
                }
            } else {
                printf(" ");
            }
        }
        printf("\n");
        pos += 16;
    }
}

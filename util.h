#ifndef _UTIL_H_
#define _UTIL_H_

#include "list.h"

typedef struct kernel_info {
    char *name;
    size_t param_size;
    size_t param_num;
    uint16_t *param_offsets;
    uint16_t *param_sizes;
    void *host_fun;
} kernel_info_t;

kernel_info_t* utils_search_info(list *kernel_infos, const char *kernelname);
void *get_sym(const char *name);
void hexdump(const uint8_t* data, size_t size);


#endif //_UTIL_H_

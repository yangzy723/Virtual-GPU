#include "my_elf.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

int elf2_get_fatbin_info(const struct fat_header *fatbin, list *kernel_infos, uint8_t** fatbin_mem, size_t* fatbin_size)
{
    struct fat_elf_header* eh;
    struct fat_text_header* th;
    const uint8_t *input_pos = NULL;
    const uint8_t *fatbin_data = NULL;
    uint8_t *text_data = NULL;
    size_t text_data_size = 0;
    size_t fatbin_total_size = 0;
    int ret = -1;
    if (fatbin == NULL || fatbin_mem == NULL || fatbin_size == NULL) {
        printf("at least one parameter is NULL\n");
        goto error;
    }
    fatbin_data = input_pos = (const uint8_t*)fatbin->text;
    if (fatbin->magic != FATBIN_STRUCT_MAGIC) {
        printf("fatbin struct magic number is wrong. Got %llx, expected %llx.\n", fatbin->magic, FATBIN_STRUCT_MAGIC);
        goto error;
    }
    printf("Fatbin: magic: %x, version: %x, text: %lx, data: %lx, ptr: %lx, ptr2: %lx, zero: %lx\n",
           fatbin->magic, fatbin->version, fatbin->text, fatbin->data, fatbin->unknown, fatbin->text2, fatbin->zero);

    if (get_elf_header((uint8_t*)fatbin->text, sizeof(struct fat_elf_header), &eh) != 0) {
        printf("Something went wrong while checking the header.\n");
        goto error;
    }
    printf("elf header: magic: %#x, version: %#x, header_size: %#x, size: %#zx\n",
           eh->magic, eh->version, eh->header_size, eh->size); 

    input_pos += eh->header_size;
    fatbin_total_size = eh->header_size + eh->size;
    do {
        if (get_text_header(input_pos, *fatbin_size - (input_pos - fatbin_data) - eh->header_size, &th) != 0) {
            printf("Something went wrong while checking the header.\n");
            goto error;
        }
        //print_header(th);
        input_pos += th->header_size;
        if (th->kind != 2) { // section does not cotain device code (but e.g. PTX)
            input_pos += th->size;
            continue;
        }
        if (th->flags & FATBIN_FLAG_DEBUG) {
            printf("fatbin contains debug information.\n");
        }

        if (th->flags & FATBIN_FLAG_COMPRESS) {
            ssize_t input_read;

            printf("fatbin contains compressed device code. Decompressing...\n");
            if ((input_read = decompress_single_section(input_pos, &text_data, &text_data_size, eh, th)) < 0) {
                printf("Something went wrong while decompressing text section.\n");
                goto error;
            }
            input_pos += input_read;
            //hexdump(text_data, text_data_size);
        } else {
            text_data = (uint8_t*)input_pos;
            text_data_size = th->size;
            input_pos += th->size;
        }
        // print_header(th);
        if (elf2_parameter_info(kernel_infos, text_data , text_data_size) != 0) {
            printf("error getting parameter info\n");
            goto error;
        }
        if (th->flags & FATBIN_FLAG_COMPRESS) {
            free(text_data);
        }
    } while (input_pos < (uint8_t*)eh + eh->header_size + eh->size);

    // if (get_elf_header((uint8_t*)fatbin->text2, sizeof(struct fat_elf_header), &eh) != 0) {
    //     LOGE(LOG_ERROR, "Something went wrong while checking the header.");
    //     goto error;
    // }
    // fatbin_total_size += eh->header_size + eh->size;

    *fatbin_mem = (void*)fatbin->text;
    *fatbin_size = fatbin_total_size;
    ret = 0;
 error:
    return ret;
}

static int get_elf_header(const uint8_t* fatbin_data, size_t fatbin_size, struct fat_elf_header **elf_header)
{
    struct fat_elf_header *eh = NULL;

    if (fatbin_data == NULL || elf_header == NULL) {
        printf("fatbin_data is NULL\n");
        return 1;
    }

    if (fatbin_size < sizeof(struct fat_elf_header)) {
        printf("fatbin_size is too small\n");
        return 1;
    }

    eh = (struct fat_elf_header*) fatbin_data;
    if (eh->magic != FATBIN_TEXT_MAGIC) {
        printf("Invalid magic  number: expected %#x but got %#x\n", FATBIN_TEXT_MAGIC, eh->magic);
        return 1;
    }

    if (eh->version != 1 || eh->header_size != sizeof(struct fat_elf_header)) {
        printf("fatbin text version is wrong or header size is inconsistent.\
            This is a sanity check to avoid reading a new fatbinary format\n");
        return 1;
    }
    
    *elf_header = eh;
    return 0;
}

static int get_text_header(const uint8_t* fatbin_data, size_t fatbin_size, struct fat_text_header **text_header)
{
    struct fat_text_header *th = NULL;

    if (fatbin_data == NULL || text_header == NULL) {
        printf("fatbin_data is NULL\n");
        return 1;
    }

    if (fatbin_size < sizeof(struct fat_text_header)) {
        printf("fatbin_size is too small\n");
        return 1;
    }

    th = (struct fat_text_header*)fatbin_data;

    if(th->obj_name_offset != 0) {
        if (((char*)th)[th->obj_name_offset + th->obj_name_len] != '\0') {
            printf("Fatbin object name is not null terminated\n");
        } else {
            char *obj_name = (char*)th + th->obj_name_offset;
            printf("Fatbin object name: %s (len:%#x)\n", obj_name, th->obj_name_len);
        }
    }

    *text_header = th;
    return 0;
}

static ssize_t decompress_single_section(const uint8_t *input, uint8_t **output, size_t *output_size,
                                         struct fat_elf_header *eh, struct fat_text_header *th)
{
    size_t padding;
    size_t input_read = 0;
    size_t output_written = 0;
    size_t decompress_ret = 0;
    const uint8_t zeroes[8] = {0};

    if (input == NULL || output == NULL || eh == NULL || th == NULL) {
        printf("invalid parameters\n");
        return 1;
    }

    // add max padding of 7 bytes
    if ((*output = malloc(th->decompressed_size + 7)) == NULL) {
        printf("Error allocating memory of size %#zx for output buffer: %s\n", th->decompressed_size, strerror(errno));
        goto error;
    }
    print_header(th);

    if ((decompress_ret = decompress(input, th->compressed_size, *output, th->decompressed_size)) != th->decompressed_size) {
        printf("Decompression failed: decompressed size is %#zx, but header says %#zx\n", decompress_ret, th->decompressed_size);
        printf("input pos: %#zx, output pos: %#zx\n", input - (uint8_t*)eh, *output);
        hexdump(input, 0x160);
        if (decompress_ret >= 0x60)
            hexdump((*output) + decompress_ret - 0x60, 0x60);
        goto error;
    }
    input_read += th->compressed_size;
    output_written += th->decompressed_size;

    padding = ((8 - (size_t)(input + input_read)) % 8);
    if (memcmp(input + input_read, zeroes, padding) != 0) {
        printf("expected %#zx zero bytes, got:\n", padding);
        hexdump(input + input_read, 0x60);
        goto error;
    }
    input_read += padding;

    padding = ((8 - (size_t)th->decompressed_size) % 8);
    // Because we always allocated enough memory for one more elf_header and this is smaller than
    // the maximal padding of 7, we do not have to reallocate here.
    memset(*output, 0, padding);
    output_written += padding;

    *output_size = output_written;
    return input_read;
 error:
    free(*output);
    *output = NULL;
    return -1;
}

/** Decompresses a fatbin file
 * @param input Pointer compressed input data
 * @param input_size Size of compressed data
 * @param output preallocated memory where decompressed output should be stored
 * @param output_size size of output buffer. Should be equal to the size of the decompressed data
 */
static size_t decompress(const uint8_t* input, size_t input_size, uint8_t* output, size_t output_size)
{
    size_t ipos = 0, opos = 0;  
    uint64_t next_nclen;  // length of next non-compressed segment
    uint64_t next_clen;   // length of next compressed segment
    uint64_t back_offset; // negative offset where redudant data is located, relative to current opos

    while (ipos < input_size) {
        next_nclen = (input[ipos] & 0xf0) >> 4;
        next_clen = 4 + (input[ipos] & 0xf);
        if (next_nclen == 0xf) {
            do {
                next_nclen += input[++ipos];
            } while (input[ipos] == 0xff);
        }
        
        if (memcpy(output + opos, input + (++ipos), next_nclen) == NULL) {
            printf("copying data\n");
            return 0;
        }
#ifdef FATBIN_DECOMPRESS_DEBUG
        printf("%#04zx nocompress (len:%#x):\n", opos, next_nclen);
        hexdump(output + opos, next_nclen);
#endif
        ipos += next_nclen;
        opos += next_nclen;
        if (ipos >= input_size || opos >= output_size) {
            break;
        }
        back_offset = input[ipos] + (input[ipos + 1] << 8);       
        ipos += 2;
        if (next_clen == 0xf+4) {
            do {
                next_clen += input[ipos++];
            } while (input[ipos - 1] == 0xff);
        }
#ifdef FATBIN_DECOMPRESS_DEBUG
        printf("%#04zx compress (decompressed len: %#x, back_offset %#x):\n", opos, next_clen, back_offset);
#endif
        if (next_clen <= back_offset) {
            if (memcpy(output + opos, output + opos - back_offset, next_clen) == NULL) {
                printf("Error copying data\n");
                return 0;
            }
        } else {
            if (memcpy(output + opos, output + opos - back_offset, back_offset) == NULL) {
                printf("Error copying data\n");
                return 0;
            }
            for (size_t i = back_offset; i < next_clen; i++) {
                output[opos + i] = output[opos + i - back_offset];
            }
        }
#ifdef FATBIN_DECOMPRESS_DEBUG
        hexdump(output + opos, next_clen);
#endif
        opos += next_clen;
    }
    printf("ipos: %#zx, opos: %#zx, ilen: %#zx, olen: %#zx\n", ipos, opos, input_size, output_size);
    return opos;
}

static void print_header(struct fat_text_header *th)
{
    printf("text_header: fatbin_kind: %#x, header_size %#x, size %#zx, compressed_size %#x,\
 minor %#x, major %#x, arch %d, decompressed_size %#zx\n\n",
        th->kind,
        th->header_size,
        th->size,
        th->compressed_size,
        th->minor,
        th->major,
        th->arch,
        th->decompressed_size);
    printf("\tunknown fields: unknown1: %#x, unknown2: %#x, zeros: %#zx\n\n",
        th->unknown1,
        th->unknown2,
        th->zero);
}

int elf2_parameter_info(list *kernel_infos, void* memory, size_t memsize)
{
    struct __attribute__((__packed__)) nv_info_entry{
        uint8_t format;
        uint8_t attribute;
        uint16_t values_size;
        uint32_t kernel_id;
        uint32_t value;
    };

    Elf *elf = NULL;
    Elf_Scn *section = NULL;
    Elf_Data *data = NULL, *symbol_table_data = NULL;
    GElf_Shdr symtab_shdr;
    size_t symnum;
    int i = 0;
    GElf_Sym sym;

    int ret = -1;
    kernel_info_t *ki = NULL;
    const char *kernel_str;

    if (memory == NULL || memsize == 0) {
        printf("memory was NULL or memsize was 0\n");
        return -1;
    }

// #define ELF_DUMP_TO_FILE 1

#ifdef ELF_DUMP_TO_FILE
    FILE* fd2 = fopen("/tmp/cricket-elf-dump", "wb");
    fwrite(memory, memsize, 1, fd2);
    fclose(fd2);
#endif

    if ((elf = elf_memory(memory, memsize)) == NULL) {
        printf("elf_memory failed\n");
        goto cleanup;
    }

    if (check_elf(elf) != 0) {
        printf("check_elf failed\n");
        goto cleanup;
    }

    if (get_symtab(elf, &symbol_table_data, &symnum, &symtab_shdr) != 0) {
        printf("could not get symbol table\n");
        goto cleanup;
    }

    if (get_section_by_name(elf, ".nv.info", &section) != 0) {
        printf("could not find .nv.info section. This means this binary does not contain any kernels.\n");
        ret = 0;    // This is not an error.
        goto cleanup;
    }

    if ((data = elf_getdata(section, NULL)) == NULL) {
        printf("elf_getdata failed\n");
        goto cleanup;
    }

    for (size_t secpos=0; secpos < data->d_size; secpos += sizeof(struct nv_info_entry)) {
        struct nv_info_entry *entry = (struct nv_info_entry *)(data->d_buf+secpos);
        // LOGE(LOG_DBG(1), "%d: format: %#x, attr: %#x, values_size: %#x kernel: %#x, sval: %#x(%d)", 
        // i++, entry->format, entry->attribute, entry->values_size, entry->kernel_id, 
        // entry->value, entry->value);

        if (entry->values_size != 8) {
            printf("unexpected values_size: %#x\n",
                 entry->values_size);
            continue;
        }

        if (entry->attribute != EIATTR_FRAME_SIZE) {
            continue;
        }

        if (entry->kernel_id >= symnum) {
            printf("kernel_id out of bounds: %#x\n", entry->kernel_id);
            continue;
        }

        if (gelf_getsym(symbol_table_data, entry->kernel_id, &sym) == NULL) {
            printf("gelf_getsym failed for entry %d\n", entry->kernel_id);
            continue;
        }

        if ((kernel_str = elf_strptr(elf, symtab_shdr.sh_link, sym.st_name) ) == NULL) {
            printf("strptr failed for entry %d\n", entry->kernel_id);
            continue;
        }

        /* When using (some?) intrinsics, nvcc adds symbols for them in the .nv.info table.
        * They are prefixed with $__internal_7_$ and are not kernels. We skip them he
        */
        const char *intrinsics_prefix = "$__internal_";
        if (strncmp(kernel_str, intrinsics_prefix, strlen(intrinsics_prefix)) == 0) {
            continue;
        }

        if (utils_search_info(kernel_infos, kernel_str) != NULL) {
            continue;
        }

        // TODO: printf("found new kernel: %s (symbol table id: %#x)\n", kernel_str, entry->kernel_id);

        if (list_append(kernel_infos, (void**)&ki) != 0) {
            printf("error on appending to list\n");
            goto cleanup;
        }

        size_t buflen = strlen(kernel_str)+1;
        if ((ki->name = malloc(buflen)) == NULL) {
            printf("malloc failed\n");
            goto cleanup;
        }
        if (strncpy(ki->name, kernel_str, buflen) != ki->name) {
            printf("strncpy failed\n");
            goto cleanup;
        }

        if (get_parm_for_kernel(elf, ki, memory, memsize) != 0) {
            printf("get_parm_for_kernel failed for kernel %s\n", kernel_str);
            goto cleanup;
        }
    }

    ret = 0;
 cleanup:
    if (elf != NULL) {
        elf_end(elf);
    }
    return ret;
}

static int check_elf(Elf *elf)
{
    Elf_Kind ek;
    GElf_Ehdr ehdr;

    int elfclass;
    char *id;
    size_t program_header_num;
    size_t sections_num;
    size_t section_str_num;
    int ret = -1;

    if ((ek = elf_kind(elf)) != ELF_K_ELF) {
        printf("elf_kind is not ELF_K_ELF, but %d\n", ek);
        goto cleanup;
    }

    if (gelf_getehdr(elf, &ehdr) == NULL) {
        printf("gelf_getehdr failed\n");
        goto cleanup;
    }

    if ((elfclass = gelf_getclass(elf)) == ELFCLASSNONE) {
        printf("gelf_getclass failed\n");
        goto cleanup;
    }

    if ((id = elf_getident(elf, NULL)) == NULL) {
        printf("elf_getident failed\n");
        goto cleanup;
    }

    printf("elfclass: %d-bit; elf ident[0..%d]: %7s\n",
        (elfclass == ELFCLASS32) ? 32 : 64,
        EI_ABIVERSION, id);

    if (elf_getshdrnum(elf, &sections_num) != 0) {
        printf("elf_getphdrnum failed\n");
        goto cleanup;
    }

    if (elf_getphdrnum(elf, &program_header_num) != 0) {
        printf("elf_getshdrnum failed\n");
        goto cleanup;
    }

    if (elf_getshdrstrndx(elf, &section_str_num) != 0) {
        printf("elf_getshstrndx Wfailed\n");
        goto cleanup;
    }

    printf("elf contains %ld sections, %ld program_headers, string table section: %ld\n",
        sections_num, program_header_num, section_str_num);

    ret = 0;
cleanup:
    return ret;
}

static int get_symtab(Elf *elf, Elf_Data **symbol_table_data, size_t *symbol_table_size, GElf_Shdr *symbol_table_shdr)
{
    GElf_Shdr shdr;
    Elf_Scn *section = NULL;

    if (elf == NULL || symbol_table_data == NULL || symbol_table_size == NULL) {
        printf("invalid argument\n");
        return -1;
    }

    if (get_section_by_name(elf, ".symtab", &section) != 0) {
        printf("could not find .symtab section\n");
        return -1;
    }

    if (gelf_getshdr(section, &shdr) == NULL) {
        printf("gelf_getshdr failed\n");
        return -1;
    }

    if (symbol_table_shdr != NULL) {
        *symbol_table_shdr = shdr;
    }

    if(shdr.sh_type != SHT_SYMTAB) {
        printf("not a symbol table: %d\n", shdr.sh_type);
        return -1;
    }

    if ((*symbol_table_data = elf_getdata(section, NULL)) == NULL) {
        printf("elf_getdata failed\n");
        return -1;
    }

    *symbol_table_size = shdr.sh_size / shdr.sh_entsize;

    return 0;
}

static int get_section_by_name(Elf *elf, const char *name, Elf_Scn **section)
{
    Elf_Scn *scn = NULL;
    GElf_Shdr shdr;
    char *section_name = NULL;
    size_t str_section_index;

    if (elf == NULL || name == NULL || section == NULL) {
        printf("invalid argument\n");
        return -1;
    }

    if (elf_getshdrstrndx(elf, &str_section_index) != 0) {
        printf("elf_getshstrndx failed\n");
        return -1;
    }

    while ((scn = elf_nextscn(elf, scn)) != NULL) {
        if (gelf_getshdr(scn, &shdr) != &shdr) {
            printf("gelf_getshdr failed\n");
            return -1;
        }
        if ((section_name = elf_strptr(elf, str_section_index, shdr.sh_name)) == NULL) {
            printf("elf_strptr failed\n");
            return -1;
        }
        if (strcmp(section_name, name) == 0) {
            *section = scn;
            return 0;
        }
    }
    return -1;
}


static int get_parm_for_kernel(Elf *elf, kernel_info_t *kernel, void* memory, size_t memsize)
{
    struct __attribute__((__packed__)) nv_info_kernel_entry {
        uint8_t format;
        uint8_t attribute;
        uint16_t values_size;
        uint32_t values;
    };
    struct __attribute__((__packed__)) nv_info_kparam_info {
        uint32_t index;
        uint16_t ordinal;
        uint16_t offset;
        uint16_t unknown : 12;
        uint8_t  cbank : 6;
        uint16_t size : 14;
        // missing are "space" (possible padding info?), and "Pointee's logAlignment"
        // these were always 0 in the kernels I tested
    };
    int ret = -1;
    char *section_name = NULL;
    Elf_Scn *section = NULL;
    Elf_Data *data = NULL;

    if (kernel == NULL || kernel->name == NULL || memory == NULL) {
        printf("at least one parameter is NULL\n");
        goto cleanup;
    }
    kernel->param_num = 0;
    kernel->param_size = 0;
    kernel->param_offsets = NULL;
    kernel->param_sizes = NULL;

    if ((section_name = get_kernel_section_from_kernel_name(kernel->name)) == NULL) {
        printf("get_kernel_section_from_kernel_name failed\n");
        goto cleanup;
    }

    if (get_section_by_name(elf, section_name, &section) != 0) {
        printf("section %s not found\n", section_name);
        goto cleanup;
    }

    if ((data = elf_getdata(section, NULL)) == NULL) {
        printf("error getting section data\n");
        goto cleanup;
    }

    //print_hexmem(data->d_buf, data->d_size);

    size_t secpos=0;
    int i=0;
    while (secpos < data->d_size) {
        struct nv_info_kernel_entry *entry = (struct nv_info_kernel_entry*)(data->d_buf+secpos);
        // printf("entry %d: format: %#x, attr: %#x, ", i++, entry->format, entry->attribute);
        if (entry->format == EIFMT_SVAL && entry->attribute == EIATTR_KPARAM_INFO) {
            if (entry->values_size != 0xc) {
                printf("EIATTR_KPARAM_INFO values size has not the expected value of 0xc\n");
                goto cleanup;
            }
            struct nv_info_kparam_info *kparam = (struct nv_info_kparam_info*)&entry->values;
            // printf("kparam: index: %#x, ordinal: %#x, offset: %#x, unknown: %#0x, cbank: %#0x, size: %#0x\n",
            //     kparam->index, kparam->ordinal, kparam->offset, kparam->unknown, kparam->cbank, kparam->size);
            // TODO: printf("param %d: offset: %#x, size: %#x\n", kparam->ordinal, kparam->offset, kparam->size);
            if (kparam->ordinal >= kernel->param_num) {
                kernel->param_offsets = realloc(kernel->param_offsets,
                                              (kparam->ordinal+1)*sizeof(uint16_t));
                kernel->param_sizes = realloc(kernel->param_sizes,
                                            (kparam->ordinal+1)*sizeof(uint16_t));
                kernel->param_num = kparam->ordinal+1;
            }
            kernel->param_offsets[kparam->ordinal] = kparam->offset;
            kernel->param_sizes[kparam->ordinal] = kparam->size;
            secpos += sizeof(struct nv_info_kernel_entry) + entry->values_size-4;
        } else if (entry->format == EIFMT_HVAL && entry->attribute == EIATTR_CBANK_PARAM_SIZE) {
            kernel->param_size = entry->values_size;
            printf("cbank_param_size: %#0x\n", entry->values_size);
            secpos += sizeof(struct nv_info_kernel_entry)-4;
        } else if (entry->format == EIFMT_HVAL) {
            // printf("hval: %#x(%d)\n", entry->values_size, entry->values_size);
            secpos += sizeof(struct nv_info_kernel_entry)-4;
        } else if (entry->format == EIFMT_SVAL) {
            // printf("sval_size: %#x ", entry->values_size);
            // for (int j=0; j*sizeof(uint32_t) < entry->values_size; j++) {
            //     printf("val%d: %#x(%d) ", j, (&entry->values)[j], (&entry->values)[j]);
            // }
            // printf("\n");
            secpos += sizeof(struct nv_info_kernel_entry) + entry->values_size-4;
        } else if (entry->format == EIFMT_NVAL) {
            // printf("nval\n");
            secpos += sizeof(struct nv_info_kernel_entry)-4;
        } else {
            printf("unknown format: %#x\n", entry->format);
            secpos += sizeof(struct nv_info_kernel_entry)-4;
        }
    }
    // printf("remaining: %d\n", data->d_size % sizeof(struct nv_info_kernel_entry));
    ret = 0;
 cleanup:
    free(section_name);
    return ret;
}

static char* get_kernel_section_from_kernel_name(const char *kernel_name)
{
    char *section_name = NULL;
    if (kernel_name == NULL) {
        printf("invalid argument\n");
        return NULL;
    }

    if (kernel_name[0] == '$') {
        const char *p;
        if ((p = strchr(kernel_name+1, '$')) == NULL) {
            printf("invalid kernel name\n");
            return NULL;
        }
        int len = (p - kernel_name) - 1;
        if (asprintf(&section_name, ".nv.info.%.*s", len, kernel_name+1) == -1) {
            printf("asprintf failed\n");
            return NULL;
        }
    } else {
        if (asprintf(&section_name, ".nv.info.%s", kernel_name) == -1) {
            printf("asprintf failed\n");
            return NULL;
        }
    }
    return section_name;
}

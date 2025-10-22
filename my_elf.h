#ifndef ELF2_H
#define ELF2_H

#include "list.h"
#include "util.h"

#include <gelf.h>
#include <libelf.h>

struct __attribute__((__packed__)) fat_header {
    uint32_t magic;
    uint32_t version;
    uint64_t text;      // points to first text section
    uint64_t data;      // points to outside of the file
    uint64_t unknown;
    uint64_t text2;     // points to second text section
    uint64_t zero;
};

struct  __attribute__((__packed__)) fat_elf_header
{
    uint32_t magic;
    uint16_t version;
    uint16_t header_size;
    uint64_t size;
};

struct  __attribute__((__packed__)) fat_text_header
{
    uint16_t kind;
    uint16_t unknown1;
    uint32_t header_size;
    uint64_t size;
    uint32_t compressed_size;       // Size of compressed data
    uint32_t unknown2;              // Address size for PTX?
    uint16_t minor;
    uint16_t major;
    uint32_t arch;
    uint32_t obj_name_offset;
    uint32_t obj_name_len;
    uint64_t flags;
    uint64_t zero;                  // Alignment for compression?
    uint64_t decompressed_size;     // Length of compressed data in decompressed representation.
                                    // There is an uncompressed footer so this is generally smaller
                                    // than size.
};

#define FATBIN_FLAG_64BIT     0x0000000000000001LL
#define FATBIN_FLAG_DEBUG     0x0000000000000002LL
#define FATBIN_FLAG_LINUX     0x0000000000000010LL
#define FATBIN_FLAG_COMPRESS  0x0000000000002000LL

#define FATBIN_STRUCT_MAGIC 0x466243b1
#define FATBIN_TEXT_MAGIC   0xBA55ED50

#define EIFMT_SVAL 4
#define EIFMT_HVAL 3
#define EIFMT_NVAL 1

#define EIATTR_PARAM_CBANK 10
#define EIATTR_CBANK_PARAM_SIZE 25
#define EIATTR_KPARAM_INFO 23
#define EIATTR_MAXREG_COUNT 27
#define EIATTR_S2RCTAID_INSTR_OFFSETS 29
#define EIATTR_EXIT_INSTR_OFFSETS 28
#define EIATTR_EXTERNS 15
#define EIATTR_CRS_STACK_SIZE 30
#define EIATTR_MAX_STACK_SIZE 35
#define EIATTR_MIN_STACK_SIZE 18 // maximal size of the stack when calling this kernel
#define EIATTR_FRAME_SIZE 17 // size of stack in this function (without subcall)

int elf2_get_fatbin_info(const struct fat_header *fatbin, list *kernel_infos, uint8_t **fatbin_mem, size_t *fatbin_size);
int elf2_parameter_info(list *kernel_infos, void *memory, size_t memsize);
static int get_elf_header(const uint8_t *fatbin_data, size_t fatbin_size, struct fat_elf_header **elf_header);
static int get_text_header(const uint8_t *fatbin_data, size_t fatbin_size, struct fat_text_header **text_header);
static ssize_t decompress_single_section(const uint8_t *input, uint8_t **output, size_t *output_size, struct fat_elf_header *eh, struct fat_text_header *th);
static size_t decompress(const uint8_t *input, size_t input_size, uint8_t *output, size_t output_size);
static int check_elf(Elf *elf);
static int get_symtab(Elf *elf, Elf_Data **symbol_table_data, size_t *symbol_table_size, GElf_Shdr *symbol_table_shdr);
static int get_section_by_name(Elf *elf, const char *name, Elf_Scn **section);
static int get_parm_for_kernel(Elf *elf, kernel_info_t *kernel, void *memory, size_t memsize);
static char *get_kernel_section_from_kernel_name(const char *kernel_name);
static void print_header(struct fat_text_header *th);

#endif
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

int elf2_get_fatbin_info(const struct fat_header *fatbin, list *kernel_infos, uint8_t **fatbin_mem, size_t *fatbin_size);

#endif
#include "vgpu/common/fatbin_parser.h"

#include <algorithm>
#include <cctype>
#include <cstring>

namespace vgpu {

namespace {

// Wrapper magic:  what __cudaRegisterFatBinary receives
static constexpr uint32_t kFatWrapMagic = 0x466243B1u;
// Actual fatbin image magic
static constexpr uint32_t kFatBinMagic  = 0xBA55ED50u;

// Follow fatbin wrapper pointer if necessary and return pointer to the raw
// fatbin image.  Updates out_size with the image size computed from the header
// (or left unchanged/0 if not parseable).
const uint8_t* resolveImage(const void* fat_data, size_t& out_size) {
    if (!fat_data) return nullptr;
    const uint8_t* p = static_cast<const uint8_t*>(fat_data);

    uint32_t magic = 0;
    std::memcpy(&magic, p, sizeof(magic));

    // If it's a wrapper struct, follow the .data pointer at offset 8.
    if (magic == kFatWrapMagic) {
        const void* data_ptr = nullptr;
        std::memcpy(&data_ptr, p + 8, sizeof(data_ptr));
        if (!data_ptr) return nullptr;
        p = static_cast<const uint8_t*>(data_ptr);
        std::memcpy(&magic, p, sizeof(magic));
    }

    if (magic != kFatBinMagic) return nullptr;

    // CUDA 12+ fatbin header variant:
    //   uint32 magic, uint32 version, uint64 data_size, uint32 unknown, uint32 header_size
    uint32_t version = 0;
    uint64_t data_size64 = 0;
    uint32_t header_size_v2 = 0;
    std::memcpy(&version,        p + 4,  sizeof(version));
    std::memcpy(&data_size64,    p + 8,  sizeof(data_size64));
    std::memcpy(&header_size_v2, p + 20, sizeof(header_size_v2));
    if (version == 0x00100001u && header_size_v2 >= 16 && header_size_v2 <= (1u << 20)) {
        out_size = static_cast<size_t>(data_size64 + static_cast<uint64_t>(header_size_v2));
        return p;
    }

    // Fatbin header layout (16 bytes):
    //   uint32 magic, uint32 version, uint32 header_size, uint32 data_size
    uint32_t header_size = 0, data_size = 0;
    std::memcpy(&header_size, p + 8,  sizeof(header_size));
    std::memcpy(&data_size,   p + 12, sizeof(data_size));
    out_size = static_cast<size_t>(header_size) + static_cast<size_t>(data_size);
    return p;
}

// Map a PTX type token to its byte size.
uint32_t ptxTypeSize(const char* tok, size_t len) {
    // Common PTX scalar types
    struct { const char* name; uint32_t sz; } table[] = {
        {"u8",1},{"s8",1},{"b8",1},
        {"u16",2},{"s16",2},{"b16",2},{"f16",2},
        {"u32",4},{"s32",4},{"b32",4},{"f32",4},
        {"u64",8},{"s64",8},{"b64",8},{"f64",8},
        {"u128",16},{"b128",16},
    };
    for (const auto& e : table) {
        if (std::strlen(e.name) == len && std::memcmp(e.name, tok, len) == 0)
            return e.sz;
    }
    return 8; // default to pointer-size
}

// Lightweight PTX parser: extract .entry / .param declarations.
std::vector<KernelParamInfo> parsePtx(const char* ptx, size_t ptx_len) {
    std::vector<KernelParamInfo> result;
    const char* end = ptx + ptx_len;
    const char* p   = ptx;

    while (p < end) {
        // --- find ".entry" ---
        const char* ep = static_cast<const char*>(memmem(p, end - p, ".entry", 6));
        if (!ep) break;
        p = ep + 6;

        // skip whitespace
        while (p < end && std::isspace((unsigned char)*p)) ++p;

        // read function name (up to '(' or whitespace)
        const char* nm = p;
        while (p < end && *p != '(' && !std::isspace((unsigned char)*p)) ++p;
        std::string func_name(nm, p);

        // skip to '('
        while (p < end && *p != '(') ++p;
        if (p >= end) break;
        ++p; // skip '('

        KernelParamInfo ki;
        ki.mangled_name = std::move(func_name);

        // parse params until ')'
        while (p < end && *p != ')' && *p != '{') {
            // skip whitespace / commas
            while (p < end && (std::isspace((unsigned char)*p) || *p == ',')) ++p;
            if (p >= end || *p == ')' || *p == '{') break;

            if (std::strncmp(p, ".param", 6) != 0) {
                // skip unknown token
                while (p < end && *p != ',' && *p != ')') ++p;
                continue;
            }
            p += 6;
            while (p < end && std::isspace((unsigned char)*p)) ++p;

            // optional .align N
            if (std::strncmp(p, ".align", 6) == 0) {
                p += 6;
                while (p < end && std::isspace((unsigned char)*p)) ++p;
                while (p < end && std::isdigit((unsigned char)*p)) ++p;
                while (p < end && std::isspace((unsigned char)*p)) ++p;
            }

            // type token: must start with '.'
            if (p >= end || *p != '.') continue;
            ++p; // skip '.'
            const char* type_start = p;
            while (p < end && (std::isalnum((unsigned char)*p) || *p == '_')) ++p;
            size_t type_len = static_cast<size_t>(p - type_start);
            uint32_t sz = ptxTypeSize(type_start, type_len);

            // skip param name
            while (p < end && std::isspace((unsigned char)*p)) ++p;
            while (p < end && *p != ',' && *p != ')' && *p != '{') {
                if (*p == '[') {
                    // Array param: .param .u8 name[N]
                    ++p;
                    const char* n = p;
                    while (p < end && *p != ']') ++p;
                    uint32_t cnt = static_cast<uint32_t>(std::atoi(n));
                    if (cnt > 1) sz *= cnt;
                    if (p < end) ++p; // skip ']'
                    break;
                }
                ++p;
            }

            ParamInfo pi;
            pi.size = sz;
            pi.alignment = std::max(sz, 4u);
            ki.params.push_back(pi);
        }

        ki.total_param_bytes = computeParamBufSize(ki.params);
        if (!ki.mangled_name.empty()) {
            result.push_back(std::move(ki));
        }
    }
    return result;
}

// Scan raw bytes for PTX blocks and parse them.
std::vector<KernelParamInfo> scanForPtx(const uint8_t* data, size_t sz) {
    std::vector<KernelParamInfo> result;
    const char* p   = reinterpret_cast<const char*>(data);
    const char* end = p + sz;

    while (p < end) {
        // find ".version" (start of PTX program)
        const char* found = static_cast<const char*>(memmem(p, end - p, ".version", 8));
        if (!found) break;

        // validate: next non-space char should be a digit
        const char* v = found + 8;
        while (v < end && std::isspace((unsigned char)*v)) ++v;
        if (v >= end || !std::isdigit((unsigned char)*v)) {
            p = found + 1;
            continue;
        }

        // Determine end of PTX: scan while ASCII-printable or whitespace
        const char* ptx_end = found;
        while (ptx_end < end &&
               (std::isprint((unsigned char)*ptx_end) || std::isspace((unsigned char)*ptx_end))) {
            ++ptx_end;
        }

        auto kernels = parsePtx(found, static_cast<size_t>(ptx_end - found));
        result.insert(result.end(), kernels.begin(), kernels.end());
        p = ptx_end;
    }
    return result;
}

}  // namespace

// -------------------------------------------------------------------------
// Public API
// -------------------------------------------------------------------------

std::vector<KernelParamInfo> parseFatbin(const void* fat_data, size_t fat_size) {
    if (!fat_data) return {};

    size_t raw_size = fat_size;
    const uint8_t* raw = resolveImage(fat_data, raw_size);

    if (raw) {
        // Use resolved image size when header-derived value is larger
        if (fat_size > 0 && raw_size > fat_size) raw_size = fat_size;
        return scanForPtx(raw, raw_size);
    }

    // Fall-back: maybe it's raw PTX text
    const char* cp = static_cast<const char*>(fat_data);
    size_t check_len = (fat_size > 0) ? fat_size : std::strlen(cp);
    if (check_len >= 8 && std::strncmp(cp, ".version", 8) == 0) {
        return parsePtx(cp, check_len);
    }

    // Last resort: scan the raw bytes provided
    if (fat_size > 0) {
        return scanForPtx(static_cast<const uint8_t*>(fat_data), fat_size);
    }
    return {};
}

uint32_t computeParamBufSize(const std::vector<ParamInfo>& params) {
    uint32_t off = 0;
    for (const auto& p : params) {
        uint32_t align = p.alignment;
        off = (off + align - 1u) & ~(align - 1u);
        off += p.size;
    }
    // Round total up to 8-byte boundary (CUDA convention)
    return (off + 7u) & ~7u;
}

std::vector<uint8_t> packArgs(const std::vector<ParamInfo>& params, void** args) {
    if (!args || params.empty()) return {};

    uint32_t total = computeParamBufSize(params);
    std::vector<uint8_t> buf(total, 0);

    uint32_t off = 0;
    for (size_t i = 0; i < params.size(); ++i) {
        const auto& p = params[i];
        uint32_t align = p.alignment;
        off = (off + align - 1u) & ~(align - 1u);
        if (args[i] && off + p.size <= total) {
            std::memcpy(buf.data() + off, args[i], p.size);
        }
        off += p.size;
    }
    return buf;
}

}  // namespace vgpu

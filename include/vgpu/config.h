#pragma once

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>

namespace vgpu::config {

inline std::string trim(std::string s) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

inline std::string resolveConfigPath() {
    const char* env = std::getenv("GPU_SCHEDULER_CONFIG");
    if (env && env[0]) return env;

    static const char* candidates[] = {
        "./vgpu.conf",
        "./config/vgpu.conf",
        "/etc/vgpu.conf",
        nullptr,
    };
    for (const char** p = candidates; *p; ++p) {
        std::error_code ec;
        if (std::filesystem::exists(*p, ec) && !ec) {
            return *p;
        }
    }

    return "";
}

inline std::unordered_map<std::string, std::string> loadConfigFile() {
    std::unordered_map<std::string, std::string> kv;
    std::string path = resolveConfigPath();
    if (path.empty()) return kv;

    std::ifstream in(path);
    if (!in.is_open()) return kv;

    std::string line;
    while (std::getline(in, line)) {
        auto hash = line.find('#');
        if (hash != std::string::npos) {
            line = line.substr(0, hash);
        }

        line = trim(line);
        if (line.empty()) continue;

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = trim(line.substr(0, eq));
        std::string value = trim(line.substr(eq + 1));
        if (key.empty() || value.empty()) continue;

        kv[key] = value;
    }

    return kv;
}

inline const std::unordered_map<std::string, std::string>& values() {
    static const std::unordered_map<std::string, std::string> kValues = loadConfigFile();
    return kValues;
}

inline std::string getEnvOrConfig(const char* key, const std::string& fallback = "") {
    const char* env = std::getenv(key);
    if (env && env[0]) return env;

    auto it = values().find(key);
    if (it != values().end()) return it->second;

    return fallback;
}

inline int getInt(const char* key, int fallback, int minimum, int maximum) {
    const std::string raw = getEnvOrConfig(key);
    if (raw.empty()) return fallback;

    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(raw.c_str(), &end, 10);
    if (errno == ERANGE || end == raw.c_str() || *end != '\0' ||
        parsed < minimum || parsed > maximum) {
        return fallback;
    }
    return static_cast<int>(parsed);
}

inline bool getBool(const char* key, bool fallback = false) {
    std::string raw = getEnvOrConfig(key);
    if (raw.empty()) return fallback;

    std::string v;
    v.reserve(raw.size());
    for (unsigned char c : raw) v.push_back(static_cast<char>(std::tolower(c)));

    if (v == "1" || v == "true" || v == "yes" || v == "on") return true;
    if (v == "0" || v == "false" || v == "no" || v == "off") return false;
    return fallback;
}

}  // namespace vgpu::config

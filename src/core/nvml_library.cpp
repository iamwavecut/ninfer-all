#include "core/nvml_library.h"

#include <string>

#if defined(_WIN32)
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

namespace ninfer {

void* open_nvml_library() noexcept {
#if defined(_WIN32)
    // The driver installs nvml.dll in System32; drivers before 2019 kept it in NVSMI.
    HMODULE module = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (module == nullptr) {
        wchar_t root[MAX_PATH];
        const DWORD length = GetEnvironmentVariableW(L"ProgramW6432", root, MAX_PATH);
        if (length > 0 && length < MAX_PATH) {
            try {
                const std::wstring path =
                    std::wstring(root) + L"\\NVIDIA Corporation\\NVSMI\\nvml.dll";
                module = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
            } catch (...) { module = nullptr; }
        }
    }
    return module;
#else
    return dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
}

void* nvml_symbol(void* library, const char* name) noexcept {
    if (library == nullptr) { return nullptr; }
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(library), name));
#else
    return dlsym(library, name);
#endif
}

void close_nvml_library(void* library) noexcept {
    if (library == nullptr) { return; }
#if defined(_WIN32)
    FreeLibrary(static_cast<HMODULE>(library));
#else
    dlclose(library);
#endif
}

} // namespace ninfer

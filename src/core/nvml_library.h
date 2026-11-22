#pragma once

namespace ninfer {

// The NVIDIA management library is the driver's, loaded at run time: a build needs no NVML headers,
// and a host without NVML still starts. Null when the library is not installed.
[[nodiscard]] void* open_nvml_library() noexcept;
// One entry point of a library open_nvml_library() returned; null when it lacks the name.
[[nodiscard]] void* nvml_symbol(void* library, const char* name) noexcept;
void close_nvml_library(void* library) noexcept;

} // namespace ninfer

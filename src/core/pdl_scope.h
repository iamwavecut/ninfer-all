#pragma once

namespace ninfer::pdl {

// Whether pdl::launch_arch (core/pdl_arch.cuh) may make a launch on this thread a programmatic
// dependent: only inside an ArchScope. An executor whose own kernels release their dependents early
// (pdl::arch_enter_glue) opens one around its decode-graph capture. Elsewhere the kernel before a
// GGUF vector launch releases it only at its exit, and the vector kernel's early start then
// measured slower (Qwen3.8-27B IQ3_S on an RTX 5090: 105.8 -> 103.4 tok/s).
inline thread_local int arch_scope_depth = 0;

class ArchScope {
public:
    ArchScope() noexcept { ++arch_scope_depth; }

    ~ArchScope() { --arch_scope_depth; }

    ArchScope(const ArchScope&)            = delete;
    ArchScope& operator=(const ArchScope&) = delete;
};

} // namespace ninfer::pdl

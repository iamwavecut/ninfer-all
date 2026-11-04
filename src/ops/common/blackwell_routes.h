#pragma once

// The NVFP4 A4 route runs Blackwell instructions (block-scaled FP4 MMA, TMA). The FP8 A8 route
// runs the plain E4M3 m16n8k32 MMA on sm_89 (Ada) and up, and the block-scaled Blackwell forms on
// 120a; its TMA schedule stays Blackwell-only. A build without 89 and 120a links
// fp8_sm86_stubs.cpp and nvfp4_sm86_stubs.cpp in their place, which refuse every call. A build of
// several architectures carrying those routes; their entry points refuse, with the same messages,
// a call on a card that does not run the code they need, before any of their kernels (which trap
// in that card's image) is launched.

#include "ops/common/device_tier.h"

#include <stdexcept>

namespace ninfer::ops::detail {

inline void require_fp8_a8_card() {
    if constexpr (kBuildTiers > 1) {
        if (!current_device_fp8_mma()) {
            throw std::runtime_error(
                "FP8 A8 execution requires an sm_89 (Ada FP8) or sm_100a or sm_120a GPU");
        }
    }
}

inline void require_nvfp4_a4_card() {
    if constexpr (kBuildTiers > 1) {
        if (!current_device_blackwell()) {
            throw std::runtime_error("NVFP4 A4 execution requires an sm_120a GPU");
        }
    }
}

} // namespace ninfer::ops::detail

#pragma once

// The codings of the INT8-family caches' planes as the causal attention kernels take them. The
// host launchers pick one per launch (causal_i8_coding in route.h); the kernel units take it as a
// launch argument and never see the storage enum.

#include <cstdint>

namespace ninfer::ops::detail {

// INT8 keys and values (int8), INT8 keys over packed int4 values (rk8v4), or packed keys over
// packed int4 values (rk4v4's Lloyd-Max indices, rk4v4-e8's E8-snapped int4, rk2v4-e8's E8 roots).
enum class CausalI8Coding : std::uint8_t { Int8, PackedValues, Lloyd4, Int4E8, RootE8 };

} // namespace ninfer::ops::detail

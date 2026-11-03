#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/common/route_table.h"
#include "ops/linear/q4/q4_shapes.h"
#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
struct ShapeEntry {
    std::int32_t n, k;
    Q4Launch (*select)(std::int32_t);
    // The unified-template table; shapes without one keep the legacy routes everywhere.
    Q4Launch (*unified)(std::int32_t) = nullptr;
    // FP32 out on the widths it names; the runtime-shape route elsewhere.
    Q4Launch (*fp32)(std::int32_t) = nullptr;
};

constexpr std::array kShapes{
    ShapeEntry{1024, 5120, select_q4_n1024_k5120, select_q4_n1024_k5120_unified},
    ShapeEntry{4096, 5120, select_q4_n4096_k5120, select_q4_n4096_k5120_unified},
    ShapeEntry{5120, 4096, select_q4_n5120_k4096},
    ShapeEntry{5120, 6144, select_q4_n5120_k6144, select_q4_n5120_k6144_unified},
    ShapeEntry{5120, 17408, select_q4_n5120_k17408},
    ShapeEntry{5120, 25600, select_q4_n5120_k25600},
    ShapeEntry{6144, 5120, select_q4_n6144_k5120, select_q4_n6144_k5120_unified},
    ShapeEntry{7168, 5120, select_q4_n7168_k5120, select_q4_n7168_k5120_unified},
    ShapeEntry{34816, 5120, select_q4_n34816_k5120, select_q4_n34816_k5120_unified},
    ShapeEntry{131072, 5120, select_q4_n131072_k5120, select_q4_n131072_k5120_unified},
    ShapeEntry{131072, 2048, select_q4_n131072_k2048, select_q4_n131072_k2048_unified},
    ShapeEntry{248320, 5120, select_q4_n248320_k5120, nullptr, select_q4_n248320_k5120_fp32},
    ShapeEntry{3456, 1152, select_q4_n3456_k1152, select_q4_n3456_k1152_unified},
    ShapeEntry{4304, 1152, select_q4_n4304_k1152, select_q4_n4304_k1152_unified},
};
} // namespace

Q4Launch select_q4_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) throw std::invalid_argument("q4 linear: T must be positive");
    for (const auto& entry : kShapes) {
        if (entry.n != n || entry.k != k) continue;
        if (entry.unified != nullptr &&
            linear_route_table(LinearRouteFamily::Q4, t) == LinearRouteTable::Unified) {
            return entry.unified(t);
        }
        return entry.select(t);
    }
    return select_q4_general_launch(n, k, t, DType::BF16);
}

Q4Launch select_q4_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("q4 linear: unsupported policy");
    return select_q4_a16_launch(n, k, t);
}

Q4Launch select_q4_fp32_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    for (const auto& entry : kShapes) {
        if (entry.n != n || entry.k != k || entry.fp32 == nullptr) continue;
        if (const Q4Launch launch = entry.fp32(t)) { return launch; }
    }
    return select_q4_general_launch(n, k, t, DType::FP32);
}

void q4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    const Q4Launch launch = out.dtype == DType::FP32
                                ? select_q4_fp32_launch(weight.n, weight.k, x.ne[1])
                                : select_q4_launch(weight.n, weight.k, x.ne[1], policy);
    launch(x, weight, out, stream);
}
} // namespace ninfer::ops::detail

#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/common/route_table.h"
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_shapes_unified.h"
#include "ops/linear/fp8/fp8_format.h"
#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
const std::array kShapes{&kFp8N14336K5120, &kFp8N16384K5120, &kFp8N34816K5120,
                         &kFp8N5120K6144,  &kFp8N5120K17408, &kFp8N248320K5120};
const std::array kUnifiedShapes{&unified::kFp8N14336K5120, &unified::kFp8N16384K5120,
                                &unified::kFp8N34816K5120, &unified::kFp8N5120K6144,
                                &unified::kFp8N5120K17408, &unified::kFp8N248320K5120};

const Fp8LinearShape& resolve_shape(std::int32_t n, std::int32_t k, LinearPolicy policy,
                                    LinearRouteTable table) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("fp8 linear: unsupported policy");
    for (const auto* shape : table == LinearRouteTable::Unified ? kUnifiedShapes : kShapes)
        if (shape->n == n && shape->k == k) return *shape;
    throw std::invalid_argument("fp8 linear: unsupported shape");
}
} // namespace

bool fp8_registered(std::int32_t n, std::int32_t k) {
    for (const auto* shape : kShapes)
        if (shape->n == n && shape->k == k) return true;
    return false;
}

std::size_t fp8_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k, LinearPolicy policy,
                                                std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("fp8 linear workspace: invalid token interval");
    if (!fp8_registered(n, k)) {
        if (!valid_linear_policy(policy)) throw std::invalid_argument("fp8 linear: unsupported policy");
        return allows_a8(policy) && fp8_general_uses_a8(max_tokens)
                   ? fp8_a8_workspace_capacity_bytes(max_tokens, k)
                   : 0;
    }
    return capacity_by_table(
        min_tokens, max_tokens,
        [&](std::int32_t width) {
            return linear_route_table(LinearRouteFamily::Fp8, width, fp8_wide_route_table(policy),
                                      allows_a8(policy));
        },
        [&](LinearRouteTable table, std::int32_t first, std::int32_t last) -> std::size_t {
            const auto& shape = resolve_shape(n, k, policy, table);
            if (!allows_a8(policy) || !shape.uses_a8(first, last)) { return 0; }
            return fp8_a8_workspace_capacity_bytes(
                last, k, shape.partial_capacity_bytes ? shape.partial_capacity_bytes(last) : 0);
        });
}

void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                  WorkspaceArena* workspace, cudaStream_t stream) {
    validate_fp8_weight(weight, "fp8 linear");
    if (x.ne[1] <= 0) throw std::invalid_argument("fp8 linear: T must be positive");
    const bool registered = fp8_registered(weight.n, weight.k);
    if (out.dtype == DType::FP32 && registered) {
        const auto& shape = resolve_shape(
            weight.n, weight.k, policy,
            linear_route_table(LinearRouteFamily::Fp8, x.ne[1], fp8_wide_route_table(policy),
                               allows_a8(policy)));
        if (shape.a16_fp32 != nullptr) return shape.a16_fp32(x, weight, out, stream);
    }
    if (out.dtype == DType::FP32 || !registered) {
        if (!valid_linear_policy(policy)) throw std::invalid_argument("fp8 linear: unsupported policy");
        if (!registered && allows_a8(policy) && fp8_general_uses_a8(x.ne[1])) {
            if (workspace == nullptr)
                throw std::invalid_argument("fp8 A8 linear requires caller workspace");
            return fp8_general_a8(x, weight, out, *workspace, stream);
        }
        return fp8_general_a16(x, weight, out, stream);
    }
    const auto& shape =
        resolve_shape(weight.n, weight.k, policy,
                      linear_route_table(LinearRouteFamily::Fp8, x.ne[1],
                                         fp8_wide_route_table(policy), allows_a8(policy)));
    if (!allows_a8(policy) || !shape.uses_a8(x.ne[1], x.ne[1]))
        return shape.a16(x, weight, out, stream);
    if (workspace == nullptr)
        throw std::invalid_argument("fp8 A8 linear requires caller workspace");
    auto scope         = workspace->scope();
    const auto scratch = allocate_fp8_a8_workspace(
        *workspace, x.ne[1], weight.k,
        shape.partial_capacity_bytes ? shape.partial_capacity_bytes(x.ne[1]) : 0);
    shape.a8(x, weight, out, scratch, stream);
}
} // namespace ninfer::ops::detail

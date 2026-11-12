#include "core/weight.h"
#include "ops/common/blackwell_routes.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

#include "core/device_launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_quantize.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_tma_launch.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail::unified {
namespace {

template <class ActivationGeometry>
void launch_quantize(const Tensor& x, const Weight& weight, Nvfp4A4Workspace workspace,
                     Nvfp4ScaleLayout layout, cudaStream_t stream, bool reciprocal) {
    const std::int32_t tokens = x.ne[1];
    constexpr int kThreads    = 256;
    const auto* input         = static_cast<const __nv_bfloat16*>(x.data);
    const std::int32_t groups_per_row = weight.k / 16;
    // The tiled plane is allocated and addressed over whole tiles, so its launch has to reach the
    // padding of the last one; the row-major plane stops at the real token count.
    const std::int32_t written_tokens =
        layout != Nvfp4ScaleLayout::RowMajor ? nvfp4_a4_padded_tokens(tokens, layout) : tokens;
    const std::int32_t tasks = written_tokens * groups_per_row;
    const int blocks         = (tasks + kThreads - 1) / kThreads;
    // The tiled plane exists only for a K whose group count divides into whole tiles. Registered
    // Ks all do; gating the instantiation keeps a future K that does not out of a hard compile
    // error and into a runtime message.
    if constexpr ((ActivationGeometry::kInputRows / 16) % kNvfp4ScaleTileGroups == 0) {
        if (layout != Nvfp4ScaleLayout::RowMajor) {
            if (groups_per_row % kNvfp4ScaleTileGroups != 0) {
                throw std::invalid_argument("nvfp4 A4 tiled scales need K groups in whole tiles");
            }
            if (layout == Nvfp4ScaleLayout::Tiled128) {
                nvfp4_a4_quantize_kernel<ActivationGeometry, kThreads, Nvfp4ScaleLayout::Tiled128>
                    <<<blocks, kThreads, 0, stream>>>(input, workspace.codes, workspace.scales,
                                                      tokens, written_tokens,
                                                      weight.input_scale_divisor, weight.k);
            } else {
                nvfp4_a4_quantize_kernel<ActivationGeometry, kThreads, Nvfp4ScaleLayout::Tiled256>
                    <<<blocks, kThreads, 0, stream>>>(input, workspace.codes, workspace.scales,
                                                      tokens, written_tokens,
                                                      weight.input_scale_divisor, weight.k);
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    } else if (layout != Nvfp4ScaleLayout::RowMajor) {
        throw std::invalid_argument("nvfp4 A4 tiled scales need K groups in whole tiles");
    }
    // The row-major plane feeds the MMA routes, the decode widths a CUDA Graph captures.
    if (reciprocal) {
        CUDA_CHECK(pdl::launch_consumer({dim3(blocks), dim3(kThreads), 0, stream},
                                        nvfp4_a4_quantize_kernel<ActivationGeometry, kThreads,
                                                                 Nvfp4ScaleLayout::RowMajor, true>,
                                        input, workspace.codes, workspace.scales, tokens,
                                        written_tokens, weight.input_scale_divisor, weight.k));
        return;
    }
    CUDA_CHECK(pdl::launch_consumer(
        {dim3(blocks), dim3(kThreads), 0, stream},
        nvfp4_a4_quantize_kernel<ActivationGeometry, kThreads, Nvfp4ScaleLayout::RowMajor>, input,
        workspace.codes, workspace.scales, tokens, written_tokens, weight.input_scale_divisor,
        weight.k));
}

} // namespace

void launch_nvfp4_a4_quantize(const Tensor& x, const Weight& weight, Nvfp4A4Workspace workspace,
                              Nvfp4ScaleLayout layout, cudaStream_t stream,
                              bool reciprocal_quotient) {
    require_nvfp4_a4_card();
    if (workspace.codes == nullptr || workspace.scales == nullptr) {
        throw std::invalid_argument("nvfp4 A4 requires caller workspace");
    }
    // The tiled layout writes the padding of the last tile, so it needs a plane allocated over the
    // padded token count. Check that here, where the layout is acted on, rather than trust each
    // caller to have sized it that way.
    if (layout != Nvfp4ScaleLayout::RowMajor) {
        const std::size_t needed =
            static_cast<std::size_t>(nvfp4_a4_padded_tokens(x.ne[1], layout)) *
            (static_cast<std::size_t>(weight.k) / 16);
        if (workspace.scale_bytes < needed) {
            throw std::invalid_argument(
                "nvfp4 A4 tiled scales need a plane allocated over whole token tiles");
        }
    }
    switch (weight.k) {
    case Nvfp4Activation5120Geometry::kInputRows:
        launch_quantize<Nvfp4Activation5120Geometry>(x, weight, workspace, layout, stream,
                                                     reciprocal_quotient);
        return;
    case Nvfp4Activation6144Geometry::kInputRows:
        launch_quantize<Nvfp4Activation6144Geometry>(x, weight, workspace, layout, stream,
                                                     reciprocal_quotient);
        return;
    case Nvfp4Activation17408Geometry::kInputRows:
        launch_quantize<Nvfp4Activation17408Geometry>(x, weight, workspace, layout, stream,
                                                      reciprocal_quotient);
        return;
    default:
        // A tensor-parallel rank's share of an output projection's input, or of a K no table
        // registers: whole 64-column words of four group scales, as the row-major plane reads.
        if (weight.k <= 0 || weight.k % 64 != 0) {
            throw std::invalid_argument("nvfp4 A4 quantize: K must be a positive multiple of 64");
        }
        launch_quantize<Nvfp4RuntimeActivationGeometry>(x, weight, workspace, layout, stream,
                                                        reciprocal_quotient);
        return;
    }
}

} // namespace ninfer::ops::detail::unified

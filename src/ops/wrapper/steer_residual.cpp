// ninfer::ops - steer_residual wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/op-development.md §2.
#include "ninfer/ops/steer_residual.h"

#include "ops/launcher/steer_residual.h" // detail::steer_residual_launch

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("steer_residual: ") + message); }
}

bool matrix(const Tensor& t) { return t.ne[2] == 1 && t.ne[3] == 1 && t.is_contiguous(); }

} // namespace

void steer_residual(SteerResidualStep step, Tensor& x, const Tensor& directions,
                    const Tensor& factors, const SteerResidualLanes& lanes, Tensor& measures,
                    const Tensor& vector, cudaStream_t stream) {
    require(x.data != nullptr && x.dtype == DType::BF16 && matrix(x) && x.ne[0] > 0 && x.ne[1] >= 0,
            "x must be a contiguous BF16 [H, T] tensor");
    const std::int32_t hidden  = x.ne[0];
    const std::int32_t columns = x.ne[1];
    const std::int32_t count   = directions.data == nullptr ? 0 : directions.ne[1];
    if (count > 0) {
        require(directions.dtype == DType::FP32 && matrix(directions) &&
                    directions.ne[0] == hidden && count <= kSteerResidualMaximumDirections,
                "directions must be a contiguous FP32 [H, E] tensor, E <= 8");
        const Tensor& scales = lanes.lane_scales;
        require(scales.data != nullptr && scales.dtype == DType::FP32 && matrix(scales) &&
                    scales.ne[0] == count && scales.ne[1] > 0,
                "lane_scales must be a contiguous FP32 [E, lanes] tensor");
        require(lanes.row_width > 0, "row_width must be positive");
        const std::int64_t rows = (std::int64_t(columns) + lanes.row_width - 1) / lanes.row_width;
        require(lanes.rows.data != nullptr && lanes.rows.dtype == DType::I32 &&
                    lanes.rows.is_contiguous() && lanes.rows.numel() >= rows,
                "rows must be a contiguous I32 tensor with a lane for every row of columns");
        require(measures.data != nullptr && measures.dtype == DType::FP32 && matrix(measures) &&
                    measures.ne[0] == count && measures.ne[1] >= columns,
                "measures must be a contiguous FP32 [E, >= T] tensor");
        if (step != SteerResidualStep::Measure) {
            require(factors.data != nullptr && factors.dtype == DType::FP32 &&
                        factors.is_contiguous() && factors.numel() == count,
                    "factors must be a contiguous FP32 [E] tensor");
        }
    }
    if (vector.data != nullptr) {
        require(step == SteerResidualStep::Write, "only a Write step adds a vector");
        require(vector.dtype == DType::FP32 && vector.is_contiguous() && vector.numel() == hidden,
                "vector must be a contiguous FP32 [H] tensor");
    }
    if (columns == 0 || (count == 0 && vector.data == nullptr)) { return; }
    detail::steer_residual_launch(step, x, count, directions, factors, lanes, measures, vector,
                                  stream);
}

} // namespace ninfer::ops

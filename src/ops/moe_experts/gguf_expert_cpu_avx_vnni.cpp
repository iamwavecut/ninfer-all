// The GGUF CPU experts' vector rows with AVX-VNNI's vpdpbusd for their integer dots: the
// AVX2 rows' bits (gguf_expert_cpu_x86_rows.h).
#define NINFER_GGUF_CPU_AVX_VNNI 1
#include "ops/moe_experts/gguf_expert_cpu_x86_rows.h"

namespace ninfer::ops::gguf_cpu_detail {

void rows_avx_vnni(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin, int row_end,
                   float* y, std::size_t y_stride) {
    rows_x86(w, x, row_begin, row_end, y, y_stride);
}

} // namespace ninfer::ops::gguf_cpu_detail

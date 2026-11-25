// PTQ1_0's vector products, in their own unit so they build in parallel with the other types': the
// tensor-core form on sm_80 and newer (ggml_bridge_ptq1_tc.cuh), the lane-per-block kernel before
// (ggml_bridge_ptq1.cuh).
#include "ggml_bridge_ptq1.cuh"
#include "ggml_bridge_ptq1_tc.cuh"

namespace ninfer::ops::gguf::detail {

int ptq1_vector_columns() { return ptq1::tc::takes() ? ptq1::tc::kMaxColumns : ptq1::kMaxColumns; }

// One column takes the lane kernel everywhere: the tensor cores would multiply seven columns of
// nothing (RTX 5090, PTQ1_0 Bonsai 2.5 decode: 148 tok/s on them, 167 on the lane kernel).
bool ptq1_lane_takes(int columns) {
    return columns >= 1 && columns <= ptq1::kMaxColumns &&
           !(columns >= 2 && columns <= ptq1::tc::kMaxColumns && ptq1::tc::takes());
}

template <>
void vec_launch<GGML_TYPE_PTQ1_0>(const VecJobs& args, int columns, bool fused,
                                  cudaStream_t stream) {
    if (args.count < 1 || args.count > kMaxVecJobs || (fused && args.count != 1)) {
        throw std::invalid_argument("gguf vector product: 1..4 products, one when fused");
    }
    for (int i = 0; i < args.count; ++i) {
        const VecArgs& job = args.job[i];
        // Rows start on a word: the lanes read each block as seven aligned words.
        if (job.k <= 0 || job.k % ptq1::kBlockValues != 0 ||
            job.row_bytes < std::int64_t(job.k / ptq1::kBlockValues) * ptq1::kBlockBytes ||
            job.row_bytes % 4 != 0 || reinterpret_cast<std::uintptr_t>(job.weight) % 4 != 0 ||
            job.rows <= 0 || job.k != args.job[0].k || job.row_bytes != args.job[0].row_bytes) {
            throw std::invalid_argument("gguf vector product: unsupported PTQ1_0 geometry");
        }
    }
    if (!ptq1_lane_takes(columns) && columns <= ptq1::tc::kMaxColumns && ptq1::tc::takes()) {
        return ptq1::tc::launch_columns(args, columns, fused, stream);
    }
    switch (columns) {
    case 1: return ptq1::launch_columns<1>(args, fused, stream);
    case 2: return ptq1::launch_columns<2>(args, fused, stream);
    case 3: return ptq1::launch_columns<3>(args, fused, stream);
    case 4: return ptq1::launch_columns<4>(args, fused, stream);
    case 5: return ptq1::launch_columns<5>(args, fused, stream);
    case 6: return ptq1::launch_columns<6>(args, fused, stream);
    case 7: return ptq1::launch_columns<7>(args, fused, stream);
    case 8: return ptq1::launch_columns<8>(args, fused, stream);
    default: break;
    }
    throw std::invalid_argument(
        "gguf vector product: PTQ1_0 columns must be 1..20 (1..8 before sm_80)");
}

} // namespace ninfer::ops::gguf::detail

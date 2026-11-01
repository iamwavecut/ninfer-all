#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan_unified.h"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"

namespace ninfer::ops::detail::unified {
void nvfp4_attn_input_a16_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                 Tensor& k, Tensor& v, cudaStream_t stream) {
    const int tokens = x.ne[1];
    // The fused weight's rows are [query; key; gate; value]; a tensor-parallel rank holds whole KV
    // heads and their query heads, so its share keeps that order and only shrinks. The output
    // tensors carry the row counts, so nothing about the split is baked.
    const AttentionInputShare share{q.ne[0], k.ne[0], gate.ne[0], v.ne[0]};
    if (!attention_input_share_admitted(share) || weight.n != share.total() || weight.k != 5120) {
        throw std::invalid_argument("nvfp4 attn_input_proj A16: unsupported profile");
    }
    // The decode and small-T routes' adapters bake the whole projection's segment rows, so only a
    // whole projection takes them; a share runs the same ladders below them.
    const bool whole = share.query == 6144 && share.key == 1024 && share.gate == 6144 &&
                       share.value == 1024;
    if (whole && tokens == 1) {
        nvfp4_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    if (whole && tokens <= 2) {
        nvfp4_attn_input_small_t_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    const auto p        = nvfp4_a16_operands(x, weight);
    const auto epilogue = LinearIdentityEpilogue{};
    const auto launch   = [&]<class Output>(Output output) {
        if (tokens <= 8) {
            launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 8, 2>, 5120>>(
                p, output, epilogue, stream);
            return;
        }
        if (tokens <= 16) {
            launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 8, 2>, 5120>>(p, output, epilogue,
                                                                           stream);
            return;
        }
        if (tokens <= 24) {
            launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 4, 1>, 5120>>(p, output, epilogue,
                                                                           stream);
            return;
        }
        if (tokens <= 32) {
            launch_nvfp4_a16_mma<
                Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>, 5120>>(
                p, output, epilogue, stream);
            return;
        }
        if (tokens <= 64) {
            launch_nvfp4_a16_mma<
                Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>, 5120>>(
                p, output, epilogue, stream);
            return;
        }
        if (tokens <= 128) {
            launch_nvfp4_a16_mma<
                Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>, 5120>>(
                p, output, epilogue, stream);
            return;
        }
        launch_nvfp4_a16_mma<
            Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 1, 3>, 5120>>(
            p, output, epilogue, stream);
    };
    if (whole) {
        launch(Nvfp4AttentionInputOutput{static_cast<__nv_bfloat16*>(q.data),
                                         static_cast<__nv_bfloat16*>(k.data),
                                         static_cast<__nv_bfloat16*>(gate.data),
                                         static_cast<__nv_bfloat16*>(v.data)});
    } else {
        launch(Nvfp4AttentionInputShareOutput{static_cast<__nv_bfloat16*>(q.data),
                                              static_cast<__nv_bfloat16*>(k.data),
                                              static_cast<__nv_bfloat16*>(gate.data),
                                              static_cast<__nv_bfloat16*>(v.data), share});
    }
}
} // namespace ninfer::ops::detail::unified

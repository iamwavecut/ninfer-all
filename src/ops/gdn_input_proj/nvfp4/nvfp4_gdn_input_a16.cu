#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan_unified.h"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output_unified.cuh"

namespace ninfer::ops::detail::unified {
void nvfp4_gdn_input_a16_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream) {
    const int tokens = x.ne[1];
    // The fused weight's rows are [qkv; z]; a tensor-parallel rank holds whole GDN key heads, so its
    // share keeps that order and only shrinks. The output tensors carry the row counts.
    const GdnInputShare share{qkv.ne[0], z.ne[0]};
    if (!gdn_input_share_admitted(share) || weight.n != share.total() || weight.k != 5120) {
        throw std::invalid_argument("nvfp4 gdn_input_proj A16: unsupported profile");
    }
    // The decode and small-T routes' adapters bake the whole projection's segment rows, so only a
    // whole projection takes them; a share runs the same ladders below them.
    const bool whole = share.qkv == 10240 && share.z == 6144;
    if (whole && tokens == 1) {
        nvfp4_gdn_input_decode_launch(x, weight, qkv, z, stream);
        return;
    }
    if (whole && tokens <= 2) {
        nvfp4_gdn_input_small_t_launch(x, weight, qkv, z, stream);
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
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 8, 1>, 5120>>(p, output, epilogue,
                                                                           stream);
            return;
        }
        if (tokens <= 32) {
            launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 8, 2>, 5120>>(p, output, epilogue,
                                                                           stream);
            return;
        }
        if (tokens <= 64) {
            launch_nvfp4_a16_mma<
                Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>, 5120>>(
                p, output, epilogue, stream);
            return;
        }
        if (tokens <= 96) {
            launch_nvfp4_a16_mma<
                Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>, 5120>>(
                p, output, epilogue, stream);
            return;
        }
        launch_nvfp4_a16_mma<
            Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 1, 3>, 5120>>(
            p, output, epilogue, stream);
    };
    if (whole) {
        launch(Nvfp4GdnInputOutput{static_cast<__nv_bfloat16*>(qkv.data),
                                   static_cast<__nv_bfloat16*>(z.data)});
    } else {
        launch(Nvfp4GdnInputShareOutput{static_cast<__nv_bfloat16*>(qkv.data),
                                        static_cast<__nv_bfloat16*>(z.data), share});
    }
}
} // namespace ninfer::ops::detail::unified

#pragma once

#include "ninfer/engine_options.h"
#include "ninfer/model_enums.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

namespace ninfer::models {

struct LoadOptions {
    EnginePurpose purpose          = EnginePurpose::Generation;
    bool vision                    = false;
    SpeculativeBackend speculative = SpeculativeBackend::None;
    // --spec dspine: the standalone drafter artifact, loaded beside the model's own.
    std::filesystem::path draft_model;
    ProposalHead proposal_head     = ProposalHead::Full;
    // Device-memory trades that store a stored Q8 vocabulary matrix (or the MoE MTP layer's routed
    // experts) at a narrower group-64 width while the weights load; the artifact is unchanged.
    bool lm_head_q4     = false;
    bool lm_head_q6     = false;
    bool embedding_q4   = false;
    bool embedding_q6   = false;
    bool mtp_experts_q4 = false;
    // Host keeps the token embedding in mapped host RAM, read in place by the gather.
    EmbeddingResidency embedding_residency = EmbeddingResidency::Device;
    ExpertResidency expert_residency = ExpertResidency::Device;
    // Execution selection carried with the instance: admit the integer small-T gate_up route at
    // verify widths (see execution::DenseParameters::verify_gate_up_policy).
    bool mlp_a8_decode = false;
    // Clearing this returns full prefill tiles to their A16 routes; see EngineOptions::prefill_a8.
    bool prefill_a8 = true;
    // Admit the NVFP4 W4A8 (int8-activation) family; see EngineOptions::nvfp4_w4a8_int.
    bool nvfp4_w4a8_int = false;
    // Prefer the dequantising NVFP4 A16 routes over a native-FP4 card's A4 permission; see
    // EngineOptions::nvfp4_w4a16.
    bool nvfp4_w4a16 = false;
    // See EngineOptions::prefill_cublas: hands wide prefill GEMMs to cuBLAS, a further quality
    // trade on top of prefill_a8 and therefore off unless asked for.
    bool prefill_cublas = false;
    // See EngineOptions::prefill_cublas_projections.
    bool prefill_cublas_projections = true;
    // Store the GDN recurrent state in FP16 between steps; the recurrence still computes in FP32.
    // Free within measurement noise for a ~2% C8 decode gain and a halved per-slot host state
    // image (docs/maintainer/quality-trade-experiments.md).
    bool gdn_state_fp16 = false;
    // Largest merged-token count one media item may occupy; larger media is downscaled at
    // preprocessing, and the Vision workspace is planned for this bound.
    std::uint32_t vision_max_merged_tokens = 16384;
    // Overlay keeps the Vision tower in pinned Host memory and places the output head, token
    // embedding, proposal head and MTP layer in an evictable device tail that one encode window
    // may borrow. Resident whenever Vision is disabled.
    VisionResidency vision_residency = VisionResidency::Resident;
    // How many devices the instance spans (`--devices`). One is the single-GPU route; more split
    // the layers into that many stages, each owning its layers' weights, KV cache and recurrent
    // state. The embedding, head and round state stay on the first device.
    std::size_t ranks = 1;
    // Layers per stage (`--stage-layers`), one count per device. Empty lets the engine choose.
    std::vector<std::uint32_t> stage_layers;
    // The `ranks` devices share every layer instead (`--tensor-parallel`): each holds a share of
    // the GDN heads and dense FFN rows or MoE experts, and the first one the rest.
    bool tensor_parallel = false;
    // The relative size of each rank's share (`--tensor-split`), one positive weight per device.
    // Empty divides evenly; the engine fills it from each card's speed and free memory.
    std::vector<double> tensor_split;
    // How the ranks divide a full-attention layer (`--tensor-attention`).
    TensorAttention tensor_attention = TensorAttention::Heads;
    // How the ranks exchange their partial sums (`--tensor-parallel-transport`).
    TensorParallelTransport tensor_parallel_transport = TensorParallelTransport::Auto;
    // How a prompt chunk's partial sums cross between the devices
    // (`--tensor-parallel-prompt-partials`).
    TensorParallelPartials tensor_parallel_prompt_partials = TensorParallelPartials::Fp32;

    bool operator==(const LoadOptions&) const = default;

    // The tensor-parallel group size, one without tensor parallelism.
    [[nodiscard]] std::size_t tensor_ranks() const noexcept { return tensor_parallel ? ranks : 1; }

    [[nodiscard]] bool speculative_enabled() const noexcept {
        return speculative != SpeculativeBackend::None;
    }

    [[nodiscard]] bool overlay_vision() const noexcept {
        return vision && vision_residency == VisionResidency::Overlay;
    }

    [[nodiscard]] bool cpu_vision() const noexcept {
        return vision && vision_residency == VisionResidency::Cpu;
    }

    // The tower's embeddings reach prefill from host memory, a chunk's columns at a time.
    [[nodiscard]] bool host_staged_vision() const noexcept {
        return overlay_vision() || cpu_vision();
    }

    [[nodiscard]] bool mtp() const noexcept { return speculative == SpeculativeBackend::Mtp; }

    [[nodiscard]] bool dflash() const noexcept { return speculative == SpeculativeBackend::DFlash; }

    [[nodiscard]] bool dflash2() const noexcept {
        return speculative == SpeculativeBackend::DFlash2;
    }

    [[nodiscard]] bool dspine() const noexcept {
        return speculative == SpeculativeBackend::DSpine;
    }

    [[nodiscard]] bool masked_draft() const noexcept { return dflash() || dflash2() || dspine(); }

    [[nodiscard]] bool proposal_enabled() const noexcept {
        return purpose == EnginePurpose::Generation && speculative != SpeculativeBackend::None &&
               proposal_head == ProposalHead::Optimized;
    }

    [[nodiscard]] std::string_view speculative_component() const noexcept {
        switch (speculative) {
        case SpeculativeBackend::None:
            return {};
        case SpeculativeBackend::Mtp:
            return "mtp";
        case SpeculativeBackend::DFlash:
            return "dflash";
        case SpeculativeBackend::DFlash2:
            return "dflash2";
        case SpeculativeBackend::DSpark:
            return "dspark";
        case SpeculativeBackend::DSpine:
            return "dspine";
        }
        return {};
    }
};

[[nodiscard]] constexpr bool is_masked_draft_backend(SpeculativeBackend backend) noexcept {
    return backend == SpeculativeBackend::DFlash || backend == SpeculativeBackend::DFlash2 ||
           backend == SpeculativeBackend::DSpine;
}

// A masked drafter whose round verifies a sparse law of 16 candidates a draft (DFlash2's frame).
[[nodiscard]] constexpr bool is_sparse_draft_backend(SpeculativeBackend backend) noexcept {
    return backend == SpeculativeBackend::DFlash2 || backend == SpeculativeBackend::DSpine;
}

// Device state images a Qwen3.5-core Program with the context catalog keeps beside the catalog's,
// one for each concurrent request: the state at the token where a stop string's match began,
// which a turn that ends at that stop string returns to. The hybrid prefix cache keeps that state
// in a snapshot slot.
[[nodiscard]] inline std::uint32_t stop_hold_slots(const ContextCacheOptions& cache,
                                                   std::uint32_t concurrency) noexcept {
    return cache.enabled && cache.mode == ContextCacheMode::Legacy && cache.stop_hold_images
               ? concurrency
               : 0U;
}

// The draft head a plan uses: an automatic head not yet resolved against an artifact
// (runtime::construct_model resolves it before loading) is the full one.
[[nodiscard]] constexpr ProposalHead planned_proposal_head(ProposalHead head) noexcept {
    return head == ProposalHead::Optimized ? ProposalHead::Optimized : ProposalHead::Full;
}

[[nodiscard]] inline LoadOptions load_options(const EngineOptions& options) noexcept {
    return {.purpose        = options.purpose,
            .vision         = options.enable_vision,
            .speculative    = options.speculative.backend,
            .draft_model    = options.speculative.draft_model,
            .proposal_head  = planned_proposal_head(options.speculative.proposal_head),
            .lm_head_q4     = options.lm_head_q4,
            .lm_head_q6     = options.lm_head_q6,
            .embedding_q4   = options.embedding_q4,
            .embedding_q6   = options.embedding_q6,
            .mtp_experts_q4 = options.mtp_experts_q4,
            .embedding_residency = options.embedding_residency,
            .expert_residency = options.expert_residency,
            .mlp_a8_decode  = options.mlp_a8_decode,
            .prefill_a8     = options.prefill_a8,
            .nvfp4_w4a8_int = options.nvfp4_w4a8_int,
            .nvfp4_w4a16    = options.nvfp4_w4a16,
            .prefill_cublas = options.prefill_cublas,
            .prefill_cublas_projections = options.prefill_cublas_projections,
            .gdn_state_fp16 = options.gdn_state_fp16,
            .vision_max_merged_tokens = options.vision_max_merged_tokens,
            .vision_residency         = options.enable_vision ? options.vision_residency
                                                              : VisionResidency::Resident,
            .ranks = std::max<std::size_t>(options.devices.size(), 1),
            .stage_layers = options.stage_layers,
            .tensor_parallel = options.tensor_parallel,
            .tensor_split    = options.tensor_split,
            .tensor_attention = options.tensor_attention,
            .tensor_parallel_transport = options.tensor_parallel_transport,
            .tensor_parallel_prompt_partials = options.tensor_parallel_prompt_partials};
}

} // namespace ninfer::models

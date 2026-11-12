#pragma once
#include "models/qwen3_5/program/internal.h"


#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/softmax_attention_envelope.h"
#include "ninfer/ops/sparse_moe.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "models/qwen3_5/execution/rotation.h"
#include "models/qwen3_5/execution/stage_runtime.h"
#include "models/qwen3_5/execution/steering.h"
#include "ninfer/ops/steer_residual.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

class NativeExpertRuntime;

using Phase = qwen3_5::TextPhase;

enum class GdnStateAction : std::uint8_t {
    UpdateInPlace,
    RecordForReplay,
};

struct NullTap {
    static constexpr bool enabled = false;
};

// An exact prefix-cache tap inside a prefill chunk (TextContext::set_gdn_taps): the state after
// the tokens before `frontier` goes to `state_slot`, and the final-norm hidden of the token before
// it to `hidden`.
struct GdnTap {
    std::uint32_t frontier = 0;
    std::int32_t state_slot = 0;
    Tensor* hidden          = nullptr;
};
inline constexpr std::size_t kMaximumGdnTaps = 4;

struct PrefillChunkResult {
    std::uint32_t processed_tokens = 0;
    bool finalized                 = false;
    // The chunk wrote the states of the in-pass taps (TextContext::set_gdn_taps).
    bool tapped = false;
    runtime::ExecutionTiming timing;
};

struct DFlashFeatureSink {
    static constexpr bool enabled = true;
    using PrefillConsumer         = std::function<void(const Tensor&, const Tensor&, bool)>;

    Tensor* features                  = nullptr;
    Tensor* positions                 = nullptr;
    Tensor* batch_features            = nullptr;
    const Tensor* batch_lanes         = nullptr;
    const Tensor* batch_valid_columns = nullptr;
    std::int32_t batch_width          = 0;
    std::int32_t batch_size           = 0;
    std::span<const std::uint32_t> layers;
    PrefillConsumer consume_prefill;
    std::uint32_t captured_mask = 0;
    std::int32_t active_tokens  = 0;

    void begin(const Tensor& value);
    void capture_layer(int layer, const Tensor& value, cudaStream_t stream);
    void capture_positions(const Tensor& source, cudaStream_t stream);
    void consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint);
};

class VisionPrefillSession;

// The most segments one batched prefill pass runs: a lane's last prompt chunk, or a branched
// read's prompt part and its branch blocks.
inline constexpr std::size_t kMaximumPrefillSegments = 64;

// What a batched prefill segment produces at its last column.
enum class SegmentOutput : std::uint8_t {
    // Samples its next token as a finalizing chunk does, with its logprob record and readout.
    Sample,
    // A branch block: the greedy token, its logprob record and readout, without the sampler.
    Read,
    // A branched read's prompt part: nothing.
    None,
};

// One segment of a batched prefill pass (TextContext::prefill_final_batch).
struct PrefillSegment {
    // The segment's tokens: a lane's prompt up to its frontier, or one branch's own tokens.
    std::span<const int> ids;
    // The absolute position of ids[0]; a branch continues its read's prompt frontier.
    std::uint32_t begin                 = 0;
    std::int32_t kv_table_row           = 0;
    std::int32_t state_source_slot      = 0;
    // Negative for a branch block: its GDN state goes to a transient buffer per layer and is not
    // kept, and its K/V rows past the prompt frontier are scratch the next block overwrites.
    std::int32_t state_destination_slot = 0;
    SegmentOutput output                = SegmentOutput::Sample;
    // The device sampling record; Read takes its logprob adjustments from it, None ignores it.
    const ops::SamplingConfig* sampling = nullptr;
    // The typed readout's token ids; empty reads nothing.
    std::span<const std::int32_t> readout;
};

// Host destinations of a batched prefill, one row per segment with output, in segment order: its
// token, kLogprobTopK logprob ids and values (written when logprobs are armed), and
// kReadoutLogprobsMaximum readout values, of which a row's first readout.size() are written.
struct PrefillBatchHost {
    std::int32_t* tokens  = nullptr;
    std::int32_t* top_ids = nullptr;
    float* top_values     = nullptr;
    float* readout        = nullptr;
};

class TextContext {
public:
    TextContext(DeviceContext& ctx, const execution::Parameters& weights, WorkspaceArena& work,
                qwen3_5::PagedKVCacheView kv, LinearAttentionStatePool& state,
                qwen3_5::RoundState& io, Tensor& prefill_hidden, std::uint32_t prefill_chunk,
                std::uint32_t text_kv_base,
                qwen3_5::PagedKVCacheView mtp_kv           = qwen3_5::PagedKVCacheView(),
                const qwen3_5::PagedKVCache* batch_text_kv = nullptr,
                const qwen3_5::PagedKVCache* batch_mtp_kv  = nullptr);
    ~TextContext();

    void set_expert_runtime(NativeExpertRuntime* experts) noexcept { experts_ = experts; }

    TextContext(const TextContext&)            = delete;
    TextContext& operator=(const TextContext&) = delete;

    // `shares` divides `weight` over the tensor-parallel ranks (see project_head), or nullptr.
    void set_proposal_head(const LinearParameters* weight, const std::int32_t* ids, int count,
                           const HeadShares* shares = nullptr) noexcept {
        proposal_head_        = weight;
        proposal_head_ids_    = ids;
        proposal_head_n_      = count;
        proposal_head_shares_ = shares;
    }

    void set_sampling(const ops::SamplingConfig* config) noexcept { sampling_config_ = config; }

    // Proposals drawn from the drafter's own distribution instead of its argmax: `configs` holds a
    // device SamplingConfig per proposal column (runtime::draft_sampling; a greedy row keeps the
    // argmax). Null keeps every proposal an argmax.
    void set_proposal_sampling(const ops::SamplingConfig* configs) noexcept {
        proposal_configs_ = configs;
    }

    // The next proposals' keys: column b draws with (seed, positions[b] + offset,
    // kSamplePurposeDecode), the key of the target's own draw for the position it proposes, so the
    // two draws agree wherever the distributions do. `positions` is a device I32 [columns].
    void set_proposal_key(const Tensor& positions, std::int32_t offset) noexcept {
        proposal_key_positions_ = positions;
        proposal_key_offset_    = offset;
    }

    // The typed readout a finalizing chunk performs on its sampled token's logits: the first
    // `count` ids of io.readout_ids, read into io.readout_values; zero reads nothing.
    void set_readout(std::int32_t count) noexcept { readout_count_ = count; }

    // Runs right after a finalizing chunk samples its token, before the drafter's work on the
    // chunk: it enqueues the token's copies to the host. The chunk then returns without waiting for
    // that work, which the stream orders before anything later.
    void set_first_token_hook(std::function<void()> hook) { first_token_hook_ = std::move(hook); }

    void set_rope_yarn(const ops::RopeYarn& yarn) noexcept { rope_yarn_ = yarn; }

    // Keys before its first query that the MTP draft attention reads past its one-page sink;
    // zero reads the whole history.
    void set_mtp_attention_window(std::uint32_t keys) noexcept { mtp_attention_window_ = keys; }

    // Present when the model is split over several devices; see StageRuntime.
    void set_stage_runtime(StageRuntime* runtime) noexcept { stage_runtime_ = runtime; }

    // Present when LoRA adapters or control vectors are loaded: the text layers then edit the
    // residual after the embedding and after each write to it (ops::steer_residual).
    void set_steering(const SteeringView* steering) noexcept { steering_ = steering; }

    void set_prefill_split_frontier(std::int64_t position) noexcept {
        prefill_split_frontier_ = position;
    }

    // Route prefill prompt attention through the fast INT8-KV prompt kernel.
    void set_fast_prefill_kernel(bool enabled) noexcept { fast_prefill_kernel_ = enabled; }

    void set_rewrite_checkpoint_hidden_output(Tensor* output) noexcept {
        rewrite_checkpoint_hidden_output_ = output;
    }

    // Lets a prompt chunk run the model's last layer as K/V only when nothing reads the chunk's
    // final hidden state: it samples no token, ends before the prompt does, and feeds no MTP
    // prompt, in-pass tap hidden, checkpoint hidden or drafter feature of that layer. The layer
    // then appends its K/V rows and skips its attention, output projection and FFN; every cache
    // byte stays as the whole layer writes it. A caller that reads prefill_hidden after such a
    // chunk (causal scoring, forced tokens) leaves it off.
    void set_last_layer_kv_only(bool allowed) noexcept { last_layer_kv_only_allowed_ = allowed; }

    // Exact prefix-cache taps inside the next chunk, in frontier order, without splitting the
    // pass: each Gated DeltaNet layer runs its convolution and recurrence up to each frontier into
    // that tap's slot and continues from there. A text chunk on one device whose columns hold
    // every frontier takes them all; otherwise it takes none (PrefillChunkResult::tapped).
    void set_gdn_taps(std::span<const GdnTap> taps) {
        if (taps.size() > kMaximumGdnTaps) {
            throw std::invalid_argument("TextContext: too many in-pass taps");
        }
        std::copy(taps.begin(), taps.end(), gdn_taps_.begin());
        gdn_tap_count_ = taps.size();
    }

    // One event per model layer: the next pass over the layer stack waits for each layer's event
    // before that layer's work (the layer's cached state is still being copied in).
    void set_layer_ready(std::span<const cudaEvent_t> events) noexcept { layer_ready_ = events; }

    void set_mtp_proposal_extent(std::uint32_t extent) noexcept { mtp_proposal_extent_ = extent; }

    void set_linear_state_slots(std::int32_t source_slot, std::int32_t destination_slot);
    void set_gdn_state_action(GdnStateAction action, const GdnReplayRecords* replay_records);

    [[nodiscard]] const LinearParameters* proposal_head() const noexcept { return proposal_head_; }

    [[nodiscard]] const std::int32_t* proposal_head_ids() const noexcept {
        return proposal_head_ids_;
    }

    [[nodiscard]] int proposal_head_n() const noexcept { return proposal_head_n_; }

    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end);
    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end, DFlashFeatureSink& sink);
    [[nodiscard]] PrefillChunkResult
    prefill_chunk(const qwen3_5::PreparedPromptData& input, std::uint32_t begin,
                  std::uint32_t nominal_length, VisionPrefillSession& vision, bool finalize_at_end);
    [[nodiscard]] PrefillChunkResult prefill_chunk(const qwen3_5::PreparedPromptData& input,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   VisionPrefillSession& vision,
                                                   bool finalize_at_end, DFlashFeatureSink& sink);
    // The last chunks of several requests' text prompts, and branched reads' blocks, in one pass
    // over the layer stack. Token-wise work runs over all columns at once; segment by segment, in
    // order, attention appends and reads the segment's KV row and each GDN layer advances its
    // state, as prefill_chunk would for that segment alone. A branch block follows its read's
    // prompt part: it shares the read's KV row past the prompt frontier and starts each GDN layer
    // from the state the prompt part left, so it sees the prompt and itself only. Segments with
    // output then read their last column: Sample samples as a finalizing chunk does (keyed by its
    // frontier), Read takes the greedy token; both gather the logprob record and the readout.
    // `host` receives every output row before this returns; prefill_hidden holds the final-norm
    // hidden of every column, segment after segment. No drafter work or pipeline stages.
    [[nodiscard]] PrefillChunkResult prefill_final_batch(std::span<const PrefillSegment> segments,
                                                         PrefillBatchHost host);
    void ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                               const Tensor& rope_positions, const Tensor& kv_table_rows,
                               const Tensor& linear_state_source_slots,
                               const Tensor& linear_state_destination_slots,
                               ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                               Tensor& logits);
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_source_slots,
                             ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                             Tensor& logits, Tensor& target_tokens);
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_source_slots,
                             ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                             Tensor& logits, Tensor& target_tokens, DFlashFeatureSink& sink);
    void mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                  const Tensor& cache_positions, const Tensor& rope_positions,
                                  const Tensor& valid_columns, const Tensor& kv_table_rows,
                                  ops::CausalAttentionExecutionEnvelope envelope,
                                  Tensor& mtp_hidden);
    void mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens);
    void mtp_forward_batch(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                           ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden,
                           int logits_column, Tensor* logits, Tensor* draft_token,
                           const Tensor* explicit_rope_positions = nullptr,
                           const Tensor* input_embeddings        = nullptr);
    void mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                             const Tensor& position, ops::CausalAttentionExecutionEnvelope envelope,
                             Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token);
private:
    [[nodiscard]] bool mtp_enabled() const noexcept {
        return mtp_kv_.valid() || batch_mtp_kv_ != nullptr;
    }

    // `kv_only` appends the layer's K/V rows and returns before its attention output.
    void attn_mix(const BlockParameters& weights, Tensor& x, int index, Phase phase,
                  bool kv_only = false);
    // Position-split attention's second half on a rank: the merge of every rank's partial for its
    // query heads, gated by the gates attn_mix kept, and its share of the output projection.
    void attn_merge_output(const BlockParameters& weights, Tensor& x,
                           std::span<const float* const> partials, const Tensor& gate,
                           Phase phase);
    void gdn_mix(const BlockParameters& weights, Tensor& x, int index, Phase phase);
    void mlp_tail(const BlockParameters& weights, Tensor& x, Phase phase,
                  const ops::SparseMoeHints& hints);
    // The main model's token embedding of `ids` into x, with the adapters' edit of it.
    void embed_input(const Tensor& ids, Tensor& x, cudaStream_t stream);
    // One steering step over x on the rank the context is bound to: `site` names the adapters'
    // factors (frontend/steering.h); `layer` adds that layer's control vector to a Write, -1 none.
    void steer(ops::SteerResidualStep step, std::uint32_t site, std::int32_t layer, Tensor& x);
    [[nodiscard]] ops::SparseMoeHints next_projection_hints(int layer) const;

    // Per-request width of a bound sequence batch; an unbatched call is one row of all columns.
    [[nodiscard]] std::int32_t row_width(std::int32_t columns) const noexcept {
        return active_sequence_batch_ != 0 ? active_sequence_width_ : columns;
    }
    void run_layers(Tensor& x, Phase phase);
    template <class Tap>
    void run_layers(Tensor& x, Phase phase, Tap& tap);
    // The layers one stage owns, on the device that stage's context is currently bound to.
    template <class Tap>
    void run_stage_layers(std::size_t stage, Tensor& x, Phase phase, Tap& tap);
    // The whole layer stack across pipeline stages, ending with the residual back on rank 0. A tap
    // sees all of its layers on rank 0; a later stage's tapped outputs cross after the residual.
    template <class Tap>
    void run_staged(Tensor& x, Phase phase, Tap& tap);
    // The whole layer stack with every rank running its share of each layer; the residual ends on
    // rank 0 as on one device.
    template <class Tap>
    void run_tensor_parallel(Tensor& x, Phase phase, Tap& tap);
    // The GDN and FFN geometry of the rank a pass is issuing: a tensor-parallel share, or the model.
    [[nodiscard]] const TextConfig& layer_config() const noexcept {
        return stage_runtime_ != nullptr && stage_runtime_->tensor_parallel
                   ? stage_runtime_->tensor_parallel->shards[tensor_rank_]
                   : config_;
    }
    // A mixer's output projection: added to the residual, or in a tensor-parallel pass written to
    // the rank's FP32 partial, which the exchange adds.
    void project_residual(const Tensor& input, const LinearParameters& p, Tensor& x,
                          InputBasis basis, bool wide_verification);
    // The Linear Attention state pool holding GDN layer `layer`, and its index within that pool.
    struct GdnStateRef {
        LinearAttentionStatePool* pool;
        std::uint32_t local;
    };
    [[nodiscard]] GdnStateRef gdn_state(std::uint32_t layer) const;
    // The shard holding GDN layer `layer` and the layer's index within it.
    struct GdnShard {
        std::size_t shard;
        std::uint32_t local;
    };
    [[nodiscard]] GdnShard gdn_shard(std::uint32_t layer) const;
    // ReplaySSM record storage for GDN layer `layer`, from the shard that holds it.
    [[nodiscard]] GdnReplayRecordLayer replay_layer(std::uint32_t layer, std::int32_t batch,
                                                    std::int32_t width) const;
    template <class Tap>
    void target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                  const Tensor& rope_positions, const Tensor& valid_columns,
                                  const Tensor& kv_table_rows,
                                  const Tensor& linear_state_source_slots,
                                  ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                                  Tensor& logits, Tensor& target_tokens, Tap& tap);

    // `ah`, where given, takes the attention's input; a divided attention normalizes on each rank.
    void mtp_forward_stem(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                          Tensor& x, Tensor* ah);
    void mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                          const Tensor& rope_positions,
                          ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden);
    // Adds the MTP layer's FFN of `mh` [hidden, T] to its residual `x`. Under tensor parallelism
    // with the FFN divided (MtpParameters::rank_ffn), every rank projects its rows of it.
    void mtp_ffn(const Tensor& mh, Tensor& x);
    // Under tensor parallelism with the attention divided (MtpParameters::rank_attention): the MTP
    // layer after its stem output `x` [hidden, T], over its last `columns` columns (none: only the
    // attention's K/V). `control` are rank 0's int32 tensors the attention reads, which every rank
    // receives; `attend(rank, share, ah, control)` runs on the rank's device and returns its gated
    // attention output [its query width, columns] for input `ah`.
    template <class Attend>
    void mtp_divided_layer(Tensor& x, std::span<const Tensor* const> control,
                           std::int32_t columns, const Attend& attend);
    // The divided keys and values of an MTP prompt chunk `ah` [hidden, T] (mtp_prefill_chunk).
    void mtp_divided_prefill_kv(const Tensor& ah, const Tensor& positions,
                                const Tensor& rope_positions);
    // The divided decode attention of mtp_forward_tail.
    void mtp_divided_tail(Tensor& x, const Tensor& positions, const Tensor& rope_positions,
                          ops::CausalAttentionExecutionEnvelope envelope);
    void window_mtp_attention(PagedKVBatchLayerView& view, Tensor& table_rows, Tensor& positions,
                              const Tensor& valid_columns);
    void mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                          const Tensor& rope_positions,
                          ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden,
                          const Tensor* input_embeddings);
    void mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                           const Tensor& positions, const Tensor& rope_positions,
                           ops::CausalAttentionExecutionEnvelope envelope, bool final_chunk,
                           Tensor* final_hidden, Tensor* logits, Tensor* draft_token);
    void proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens);
    // Projects `hidden` [hidden, columns] through the vocabulary head `whole` into `logits`
    // [rows, columns] (a column window of a wider buffer is fine). Under tensor parallelism with
    // `shares` divided, every rank projects its own rows of the head instead and rank 0 gathers
    // them into `logits`.
    void project_head(const Tensor& hidden, const LinearParameters& whole,
                      const HeadShares* shares, Tensor& logits);

    struct MultimodalPrefill {
        std::span<const int> token_ids;
        std::span<const std::int32_t> positions;
        VisionPrefillSession* vision = nullptr;
        std::uint32_t begin          = 0;
        std::int32_t rope_delta      = 0;
    };

    struct TextPrefill {
        std::span<const int> token_ids;
        std::uint32_t begin = 0;
    };

    template <class Tap>
    [[nodiscard]] PrefillChunkResult
    prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                 const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end);

    // A batched prefill segment's columns and sequence bindings within one pass. A negative
    // destination slot leaves the segment's GDN state in a transient buffer (a branch block).
    struct SegmentBinding {
        std::int32_t column  = 0;
        std::int32_t columns = 0;
        Tensor positions;
        Tensor kv_table_row;
        ops::CausalAttentionExecutionEnvelope envelope;
        std::int32_t state_source_slot      = 0;
        std::int32_t state_destination_slot = 0;
    };

    DeviceContext& ctx_;
    const Parameters& parameters_;
    NativeExpertRuntime* experts_ = nullptr;
    const TextConfig& config_;
    WorkspaceArena& work_;
    qwen3_5::PagedKVCacheView kv_;
    qwen3_5::PagedKVCacheView mtp_kv_;
    const qwen3_5::PagedKVCache* batch_text_kv_ = nullptr;
    const qwen3_5::PagedKVCache* batch_mtp_kv_  = nullptr;
    LinearAttentionStatePool& state_;
    qwen3_5::RoundState& io_;
    Tensor& prefill_hidden_;
    std::uint32_t prefill_chunk_;
    bool fast_prefill_kernel_ = false;
    std::uint32_t text_kv_base_;
    const Tensor* active_cache_positions_                                          = nullptr;
    const Tensor* active_rope_positions_                                           = nullptr;
    const Tensor* active_kv_table_rows_                                            = nullptr;
    const Tensor* active_linear_state_source_slots_                                = nullptr;
    const Tensor* active_linear_state_destination_slots_                           = nullptr;
    const Tensor* active_valid_columns_                                            = nullptr;
    const Tensor* active_backend_kv_table_rows_                                    = nullptr;
    const ops::CausalAttentionExecutionEnvelope* active_causal_attention_envelope_ = nullptr;
    std::int32_t active_sequence_batch_                                            = 0;
    std::int32_t active_sequence_width_                                            = 0;
    // Non-empty during prefill_final_batch: a prefill pass's sequences, column range by range.
    std::span<const SegmentBinding> active_segments_;
    // During prefill_final_batch: each column's KV table row, the lane its steering scales are read
    // from.
    const Tensor* active_column_rows_                                              = nullptr;
    std::int32_t rope_delta_                                                       = 0;
    std::int32_t linear_state_source_slot_                                         = 0;
    std::int32_t linear_state_destination_slot_                                    = 0;
    GdnStateAction gdn_state_action_          = GdnStateAction::UpdateInPlace;
    const GdnReplayRecords* replay_records_   = nullptr;
    std::int64_t prefill_split_frontier_      = -1;
    Tensor* rewrite_checkpoint_hidden_output_ = nullptr;
    bool last_layer_kv_only_allowed_          = false;
    // The running prefill pass runs the model's last layer as K/V only.
    bool kv_only_pass_ = false;
    std::array<GdnTap, kMaximumGdnTaps> gdn_taps_{};
    std::size_t gdn_tap_count_ = 0;
    // The running chunk's tap columns; gdn_live_taps_ == 0 runs none.
    std::array<std::int32_t, kMaximumGdnTaps> gdn_tap_columns_{};
    std::size_t gdn_live_taps_ = 0;
    std::uint32_t mtp_proposal_extent_        = 0;
    std::span<const cudaEvent_t> layer_ready_;

    const Weight* embed_                        = nullptr;
    const Tensor* final_norm_                   = nullptr;
    const LinearParameters* lm_head_            = nullptr;
    const LinearParameters* proposal_head_      = nullptr;
    const HeadShares* proposal_head_shares_     = nullptr;
    const std::int32_t* proposal_head_ids_      = nullptr;
    int proposal_head_n_                        = 0;
    const ops::SamplingConfig* sampling_config_ = nullptr;
    const ops::SamplingConfig* proposal_configs_ = nullptr;
    Tensor proposal_key_positions_;
    std::int32_t proposal_key_offset_           = 0;
    std::int32_t readout_count_                 = 0;
    std::function<void()> first_token_hook_;
    ops::RopeYarn rope_yarn_;
    std::uint32_t mtp_attention_window_         = 0;
    const MtpParameters* mtp_                   = nullptr;
    StageRuntime* stage_runtime_                = nullptr;
    const SteeringView* steering_               = nullptr;
    // A tensor-parallel pass: the rank whose share is being issued, and where its output
    // projections write their FP32 partial instead of adding to the residual.
    std::size_t tensor_rank_  = 0;
    Tensor* residual_partial_ = nullptr;
    // A position-split attention pass: where the rank's attention writes its FP32 partial output
    // (ops::causal_attention_partial_elements) and where it keeps its query heads' gates [256,
    // heads, T] for attn_merge_output, in place of the output projection.
    float* attention_partial_ = nullptr;
    Tensor* attention_gate_   = nullptr;
};

} // namespace ninfer::models::qwen3_5::execution

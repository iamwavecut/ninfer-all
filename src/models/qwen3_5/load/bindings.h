#pragma once

#include "artifact/binder.h"
#include "core/qtype.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/expert_sources.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/weights.h"
#include "ops/common/device_tier.h"

#include <map>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::loading {

[[nodiscard]] FrontendResources bind_resources(artifact::Binder& binder, const Config& config);

struct PendingWeight {
    artifact::ParameterReference reference;
    std::vector<WeightUse> uses;
    std::vector<std::string> source_objects;
    std::size_t rank = 0; // see BoundWeight::rank
};

// The piece of a parameter one tensor-parallel rank holds: one range of a matrix's rows, or the
// given ranges of its columns (a vector's elements), side by side. A replica is the one range that
// covers every column.
struct ParameterPiece {
    enum class Cut : std::uint8_t { Rows, Columns };
    Cut cut = Cut::Rows;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
};

class Bindings {
public:
    explicit Bindings(artifact::Binder& binder, const LoadOptions& options = {})
        : binder(binder), expert_residency(options.expert_residency),
          resident_mtp_experts(options.mtp_experts_q4) {}

    [[nodiscard]] WeightId parameter(std::string name, artifact::Shape shape,
                                     std::vector<std::string> inputs   = {},
                                     std::optional<QType> exact_format = {},
                                     artifact::Residency residency = artifact::Residency::Device);
    [[nodiscard]] WeightId direct(std::string name, artifact::Shape shape,
                                  QType format                  = QType::BF16,
                                  artifact::Residency residency = artifact::Residency::Device);

    // Binds `piece` of a parameter as a device weight of its own on `rank`, named `name@rank`
    // or `label`. The stored objects behind the parameter get no device demand from this.
    [[nodiscard]] WeightId piece(const std::string& name, const artifact::Shape& shape,
                                 const std::vector<std::string>& inputs,
                                 std::optional<QType> exact_format, std::size_t rank,
                                 const ParameterPiece& piece, const std::string& label = {});

    // A parameter that joint_piece binds on a rank.
    struct JointPart {
        std::string name;
        artifact::Shape shape;
        std::vector<std::string> inputs;
    };

    // Binds the matrix parameters `parts` on `rank`, each named `name@rank`, as views into one
    // device selection of the stored object behind them all: the run of its rows they cover
    // together, which they must cover exactly. Views stay side by side as the object holds them,
    // so an Op that reads several parameters as one bank (a rank's experts, or a replica of the
    // router and shared score) reads them there.
    [[nodiscard]] std::vector<WeightId> joint_piece(std::span<const JointPart> parts,
                                                    std::size_t rank);
    // The same for a piece of each parameter, in one device selection: rows [begin, end) of each
    // part's own rows (a row cut, one range), or the columns of `piece.ranges` of every part (a
    // column cut). Each view takes the piece's shape; a column cut refuses a rotated or gathered
    // input, whose columns a piece would have to carry. The parts lie in the object in order.
    [[nodiscard]] std::vector<WeightId> joint_piece(std::span<const JointPart> parts,
                                                    std::size_t rank, const ParameterPiece& piece);

    // Where a column piece of matrix parameter `name` (`shape`, read through `inputs`) may end:
    // out[c] for c in [0, columns] says whether the columns [0, c) are whole runs of its stored
    // bytes (artifact::selection_column_unit); with an input gather (`input_columns`), a run is
    // whole when all or none of the logical columns its stored columns read lie below c.
    [[nodiscard]] std::vector<bool> column_ends(const std::string& name, const artifact::Shape& shape,
                                                const std::vector<std::string>& inputs);

    // Where a row piece of matrix parameter `name` may end: out[r] for r in [0, rows] says whether
    // rows [0, r) end on a row bound of the stored object holding row r
    // (artifact::selection_row_unit; NVFP4 scale tiles), or at a stored object's end.
    [[nodiscard]] std::vector<bool> row_ends(const std::string& name, const artifact::Shape& shape);

    [[nodiscard]] const PendingWeight& at(WeightId id) const { return weights.at(id.index); }

    [[nodiscard]] WeightUseId use(WeightId id, std::string_view input) const;

    // Requests that every stored object behind a parameter be materialized in `target` instead
    // (see artifact/transcode.h). Returns false, leaving the objects alone, when they already
    // store `target`; throws std::invalid_argument when they are not transcodable Q8.
    bool transcode(WeightId id, QType target, std::string_view option);
    // Places every stored object behind a device parameter in the evictable device tail.
    void evict(WeightId id, std::uint32_t rank);
    // Materializes every stored object behind a device parameter on another pipeline rank's
    // device. Rank 0 is the default and this is then a no-op, so the single-device path is
    // untouched.
    void place(WeightId id, std::size_t rank);

    artifact::Binder& binder;
    ExpertResidency expert_residency;
    bool resident_mtp_experts;
    std::vector<PendingWeight> weights;

private:
    // `whole_gathers` false leaves an input gather out, and `whole_signs` false the Hadamard signs,
    // for a piece that binds its own.
    void add_uses(PendingWeight& pending, const std::string& name,
                  const std::vector<std::string>& inputs, bool whole_gathers = true,
                  bool whole_signs = true);
    // One device weight per distinct sign binding, however many Uses name it.
    WeightId hadamard_signs(const artifact::Binding& binding, std::uint64_t width,
                            const std::string& use);
    // One device weight per distinct input-gather binding, the same way.
    WeightId input_columns(const artifact::Binding& binding, std::uint64_t width,
                           const std::string& use);
    // A tensor-parallel piece's gather: the entries of `gather` at the stored `columns`, less
    // `base`, on `rank`.
    WeightId gather_piece(const artifact::Binding& gather,
                          const std::vector<std::pair<std::uint64_t, std::uint64_t>>& columns,
                          std::uint64_t base, std::size_t rank, const std::string& use);

    std::map<std::string, WeightId, std::less<>> parameters_;
    std::map<std::string, WeightId, std::less<>> signs_;
    std::map<std::string, WeightId, std::less<>> columns_;
    std::map<std::string, WeightId, std::less<>> gather_pieces_;
};

[[nodiscard]] AttentionWeights bind_attention(Bindings& bindings, const TextConfig& config,
                                              const std::string& prefix);
[[nodiscard]] DenseWeights bind_dense(Bindings& bindings, std::uint64_t hidden,
                                      std::uint64_t intermediate, const std::string& prefix,
                                      bool draft = false);
// `ffn` and `attention`, where given, are the block's FFN and attention already bound (a
// tensor-parallel rank 0's shares) in place of the whole ones.
[[nodiscard]] BlockWeights bind_block(Bindings& bindings, const TextConfig& config,
                                      const std::string& prefix, MixerKind mixer,
                                      const DenseWeights* ffn             = nullptr,
                                      const AttentionWeights* attention = nullptr);
// `stages` decides which device holds each whole layer; its identity form (one stage) binds exactly
// what a single-GPU load always bound.
[[nodiscard]] TextWeights bind_text(Bindings& bindings, const TextConfig& config,
                                    const LoadOptions& options, StagePlan stages);
// The stage plan `options.ranks` devices ask for over `layers` layers: `options.stage_layers` when
// given, equal counts otherwise. Throws when the model cannot be served by that many devices.
[[nodiscard]] StagePlan plan_stage_plan(std::uint32_t layers, const LoadOptions& options);
// Every parameter a whole layer owns: what moves with it when the layer is placed on a stage.
[[nodiscard]] std::vector<WeightId> layer_weights(const BlockWeights& block);
// Pinned residency keeps the tower in the page-locked Host block, one contiguous group per stage
// (patch/position embedding, each layer, merger) in binding order.
// `output_hidden` is the text model's hidden width, which the merger projects to.
[[nodiscard]] VisionWeights bind_vision(Bindings& bindings, const VisionConfig& config,
                                        std::uint64_t output_hidden,
                                        artifact::Residency residency);
// Every rank's share of a dense FFN (`shares`, one per rank) bound at `prefix`, or none where its
// stored runs do not cut at every share's end.
[[nodiscard]] std::vector<DenseWeights> bind_dense_shares(Bindings& bindings, std::uint64_t hidden,
                                                          std::uint64_t intermediate,
                                                          const std::string& prefix,
                                                          std::span<const TensorShare> shares);
// Every share of an attention mixer bound at `prefix` for the ranks holding KV heads (a prefix of
// `shares`, one per rank), or none where fewer than two ranks hold them, the ranks split positions
// instead, or its stored runs do not cut at every share's heads.
[[nodiscard]] std::vector<AttentionWeights>
bind_attention_shares(Bindings& bindings, const TextConfig& config, const std::string& prefix,
                      std::span<const TensorShare> shares);
// A copy of a norm for `rank`: rank 0 binds it as it is, another rank a piece that covers it.
[[nodiscard]] WeightId bind_norm_replica(Bindings& bindings, const std::string& name,
                                         std::uint64_t width, std::size_t rank);
[[nodiscard]] MtpWeights bind_mtp(Bindings& bindings, const TextConfig& config,
                                  const TextWeights& target);
[[nodiscard]] DraftWeights bind_draft(Bindings& bindings, const DraftConfig& config,
                                      const TextConfig& target, const TextWeights& weights,
                                      const std::string& component);
void bind_dflash2(Bindings& bindings, DraftWeights& weights, const DraftConfig& config,
                  const TextConfig& target);
[[nodiscard]] ProposalWeights bind_proposal(Bindings& bindings, const artifact::Proposal& proposal,
                                            const TextConfig& target, const LoadOptions& options,
                                            std::uint32_t public_tokens);
// --lm-head-q4/q6, --embedding-q4/q6 and --mtp-experts-q4: validate the combination and request
// the load-time transcodes on the selected parameters.
void apply_storage_trades(Bindings& bindings, const Config& config, const ModelWeights& weights,
                          const LoadOptions& options);
// The FP8 A8 and NVFP4 A4 permissions a stored weight keeps only on a rank whose device runs the
// 120a code (`rank_tiers`, one per rank): elsewhere its uses take the A16 routes (prepare.cpp).
void apply_rank_policies(Bindings& bindings, std::span<const ops::DeviceTier> rank_tiers);
// sm_70: the panel layout for the projection weights its routes read, on the ranks whose devices
// run Volta's code (`rank_tiers`, one per rank; storage_trades.cpp).
void apply_device_layouts(Bindings& bindings, const ModelWeights& weights,
                          std::span<const ops::DeviceTier> rank_tiers);
// --vision-residency overlay: rank the device weights an exclusive Vision window may borrow.
// `mtp_parameters` and `draft_parameters` are the half-open WeightId index ranges registered by
// bind_mtp and bind_draft.
void apply_vision_overlay_placement(Bindings& bindings, const ModelWeights& weights,
                                    std::pair<std::size_t, std::size_t> mtp_parameters,
                                    std::pair<std::size_t, std::size_t> draft_parameters);
// Byte ranges of the Vision groups inside the materialized pinned block.
[[nodiscard]] VisionOverlayLayout vision_overlay_layout(const VisionWeights& weights,
                                                        std::span<const BoundWeight> bound,
                                                        std::span<const std::byte> pinned_block);
[[nodiscard]] std::vector<BoundWeight>
resolve_weights(std::vector<PendingWeight>&& pending,
                const artifact::MaterializedArtifact& materialized);

[[nodiscard]] NativeExpertSources expert_sources(const artifact::Reader& reader,
                                                  std::span<const PendingWeight> pending,
                                                  ModelWeights& weights,
                                                  const artifact::MaterializedArtifact& backing,
                                                  const LoadOptions& options,
                                                  const std::filesystem::path& artifact_path);

} // namespace ninfer::models::qwen3_5::loading

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/rotation.h"
#include "core/qtype.h"

#include "core/weight_view.h"

#include <array>
#include <limits>
#include <type_traits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_5::execution {
namespace {

template <class Function>
auto with_context(const std::string& context, Function&& function) {
    try {
        return function();
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(context + ": " + error.what());
    }
}

class Prepare {
public:
    explicit Prepare(const Model& model) : model_(model) {}

    // The integer-activation route is an sm_86 addition: it feeds groupwise-int weights to the s8
    // tensor cores, which Ampere has and which no A16 route uses. It is registered per exact shape
    // rather than per format, because a registered route exists only for the profiles a kernel was
    // written for; any other projection keeps what its Use permits.
    void integer_route(LinearParameters& p, QType format, std::int32_t n, std::int32_t k) const {
#if defined(NINFER_SM8X_COMPAT)
        if (!model_.options().prefill_a8) { return; }
        if (p.policy == ops::LinearPolicy::A16Only && p.weight.qtype == format &&
            p.weight.n == n && p.weight.k == k) {
            // The cuBLAS route is a superset: it admits everything AllowA8Int does and adds the
            // materialise-and-call-cuBLAS path above its width gate, so the resolver still picks
            // the integer mainloop for narrow calls.
            p.policy = model_.options().prefill_cublas ? ops::LinearPolicy::AllowPrefillCublas
                                                       : ops::LinearPolicy::AllowA8Int;
        }
#else
        (void)p; (void)format; (void)n; (void)k;
#endif
    }

    // Split parents reach their Op as a pair rather than a SingleProjectionWeight, so the pair
    // carries the decision. Keyed on both parents' exact shapes, for the same reason.
    void integer_route_pair(ops::ProjectionWeights& projection, QType first_format,
                            std::int32_t first_rows, QType second_format, std::int32_t second_rows,
                            std::int32_t input_rows) const {
#if defined(NINFER_SM8X_COMPAT)
        if (!model_.options().prefill_a8) { return; }
        auto* pair = std::get_if<ops::PairedProjectionWeights>(&projection);
        if (pair == nullptr || pair->policy != ops::LinearPolicy::A16Only) { return; }
        if (pair->first.qtype == first_format && pair->first.n == first_rows &&
            pair->first.k == input_rows && pair->second.qtype == second_format &&
            pair->second.n == second_rows && pair->second.k == input_rows) {
            // The split projections are the separable half of the trade, so they take the cuBLAS
            // policy only when both settings ask for it.
            pair->policy = model_.options().prefill_cublas &&
                                   model_.options().prefill_cublas_projections
                               ? ops::LinearPolicy::AllowPrefillCublas
                               : ops::LinearPolicy::AllowA8Int;
        }
#else
        (void)projection; (void)first_format; (void)first_rows; (void)second_format;
        (void)second_rows; (void)input_rows;
#endif
    }

    // The NVFP4 int8 route is the W4A8 counterpart of the integer routes above: the stored E2M1
    // codes feed the s8 tensor cores with s8-quantised activations. --nvfp4-w4a8-int is its only
    // switch, and once registered the route serves every width (the wide mainloops pad their last
    // token tile), so prefill and decode are the same decision. Which wide mainloop answers follows
    // the route table: the own table runs MMQ, the unified table the grouped ladder.
    //
    // A card with native FP4 tensor cores stores AllowA4 for an NVFP4 weight, and there the flag
    // replaces that permission: W4A8 keeps activation precision the A4 route trades away, on an s8
    // MMA every sm_80+ part runs. Registered per shape like the integer routes, a tensor-parallel
    // rank's share of the shape included (nvfp4_shape_or_share).
    void nvfp4_int8_route(LinearParameters& p, std::int32_t n, std::int32_t k) const {
        if (!model_.options().nvfp4_w4a8_int || model_.options().nvfp4_w4a16) { return; }
        if (p.policy != ops::LinearPolicy::A16Only && p.policy != ops::LinearPolicy::AllowA4) {
            return;
        }
        if (!nvfp4_shape_or_share(p.weight, n, k)) { return; }
        // The family reads both dimensions at launch time and writes whichever dtype the result
        // names, so a share is a shape it serves. Its alignment is the launcher's: rows a whole
        // 128-tile, K a whole 64-group (four G16 blocks), which the loader's cuts give every share.
        if ((p.weight.n % 128) != 0 || (p.weight.k % 64) != 0) { return; }
        p.policy = ops::LinearPolicy::AllowNvfp4Int8;
    }

    // --nvfp4-w4a16 asks a native-FP4 card for the dequantising A16 routes instead of A4. On every
    // other card the permission was already lowered to A16Only while loading, so the flag is a
    // no-op there: it only ever trades A4 away. A rank's share takes the general A16 route, which
    // reads both dimensions at launch time.
    void nvfp4_w4a16_route(LinearParameters& p, std::int32_t n, std::int32_t k) const {
        if (!model_.options().nvfp4_w4a16) { return; }
        if (p.policy == ops::LinearPolicy::AllowA4 && nvfp4_shape_or_share(p.weight, n, k)) {
            p.policy = ops::LinearPolicy::A16Only;
        }
    }

    // The attention and GDN input projections reach their Op as one fused single parent on NVFP4,
    // or, for a tensor-parallel rank, as the parts its share is stored in. Both carriers hold the
    // policy the Op reads, so the flag promotes whichever the rank was bound as; a part is a
    // segment of the fused parent (whole KV heads, or the GDN key heads with their value heads), so
    // its rows are a whole 128-row tile rather than the parent's exact extent.
    void nvfp4_int8_route_single(ops::ProjectionWeights& projection, std::int32_t n,
                                 std::int32_t k) const {
        if (auto* single = std::get_if<ops::SingleProjectionWeight>(&projection)) {
            nvfp4_int8_route(*single, n, k);
            return;
        }
        auto* parts = std::get_if<ops::PartProjectionWeights>(&projection);
        if (parts == nullptr || parts->parts.empty()) { return; }
        if (!model_.options().nvfp4_w4a8_int || model_.options().nvfp4_w4a16) { return; }
        if (parts->policy != ops::LinearPolicy::A16Only &&
            parts->policy != ops::LinearPolicy::AllowA4) {
            return;
        }
        if (!nvfp4_parts(*parts, k)) { return; }
        parts->policy = ops::LinearPolicy::AllowNvfp4Int8;
    }

    void nvfp4_w4a16_route_single(ops::ProjectionWeights& projection, std::int32_t n,
                                  std::int32_t k) const {
        if (auto* single = std::get_if<ops::SingleProjectionWeight>(&projection)) {
            nvfp4_w4a16_route(*single, n, k);
            return;
        }
        auto* parts = std::get_if<ops::PartProjectionWeights>(&projection);
        if (parts == nullptr || parts->parts.empty() || !model_.options().nvfp4_w4a16) { return; }
        if (parts->policy == ops::LinearPolicy::AllowA4 && nvfp4_parts(*parts, k)) {
            parts->policy = ops::LinearPolicy::A16Only;
        }
    }

    // A tensor-parallel rank binds a projection at its share of the divided dimension: gate and up
    // rows, or the columns of down and the mixer outputs. The share need not divide the parent:
    // --tensor-split sizes the ranks by measured bandwidth.
    static bool nvfp4_shape_or_share(const Weight& w, std::int32_t n, std::int32_t k) {
        if (w.qtype != QType::NVFP4) { return false; }
        const bool whole        = w.n == n && w.k == k;
        const bool column_share = w.k == k && w.n > 0 && w.n < n;
        const bool row_share    = w.n == n && w.k > 0 && w.k < k;
        return whole || column_share || row_share;
    }

    // Every part of a rank's input projection is NVFP4 at the full input width, in whole 128-row
    // tiles; project_parts runs each through plain Linear under the parts' one policy.
    static bool nvfp4_parts(const ops::PartProjectionWeights& parts, std::int32_t k) {
        for (const auto& part : parts.parts) {
            if (part.weight.qtype != QType::NVFP4 || part.weight.k != k || part.weight.n <= 0 ||
                (part.weight.n % 128) != 0) {
                return false;
            }
        }
        return true;
    }

    LinearParameters linear(WeightId id) const {
        return with_context(model_.weight(id).name,
                            [&] { return ops::prepare_linear_weight(model_.input(id)); });
    }

    LinearParameters linear(WeightUseId id) const {
        return with_context(model_.weight(id.parameter).name,
                            [&] { return ops::prepare_linear_weight(model_.input(id)); });
    }

    // A ternary output or proposal head takes the integer route at decode, verify and draft widths
    // like the text projections (any row count of whole 64-row blocks at the hidden width).
    LinearParameters head(WeightUseId id) const {
        LinearParameters out = rotated_linear(id);
        integer_route(out, QType::T2_G128_FP16, out.weight.n, 5120);
        return out;
    }

    LinearParameters head(WeightId id) const {
        LinearParameters out = rotated_linear(id);
        integer_route(out, QType::T2_G128_FP16, out.weight.n, 5120);
        return out;
    }

    // Each rank's rows of a divided head, read as the whole head is: through the Uses when the
    // weights name them, else as the parameters themselves.
    HeadShares head_shares(const HeadShareWeights& w) const {
        HeadShares out;
        for (std::size_t rank = 0; rank < w.shares.size(); ++rank) {
            out.shares.push_back(w.uses.empty() ? head(w.shares[rank]) : head(w.uses[rank]));
        }
        out.rows.assign(w.rows.begin(), w.rows.end());
        return out;
    }

    // Sites whose execution rotates the activation of a Hadamard-rotated matrix.
    LinearParameters rotated_linear(WeightId id) const {
        return with_context(model_.weight(id).name,
                            [&] { return ops::prepare_linear_weight(model_.rotated_input(id)); });
    }

    LinearParameters rotated_linear(WeightUseId id) const {
        return with_context(model_.weight(id.parameter).name,
                            [&] { return ops::prepare_linear_weight(model_.rotated_input(id)); });
    }

    Tensor tensor(WeightId id) const {
        const auto& bound = model_.weight(id);
        return with_context(bound.name, [&] {
            const auto& view = bound.view;
            if (view.shape.size() > 4) {
                throw std::invalid_argument("direct parameter exceeds Tensor rank");
            }
            std::array<std::int32_t, 4> axes{1, 1, 1, 1};
            for (std::size_t i = 0; i < view.shape.size(); ++i) {
                const auto extent = view.shape[view.shape.size() - 1 - i];
                if (extent > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
                    throw std::invalid_argument("direct parameter exceeds Tensor extent");
                }
                axes[i] = static_cast<std::int32_t>(extent);
            }
            return weight_tensor(view, {axes[0], axes[1], axes[2], axes[3]});
        });
    }

    static bool gguf(const ops::WeightInput& input) {
        return !input.weight.parts.empty() && input.weight.parts.front().parent != nullptr &&
               is_gguf(input.weight.parts.front().parent->geometry.format);
    }

    // GGUF gate and up, joined or two parents; or a tensor-parallel rank's rows of gate and up,
    // two device objects of any format (ops::linear_swiglu's two-parent form).
    DenseParameters dense(const DenseWeights& w) const {
        const auto gate = model_.rotated_input(w.gate);
        const auto up   = model_.rotated_input(w.up);
        if (const std::array inputs{gate, up}; gguf(gate) || !ops::joins(inputs)) {
            return with_context(model_.weight(w.gate).name, [&] {
                DenseParameters out;
                if (ops::joins(inputs)) {
                    out.gate_up = ops::prepare_linear_swiglu_weight(gate, up);
                } else {
                    out.gate_up = ops::prepare_linear_weight(gate);
                    out.up      = ops::prepare_linear_weight(up);
                }
                out.down = rotated_linear(w.down);
                // A tensor-parallel rank holds gate and up as two parents: its share of the 17408
                // gate rows, and its share of down's 17408 columns. Promote on those parent shapes.
                // The two-parent form runs gate and up through plain Linear into unrounded FP32
                // planes under the gate's policy, so an int8 gate or up takes the family's FP32
                // writer, which is exactly what that plane is.
                nvfp4_w4a16_route(out.gate_up, 17408, 5120);
                nvfp4_w4a16_route(out.down, 5120, 17408);
                nvfp4_int8_route(out.gate_up, 17408, 5120);
                nvfp4_int8_route(out.down, 5120, 17408);
                out.verify_gate_up_policy = out.gate_up.policy;
                return out;
            });
        }
        DenseParameters out{with_context(model_.weight(w.gate).name,
                                         [&] { return ops::prepare_linear_swiglu_weight(gate, up); }),
                            rotated_linear(w.down)};
        integer_route(out.gate_up, QType::Q4_G64_FP16, 34816, 5120);
        // The Q4 down projection (Q4 imatrix recipes) reaches the same A8 add route as Q5.
        integer_route(out.down, QType::Q4_G64_FP16, 5120, 17408);
        integer_route(out.down, QType::Q5_G64_FP16, 5120, 17408);
        integer_route(out.gate_up, QType::T2_G128_FP16, 34816, 5120);
        integer_route(out.down, QType::T2_G128_FP16, 5120, 17408);
        nvfp4_w4a16_route(out.gate_up, 34816, 5120);
        nvfp4_w4a16_route(out.down, 5120, 17408);
        nvfp4_int8_route(out.gate_up, 34816, 5120);
        nvfp4_int8_route(out.down, 5120, 17408);
        // --mlp-a8-decode is a separate verify-phase trade from --prefill-a8: it must admit the
        // decode route by this profile's own format/shape, not by whether prefill promotion
        // already ran, so the two flags stay orthogonal as documented.
        out.verify_gate_up_policy = out.gate_up.policy;
#if defined(NINFER_SM8X_COMPAT)
        if (model_.options().mlp_a8_decode && out.gate_up.weight.qtype == QType::Q4_G64_FP16 &&
            out.gate_up.weight.n == 34816 && out.gate_up.weight.k == 5120) {
            out.verify_gate_up_policy = ops::LinearPolicy::AllowA8IntDecode;
        }
#endif
        return out;
    }

    FfnParameters ffn(const std::variant<DenseWeights, MoeWeights>& w) const {
        if (const auto* d = std::get_if<DenseWeights>(&w)) { return dense(*d); }
        const auto& moe = std::get<MoeWeights>(w);
        if (std::get<MoeConfig>(model_.config().text.ffn).num_experts_per_tok != 8) {
            throw std::invalid_argument("SparseMoe implements top-8 routing");
        }
        if (moe.delivery_layer) {
            const auto& source = model_.expert_sources().layers.at(*moe.delivery_layer);
            const std::array router{model_.input(moe.router), model_.input(moe.shared_score)};
            ops::SparseMoeWeights out;
            out.router_shared_gate = ops::prepare_linear_weight(router).weight;
            out.shared_gate_up = ops::prepare_linear_swiglu_weight(
                model_.input(moe.shared.gate), model_.input(moe.shared.up)).weight;
            out.shared_down = ops::prepare_linear_weight(model_.input(moe.shared.down)).weight;
            out.routed_gate_up.qtype = source.gate_up.format;
            out.routed_down.qtype = source.down.format;
            const auto gu = out.routed_gate_up.qtype, down = out.routed_down.qtype;
            const bool grouped = out.shared_gate_up.qtype == QType::Q8_G32_FP16 &&
                                  out.shared_down.qtype == QType::Q8_G32_FP16 &&
                ((gu == QType::Q4_G64_FP16 &&
                  (down == QType::Q5_G64_FP16 || down == QType::Q6_G64_FP16)) ||
                 (gu == QType::Q8_G32_FP16 && down == QType::Q8_G32_FP16));
            const bool nvfp4 = gu == QType::NVFP4 && down == QType::NVFP4 &&
                                out.shared_gate_up.qtype == QType::NVFP4 &&
                                out.shared_down.qtype == QType::NVFP4;
            if ((!grouped && !nvfp4) || out.router_shared_gate.qtype != QType::BF16 ||
                out.router_shared_gate.n != 257 || out.router_shared_gate.k != 2048 ||
                moe.experts.size() != 256 || out.shared_gate_up.n != 1024 ||
                out.shared_gate_up.k != 2048 || out.shared_down.n != 2048 ||
                out.shared_down.k != 512) {
                throw std::invalid_argument("streamed SparseMoe native geometry/profile differs");
            }
            if (nvfp4) {
#if defined(NINFER_SM8X_COMPAT) && !defined(NINFER_SM120_NVFP4)
                throw std::invalid_argument("SparseMoe NVFP4 expert banks need an sm_120a build");
#endif
                std::optional<float> activation_divisor, down_divisor;
                for (const auto& e : moe.experts) {
                    for (const auto id : {e.gate, e.up, e.down}) {
                        const auto& uses = model_.weight(id).uses;
                        if (uses.size() != 1 || uses[0].policy != ops::LinearPolicy::AllowA4 ||
                            !uses[0].activation_input_divisor) {
                            throw std::invalid_argument("streamed NVFP4 experts require AllowA4");
                        }
                        if (id == e.down) {
                            if (!down_divisor) { down_divisor = uses[0].activation_input_divisor; }
                            if (down_divisor != uses[0].activation_input_divisor) {
                                throw std::invalid_argument("streamed NVFP4 down activation divisors differ");
                            }
                            continue;
                        }
                        if (!activation_divisor) { activation_divisor = uses[0].activation_input_divisor; }
                        if (activation_divisor != uses[0].activation_input_divisor) {
                            throw std::invalid_argument("streamed NVFP4 gate/up activation divisors differ");
                        }
                    }
                }
                out.routed_gate_up.input_scale_divisor = *activation_divisor;
                out.routed_down.input_scale_divisor = *down_divisor;
                if (model_.input(moe.shared.gate).policy != ops::LinearPolicy::AllowA4 ||
                    model_.input(moe.shared.up).policy != ops::LinearPolicy::AllowA4 ||
                    model_.input(moe.shared.down).policy != ops::LinearPolicy::AllowA4 ||
                    *activation_divisor != out.shared_gate_up.input_scale_divisor) {
                    throw std::invalid_argument("streamed NVFP4 shared activation profile differs");
                }
            }
            return StreamedMoeParameters{out, *moe.delivery_layer};
        }
        std::vector<ops::WeightInput> gate_up, down;
        gate_up.reserve(2 * moe.experts.size());
        down.reserve(moe.experts.size());
        for (const auto& expert : moe.experts) {
            gate_up.push_back(model_.input(expert.gate));
            gate_up.push_back(model_.input(expert.up));
            down.push_back(model_.input(expert.down));
        }
        if (moe.share) {
            return with_context(model_.weight(moe.router).name, [&]() -> FfnParameters {
                std::vector<ops::WeightInput> shared;
                if (moe.share->shared_expert) {
                    for (const WeightId id : {moe.shared.gate, moe.shared.up, moe.shared.down}) {
                        shared.push_back(model_.input(id));
                    }
                }
                const auto held = [&](std::size_t i) {
                    return shared.empty() ? nullptr : &shared[i];
                };
                MoeShareParameters out;
                out.weights = ops::prepare_sparse_moe_share_weights(
                    model_.input(moe.router), model_.input(moe.shared_score), gate_up, down,
                    held(0), held(1), held(2));
                out.share = {.expert_begin  = static_cast<std::int32_t>(moe.share->expert_begin),
                             .expert_count  = static_cast<std::int32_t>(moe.experts.size()),
                             .shared_expert = moe.share->shared_expert,
                             .intermediate  = moe.share->intermediate != 0
                                                  ? static_cast<std::int32_t>(moe.share->intermediate)
                                                  : 512};
                return out;
            });
        }
        return with_context(model_.weight(moe.router).name, [&] {
            return ops::prepare_sparse_moe_weights(
                model_.input(moe.router), model_.input(moe.shared_score), gate_up, down,
                model_.input(moe.shared.gate), model_.input(moe.shared.up),
                model_.input(moe.shared.down));
        });
    }

    BlockParameters block(const BlockWeights& w) const {
        BlockParameters out;
        out.input_norm          = tensor(w.input_norm);
        out.post_attention_norm = tensor(w.post_attention_norm);
        out.ffn                 = ffn(w.ffn);
        if (const auto* a = std::get_if<AttentionWeights>(&w.mixer)) {
            LinearParameters attention_output = rotated_linear(a->output);
            integer_route(attention_output, QType::Q4_G64_FP16, 5120, 6144);
            integer_route(attention_output, QType::Q5_G64_FP16, 5120, 6144);
            integer_route(attention_output, QType::T2_G128_FP16, 5120, 6144);
            nvfp4_w4a16_route(attention_output, 5120, 6144);
            nvfp4_int8_route(attention_output, 5120, 6144);
            ops::ProjectionWeights attention_projection = ops::prepare_attn_input_proj_weights(
                model_.rotated_input(a->query), model_.rotated_input(a->key),
                model_.rotated_input(a->gate), model_.rotated_input(a->value));
            nvfp4_w4a16_route_single(attention_projection, 14336, 5120);
            nvfp4_int8_route_single(attention_projection, 14336, 5120);
            integer_route_pair(attention_projection, QType::Q4_G64_FP16, 7168, QType::Q5_G64_FP16,
                               7168, 5120);
            integer_route_pair(attention_projection, QType::T2_G128_FP16, 7168, QType::T2_G128_FP16,
                               7168, 5120);
            out.mixer = AttentionParameters{std::move(attention_projection), tensor(a->query_norm),
                                            tensor(a->key_norm), std::move(attention_output)};
            out.projection_prefetch =
                prefetch(std::get<AttentionParameters>(out.mixer).projection, a->query);
        } else {
            const auto& g              = std::get<GdnWeights>(w.mixer);
            LinearParameters gdn_output = rotated_linear(g.output);
            integer_route(gdn_output, QType::Q4_G64_FP16, 5120, 6144);
            integer_route(gdn_output, QType::Q5_G64_FP16, 5120, 6144);
            integer_route(gdn_output, QType::T2_G128_FP16, 5120, 6144);
            nvfp4_w4a16_route(gdn_output, 5120, 6144);
            nvfp4_int8_route(gdn_output, 5120, 6144);
            ops::ProjectionWeights gdn_projection = ops::prepare_gdn_input_proj_weights(
                model_.rotated_input(g.query), model_.rotated_input(g.key),
                model_.rotated_input(g.value), model_.rotated_input(g.z));
            nvfp4_w4a16_route_single(gdn_projection, 16384, 5120);
            nvfp4_int8_route_single(gdn_projection, 16384, 5120);
            integer_route_pair(gdn_projection, QType::Q4_G64_FP16, 4096, QType::Q5_G64_FP16, 12288,
                               5120);
            integer_route_pair(gdn_projection, QType::T2_G128_FP16, 4096, QType::T2_G128_FP16,
                               12288, 5120);
            out.mixer = GdnParameters{
                std::move(gdn_projection),
                ops::prepare_gdn_gating_proj_weights(model_.input(g.a_projection),
                                                         model_.input(g.b_projection)),
                tensor(g.a_log),
                tensor(g.dt_bias),
                tensor(g.convolution),
                tensor(g.norm),
                std::move(gdn_output)};
            out.projection_prefetch =
                prefetch(std::get<GdnParameters>(out.mixer).projection, g.query);
        }
        return out;
    }

    // A tensor-parallel rank's share of a layer. An attention layer's share has no mixer when the
    // rank holds no KV heads.
    BlockParameters rank_block(const RankLayerWeights& w) const {
        BlockParameters out;
        out.input_norm          = tensor(w.input_norm);
        out.post_attention_norm = tensor(w.post_attention_norm);
        out.ffn                 = ffn(w.ffn);
        if (w.attention) {
            const AttentionWeights& a = *w.attention;
            out.mixer                 = AttentionParameters{
                ops::prepare_attn_input_proj_weights(
                    model_.rotated_input(a.query), model_.rotated_input(a.key),
                    model_.rotated_input(a.gate), model_.rotated_input(a.value)),
                tensor(a.query_norm), tensor(a.key_norm), rotated_linear(a.output)};
        }
        if (w.gdn) {
            const GdnWeights& g = *w.gdn;
            out.mixer           = GdnParameters{
                ops::prepare_gdn_input_proj_weights(model_.rotated_input(g.query),
                                                    model_.rotated_input(g.key),
                                                    model_.rotated_input(g.value),
                                                    model_.rotated_input(g.z)),
                ops::prepare_gdn_gating_proj_weights(model_.input(g.a_projection),
                                                     model_.input(g.b_projection)),
                tensor(g.a_log),
                tensor(g.dt_bias),
                tensor(g.convolution),
                tensor(g.norm),
                rotated_linear(g.output)};
        }
        return out;
    }

    ops::SparseMoeHints prefetch(const ops::ProjectionWeights& projection, WeightId query) const {
        const Weight& weight = std::visit(
            [](const auto& p) -> const Weight& {
                using Kind = std::decay_t<decltype(p)>;
                if constexpr (std::is_same_v<Kind, ops::SingleProjectionWeight>) {
                    return p.weight;
                } else if constexpr (std::is_same_v<Kind, ops::PairedProjectionWeights>) {
                    return p.first;
                } else {
                    return p.parts.front().weight;
                }
            },
            projection);
        const auto& geometry = model_.weight(query).view.parts.front().parent->geometry;
        const auto row_bytes = geometry.layout == QuantLayout::Contiguous
                                   ? std::uint64_t(weight.k) * dtype_size(DType::BF16)
                                   : geometry.code_bytes_per_row;
        return {weight.qdata, static_cast<std::size_t>(row_bytes * weight.n)};
    }

    MtpParameters mtp(const MtpWeights& w) const {
        const auto& a = std::get<AttentionWeights>(w.layer.mixer);
        const std::array inputs{model_.input(a.query), model_.input(a.key), model_.input(a.gate),
                                model_.input(a.value)};
        MtpParameters out;
        out.input_projection    = linear(w.input_projection);
        out.embedding_norm      = tensor(w.embedding_norm);
        out.hidden_norm         = tensor(w.hidden_norm);
        out.input_norm          = tensor(w.layer.input_norm);
        out.post_attention_norm = tensor(w.layer.post_attention_norm);
        out.final_norm          = tensor(w.final_norm);
        out.ffn                 = ffn(w.layer.ffn);
        for (const DenseWeights& share : w.rank_ffn) { out.rank_ffn.push_back(ffn(share)); }
        out.output_head        = head(w.output_head_use);
        out.output_head_shares = head_shares(w.output_head_shares);
        if (!w.rank_attention.empty()) {
            // Divided: each rank projects its heads through the text layers' attention form.
            const auto share = [&](const AttentionWeights& s) {
                return AttentionParameters{
                    ops::prepare_attn_input_proj_weights(model_.input(s.query),
                                                         model_.input(s.key),
                                                         model_.input(s.gate),
                                                         model_.input(s.value)),
                    tensor(s.query_norm), tensor(s.key_norm), linear(s.output)};
            };
            out.rank_attention.push_back(share(a));
            for (const AttentionWeights& s : w.rank_attention) {
                out.rank_attention.push_back(share(s));
            }
            out.query_norm  = tensor(a.query_norm);
            out.key_norm    = tensor(a.key_norm);
            out.whole_key   = linear(w.whole_key);
            out.whole_value = linear(w.whole_value);
            out.rank_input_norm.push_back(out.input_norm);
            out.rank_post_attention_norm.push_back(out.post_attention_norm);
            for (std::size_t rank = 0; rank < w.rank_input_norm.size(); ++rank) {
                out.rank_input_norm.push_back(tensor(w.rank_input_norm[rank]));
                out.rank_post_attention_norm.push_back(tensor(w.rank_post_attention_norm[rank]));
            }
            return out;
        }
        // A bank whose parts share no contiguous parent -- GGUF parts of different block types, or
        // mixed formats such as Q8 K/V beside BF16 Q/gate -- has no packed form, and the dense
        // rows below serve alone.
        const bool packable = ops::joins(inputs);
        if (!packable && model_.config().text.architecture != Architecture::Qwen3_5) {
            throw std::invalid_argument("MTP attention weights share no contiguous parent");
        }
        if (packable) {
            out.projection.packed = ops::prepare_linear_weight(inputs);
            nvfp4_w4a16_route(out.projection.packed, 14336, 5120);
            nvfp4_int8_route(out.projection.packed, 14336, 5120);
        }
        if (model_.config().text.architecture == Architecture::Qwen3_5) {
            out.projection.rows = {linear(a.query), linear(a.key), linear(a.gate), linear(a.value)};
        }
        out.query_norm  = tensor(a.query_norm);
        out.key_norm    = tensor(a.key_norm);
        out.output      = linear(a.output);
        nvfp4_w4a16_route(out.output, 5120, 6144);
        nvfp4_int8_route(out.output, 5120, 6144);
        return out;
    }

    NormParameters norm(const NormWeights& w) const { return {tensor(w.weight), tensor(w.bias)}; }

    std::optional<Tensor> joined_bias(const std::array<WeightId, 3>& ids) const {
        WeightView view;
        std::uint64_t count = 0;
        for (const auto id : ids) {
            const auto& input = model_.weight(id).view;
            count += weight_element_count(input.shape);
            for (const auto& part : input.parts) {
                if (!view.parts.empty() && (view.parts.back().parent != part.parent ||
                                            view.parts.back().end != part.begin)) {
                    return std::nullopt;
                }
                view.parts.push_back(part);
            }
        }
        if (count > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument("Vision bias exceeds Tensor extent");
        }
        view.shape = {count};
        return weight_tensor(view, {static_cast<std::int32_t>(count)});
    }

    VisionParameters vision(const VisionWeights& w) const {
        VisionParameters out;
        out.patch_embedding      = linear(w.patch_embedding);
        out.patch_embedding_bias = tensor(w.patch_embedding_bias);
        out.position_embedding   = tensor(w.position_embedding);
        out.layers.reserve(w.layers.size());
        for (std::size_t i = 0; i < w.layers.size(); ++i) {
            out.layers.push_back(with_context("vision/layers/" + std::to_string(i), [&] {
                const auto& layer = w.layers[i];
                const std::array qkv{model_.input(layer.query), model_.input(layer.key),
                                     model_.input(layer.value)};
                const std::array ids{layer.query_bias, layer.key_bias, layer.value_bias};
                const auto bias = joined_bias(ids);
                if (!bias) {
                    throw std::invalid_argument(
                        "Vision QKV bias: this fixed call requires a contiguous bias bank");
                }
                return VisionBlockParameters{norm(layer.norm1),
                                             norm(layer.norm2),
                                             ops::prepare_linear_weight(qkv),
                                             *bias,
                                             linear(layer.output),
                                             linear(layer.fc1),
                                             linear(layer.fc2),
                                             tensor(layer.output_bias),
                                             tensor(layer.fc1_bias),
                                             tensor(layer.fc2_bias)};
            }));
        }
        out.merger_norm     = norm(w.merger_norm);
        out.merger_fc1      = linear(w.merger_fc1);
        out.merger_fc2      = linear(w.merger_fc2);
        out.merger_fc1_bias = tensor(w.merger_fc1_bias);
        out.merger_fc2_bias = tensor(w.merger_fc2_bias);
        return out;
    }

    DynamicConvParameters convolution(const DynamicConvWeights& w) const {
        return {tensor(w.base_kernel), linear(w.kernel_projection)};
    }

    DraftParameters draft(const DraftWeights& w) const {
        DraftParameters out;
        out.feature_projection = linear(w.feature_projection);
        out.context_norm       = tensor(w.context_norm);
        out.final_norm         = tensor(w.final_norm);
        out.output_head        = head(w.output_head_use);
        out.layers.reserve(w.layers.size());
        for (std::size_t i = 0; i < w.layers.size(); ++i) {
            out.layers.push_back(with_context(
                std::string(model_.options().speculative_component()) + "/layers/" +
                    std::to_string(i),
                [&] {
                    const auto& layer = w.layers[i];
                    const auto& a     = layer.attention;
                    DraftBlockParameters result;
                    result.input_norm          = tensor(layer.input_norm);
                    result.post_attention_norm = tensor(layer.post_attention_norm);
                    result.query_key_value     = ops::prepare_attn_input_proj_weights(
                        model_.input(a.query), model_.input(a.key), model_.input(a.value));
                    result.context_key   = linear(a.context_key);
                    result.context_value = linear(a.context_value);
                    result.query_norm    = tensor(a.query_norm);
                    result.key_norm      = tensor(a.key_norm);
                    result.output        = linear(a.output);
                    result.mlp           = dense(layer.mlp);
                    if (layer.attention_conv) {
                        result.attention_conv = convolution(*layer.attention_conv);
                    }
                    if (layer.mlp_conv) { result.mlp_conv = convolution(*layer.mlp_conv); }
                    return result;
                }));
        }
        if (w.selector) {
            out.selector = SelectorParameters{linear(w.selector->hidden_projection),
                                              tensor(w.selector->predecessor_codebook),
                                              tensor(w.selector->successor_codebook)};
        }
        return out;
    }

private:
    const Model& model_;
};

} // namespace

Parameters::Parameters(const Model& source) : model(source) {
    const Prepare prepare(model);
    const auto& w        = model.weights();
    text.token_embedding = native_weight(model.weight(w.text.token_embedding).view);
    text.output_head        = prepare.head(w.text.output_head_use);
    text.output_head_shares = prepare.head_shares(w.text.output_head_shares);
    if (text.token_embedding.qtype == QType::T2_G128_FP16) {
        // A ternary checkpoint stores its token table rotated by the same hidden-width signs as
        // every 5120-wide input, the output head's among them; the table has no Use of its own.
        if (!rotated(text.output_head.hadamard_signs) ||
            text.output_head.hadamard_signs.ne[0] != text.token_embedding.k) {
            throw std::invalid_argument(
                "a T2 token embedding needs a rotated output head carrying the hidden-width signs");
        }
        text.token_embedding_signs = text.output_head.hadamard_signs;
    }
    text.final_norm      = prepare.tensor(w.text.final_norm);
    text.rank_count = w.text.stages.stages();
    text.stage_begin.clear();
    for (std::size_t stage = 0; stage < w.text.stages.stages(); ++stage) {
        text.stage_begin.push_back(w.text.stages.stage_begin(stage));
    }
    text.layers.reserve(w.text.layers.size());
    for (std::size_t i = 0; i < w.text.layers.size(); ++i) {
        text.layers.push_back(with_context("text/layers/" + std::to_string(i),
                                           [&] { return prepare.block(w.text.layers[i]); }));
        text.layers.back().rank = w.text.stages.placement(static_cast<std::uint32_t>(i)).stage;
    }
    if (!w.text.rank_layers.empty()) {
        text.rank_count    = w.text.rank_layers.size() + 1;
        text.tensor_shares = w.text.tensor_shares;
        text.rank_layers.resize(w.text.rank_layers.size());
        for (std::size_t r = 0; r < w.text.rank_layers.size(); ++r) {
            text.rank_layers[r].reserve(w.text.rank_layers[r].size());
            for (std::size_t i = 0; i < w.text.rank_layers[r].size(); ++i) {
                text.rank_layers[r].push_back(
                    with_context("text/layers/" + std::to_string(i) + "@" + std::to_string(r + 1),
                                 [&] { return prepare.rank_block(w.text.rank_layers[r][i]); }));
                text.rank_layers[r].back().rank = r + 1;
            }
        }
    }
    if (w.mtp) {
        mtp = with_context("mtp", [&] { return prepare.mtp(*w.mtp); });
    }
    if (w.vision) {
        vision = with_context("vision", [&] { return prepare.vision(*w.vision); });
    }
    if (w.draft) {
        draft = with_context(std::string(model.options().speculative_component()),
                             [&] { return prepare.draft(*w.draft); });
    }
    if (w.proposal) {
        proposal = ProposalParameters{prepare.head(w.proposal->head), std::nullopt,
                                      w.proposal->rows, prepare.head_shares(w.proposal->head_shares)};
        if (w.proposal->token_ids) { proposal->token_ids = prepare.tensor(*w.proposal->token_ids); }
    }
    dspine = model.dspine();
}

} // namespace ninfer::models::qwen3_5::execution

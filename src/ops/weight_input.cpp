#include "ninfer/ops/weight_input.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

inline bool debug_weights() {
    static const bool enabled = [] {
        const char* v = std::getenv("NINFER_DEBUG_WEIGHTS");
        return v != nullptr && *v != '\0';
    }();
    return enabled;
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

const std::vector<std::uint64_t>& matrix(const WeightInput& input) {
    require(input.weight.shape.size() == 2, "projection weight must be a matrix");
    const auto elements = weight_element_count(input.weight.shape);
    require(!input.weight.parts.empty(), "projection weight has no parent regions");
    std::uint64_t covered = 0;
    for (const auto& part : input.weight.parts) {
        require(part.parent && part.begin < part.end &&
                    part.end <= part.parent->geometry.elements &&
                    part.end - part.begin <= elements - covered,
                "projection region exceeds its parent or logical matrix");
        covered += part.end - part.begin;
    }
    require(covered == elements, "projection regions do not cover its logical matrix");
    return input.weight.shape;
}

WeightView concatenate_rows(std::span<const WeightInput> inputs) {
    require(!inputs.empty(), "projection bank is empty");
    const auto columns = matrix(inputs.front())[1];
    WeightView out{{0, columns}, {}};
    for (const auto& input : inputs) {
        const auto& shape = matrix(input);
        require(shape[1] == columns &&
                    shape[0] <= std::numeric_limits<std::uint64_t>::max() - out.shape[0],
                "projection bank has inconsistent columns or overflowing rows");
        out.shape[0] += shape[0];
        out.parts.insert(out.parts.end(), input.weight.parts.begin(), input.weight.parts.end());
    }
    return out;
}

bool contiguous(const WeightView& view) {
    if (view.parts.empty()) { return false; }
    const auto* parent = view.parts.front().parent;
    auto end           = view.parts.front().begin;
    for (const auto& part : view.parts) {
        if (!parent || part.parent != parent || part.begin != end || part.end <= part.begin) {
            return false;
        }
        end = part.end;
    }
    return true;
}

LinearPolicy common_policy(std::span<const WeightInput> inputs) {
    auto result = LinearPolicy::AllowA4;
    for (const auto& input : inputs) {
        switch (input.policy) {
        case LinearPolicy::A16Only:
            result = LinearPolicy::A16Only;
            break;
        case LinearPolicy::AllowA8:
            if (result == LinearPolicy::AllowA4) { result = LinearPolicy::AllowA8; }
            break;
        case LinearPolicy::AllowA4:
            break;
        default:
            throw std::invalid_argument("invalid projection activation policy");
        }
    }
    return result;
}

// Rows fused into one native operand read one activation, so they must agree on its rotation.
Tensor common_signs(std::span<const WeightInput> inputs) {
    const Tensor signs = inputs.front().hadamard_signs;
    for (const auto& input : inputs) {
        require(input.hadamard_signs.data == signs.data &&
                    input.hadamard_signs.ne[0] == signs.ne[0],
                "fused projection rows must share one Hadamard rotation");
    }
    return signs;
}

SingleProjectionWeight single(std::span<const WeightInput> inputs) {
    auto view         = concatenate_rows(inputs);
    const auto region = contiguous_weight_region(view);
    const auto policy = common_policy(inputs);
    float divisor     = 0;
    if (region.parent->geometry.format == QType::NVFP4) {
        require(std::isfinite(region.parent->weight_scale_divisor) &&
                    region.parent->weight_scale_divisor > 0,
                "NVFP4 parent requires a positive weight divisor");
        for (const auto& input : inputs) {
            if (!input.activation_input_divisor) {
                require(!allows_a4(policy), "NVFP4 A4 native input requires an activation divisor");
                continue;
            }
            require(std::isfinite(*input.activation_input_divisor) &&
                        *input.activation_input_divisor > 0,
                    "NVFP4 native input requires a positive activation divisor");
            if (divisor == 0) {
                // A16 does not read this auxiliary; retain a stored positive value for the ABI.
                divisor = *input.activation_input_divisor;
            } else if (allows_a4(policy)) {
                // Current NVFP4 consumers use A16 for AllowA8 as well. Only their A4 route
                // quantizes the shared activation and therefore requires a common divisor.
                require(std::bit_cast<std::uint32_t>(divisor) ==
                            std::bit_cast<std::uint32_t>(*input.activation_input_divisor),
                        "shared NVFP4 native input requires identical activation divisors");
            }
        }
        // The legacy native ABI validates this field even on A16, where it is never read.
        // Keep that ABI detail out of the artifact's optional Use auxiliaries.
        if (divisor == 0) { divisor = 1.0F; }
    }
    SingleProjectionWeight result{native_weight(view, divisor), policy, common_signs(inputs)};
    const Tensor& columns = inputs.front().input_columns;
    for (const auto& input : inputs) {
        require(input.input_columns.data == columns.data,
                "fused projection rows must share one input gather");
    }
    if (columns.data != nullptr) {
        // Only the GGUF linears read one; any other route would multiply the stored columns as
        // if they were in the input's order.
        require(is_gguf(result.weight.qtype), "an input gather requires a GGUF matrix");
        require(columns.dtype == DType::I32 && columns.numel() == result.weight.k,
                "an input gather must be INT32 [K]");
        result.weight.input_columns = static_cast<const std::int32_t*>(columns.data);
    }
    return result;
}

bool gguf_parent(const WeightInput& input) {
    return !input.weight.parts.empty() && input.weight.parts.front().parent != nullptr &&
           is_gguf(input.weight.parts.front().parent->geometry.format);
}

// The parts of an input projection, one per logical input: GGUF parts, merged where consecutive
// logical inputs continue one parent's rows into one output's rows, or a tensor-parallel rank's
// pieces of another format, each its own device object.
PartProjectionWeights part_projection(std::span<const WeightInput> inputs,
                                      std::span<const std::int32_t> outputs,
                                      std::span<const std::int32_t> rows) {
    PartProjectionWeights result;
    const bool gguf = gguf_parent(inputs.front());
    // q8_1 GGUF arithmetic needed no permission. The optional floating-point prefill
    // routes do: a composed projection grants only what every logical Use permits.
    if (gguf) { result.policy = common_policy(inputs); }
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        require(gguf_parent(inputs[i]) == gguf, "input projection: parts cannot mix formats");
        auto part = single(inputs.subspan(i, 1));
        if (!gguf) { result.policy = part.policy; }
        if (gguf && !result.parts.empty()) {
            auto& last = result.parts.back();
            const auto block = gguf_block_shape(last.weight.qtype);
            const std::int64_t row_bytes = std::int64_t(last.weight.k / block.elements) * block.bytes;
            const auto* end = static_cast<const std::byte*>(last.weight.qdata) + last.weight.n * row_bytes;
            if (part.weight.qtype == last.weight.qtype && part.weight.qdata == end &&
                outputs[i] == last.output && rows[i] == last.row + last.weight.n) {
                last.weight.n += part.weight.n;
                last.weight.shape[0] = last.weight.padded_shape[0] = last.weight.n;
                continue;
            }
        }
        result.parts.push_back({part.weight, outputs[i], rows[i]});
    }
    result.hadamard_signs = common_signs(inputs);
    return result;
}

ProjectionWeights input_projection(std::span<const WeightInput, 4> inputs, bool attention) {
    const auto& q      = matrix(inputs[0]);
    const auto& k      = matrix(inputs[1]);
    const auto& third  = matrix(inputs[2]);
    const auto& fourth = matrix(inputs[3]);
    const bool dense =
        attention ? q == std::vector<std::uint64_t>{6144, 5120} &&
                        k == std::vector<std::uint64_t>{1024, 5120} && third == q && fourth == k
                  : q == std::vector<std::uint64_t>{2048, 5120} && k == q &&
                        third == std::vector<std::uint64_t>{6144, 5120} && fourth == third;
    const bool moe =
        attention ? q == std::vector<std::uint64_t>{4096, 2048} &&
                        k == std::vector<std::uint64_t>{512, 2048} && third == q && fourth == k
                  : q == std::vector<std::uint64_t>{2048, 2048} && k == q &&
                        third == std::vector<std::uint64_t>{4096, 2048} && fourth == third;
    // A tensor-parallel rank's share of a GDN projection: whole key heads of 128 rows and the
    // value heads that read them, a whole multiple of the key rows. A whole MoE layer keeps its
    // fused route.
    const bool dense_share = !attention && !dense && !moe && q.size() == 2 && k == q &&
                             fourth == third &&
                             third.size() == 2 && third[1] == q[1] && q[0] > 0 &&
                             q[0] % 128 == 0 && third[0] % q[0] == 0;
    // A rank's share of an attention projection: whole KV heads of 256 rows and their query heads,
    // a whole multiple of the KV rows.
    const bool attention_share = attention && !dense && !moe && q.size() == 2 && k.size() == 2 &&
                                 third == q && fourth == k && k[1] == q[1] && k[0] > 0 &&
                                 k[0] % 256 == 0 && q[0] % k[0] == 0;
    // A position-split rank's pieces of every head: four device objects, of which neither pair
    // joins as a stored attention projection's parents do.
    const bool whole_pieces = attention && (dense || moe) && !joins(inputs.first<2>()) &&
                              !joins(inputs.last<2>());
    const bool share = dense_share || attention_share || whole_pieces;
    
    if (std::any_of(inputs.begin(), inputs.end(), gguf_parent) || share) {
        require(q[1] == k[1] && q[1] == third[1] && q[1] == fourth[1] &&
                    (attention ? third == q && fourth == k && q[0] % k[0] == 0
                               : k == q && fourth == third && third[0] % q[0] == 0),
                "input projection: inconsistent GGUF logical geometry");
        const auto q = static_cast<std::int32_t>(matrix(inputs[0])[0]);
        const auto k = static_cast<std::int32_t>(matrix(inputs[1])[0]);
        if (attention) {
            const std::array<std::int32_t, 4> outputs{0, 2, 1, 3};
            const std::array<std::int32_t, 4> rows{0, 0, 0, 0};
            return part_projection(inputs, outputs, rows);
        }
        const std::array<std::int32_t, 4> outputs{0, 0, 0, 1};
        const std::array<std::int32_t, 4> rows{0, q, q + k, 0};
        return part_projection(inputs, outputs, rows);
    }
    require(dense || moe, "input projection: unsupported native logical projection geometry");
    const auto joined = concatenate_rows(inputs);
    if (debug_weights()) {
        fprintf(stderr, "[debug] input_projection attention=%d contiguous=%d\n",
                (int)attention, (int)contiguous(joined));
    }
    if (contiguous(joined)) {
        auto result       = single(inputs);
        const auto format = result.weight.qtype;
        const bool supported =
            (moe && format == QType::Q8_G32_FP16) ||
            (dense && (format == QType::NVFP4 || format == QType::FP8_E4M3FN_ROW_BF16 ||
                       (attention && format == QType::BF16)));
        require(supported, "input projection: unsupported single-parent format");
        return result;
    }
    require(dense, "input projection: unsupported multi-parent geometry");
    auto first  = single(inputs.first<2>());
    auto second = single(inputs.last<2>());
    // Populate per-sub-weight qtype info for mixed weights. For attention: first=query+key,
    // second=gate+value. For GDN: first=q+k, second=v+z.
    if (attention) {
        const auto q_fmt = inputs[0].weight.parts[0].parent->geometry.format;
        const auto k_fmt = inputs[1].weight.parts[0].parent->geometry.format;
        if (debug_weights()) {
            fprintf(stderr, "[debug] input_projection qk: q=%d k=%d g=%d v=%d first_qtype=%d second_qtype=%d\n",
                    (int)q_fmt, (int)k_fmt,
                    (int)inputs[2].weight.parts[0].parent->geometry.format,
                    (int)inputs[3].weight.parts[0].parent->geometry.format,
                    (int)first.weight.qtype, (int)second.weight.qtype);
        }
        if (q_fmt != k_fmt) {
            first.weight.second_sub_qtype = k_fmt;
            first.weight.second_sub_row   = static_cast<std::int32_t>(inputs[0].weight.shape[0]);
            if (debug_weights()) {
                fprintf(stderr, "[debug] input_projection MIXED qk: q=%d k=%d row=%d\n",
                        (int)q_fmt, (int)k_fmt, first.weight.second_sub_row);
            }
        }
        const auto g_fmt = inputs[2].weight.parts[0].parent->geometry.format;
        const auto v_fmt = inputs[3].weight.parts[0].parent->geometry.format;
        if (g_fmt != v_fmt) {
            second.weight.second_sub_qtype = v_fmt;
            second.weight.second_sub_row   = static_cast<std::int32_t>(inputs[2].weight.shape[0]);
            if (debug_weights()) {
                fprintf(stderr, "[debug] input_projection MIXED gv: g=%d v=%d row=%d\n",
                        (int)g_fmt, (int)v_fmt, second.weight.second_sub_row);
            }
        }
    } else {
        const auto q_fmt = inputs[0].weight.parts[0].parent->geometry.format;
        const auto k_fmt = inputs[1].weight.parts[0].parent->geometry.format;
        if (q_fmt != k_fmt) {
            first.weight.second_sub_qtype = k_fmt;
            first.weight.second_sub_row   = static_cast<std::int32_t>(inputs[0].weight.shape[0]);
        }
        const auto v_fmt = inputs[2].weight.parts[0].parent->geometry.format;
        const auto z_fmt = inputs[3].weight.parts[0].parent->geometry.format;
        if (v_fmt != z_fmt) {
            second.weight.second_sub_qtype = z_fmt;
            second.weight.second_sub_row   = static_cast<std::int32_t>(inputs[2].weight.shape[0]);
        }
    }
    const bool q4_q4 =
        first.weight.qtype == QType::Q4_G64_FP16 && second.weight.qtype == QType::Q4_G64_FP16;
    const bool q4_q5 =
        first.weight.qtype == QType::Q4_G64_FP16 && second.weight.qtype == QType::Q5_G64_FP16;
    const bool q5_q5 =
        first.weight.qtype == QType::Q5_G64_FP16 && second.weight.qtype == QType::Q5_G64_FP16;
    const bool q4_q6 =
        first.weight.qtype == QType::Q4_G64_FP16 && second.weight.qtype == QType::Q6_G64_FP16;
    const bool q6_q5 =
        first.weight.qtype == QType::Q6_G64_FP16 && second.weight.qtype == QType::Q5_G64_FP16;
    const bool q6_q6 =
        first.weight.qtype == QType::Q6_G64_FP16 && second.weight.qtype == QType::Q6_G64_FP16;
    const bool ternary = is_ternary(first.weight.qtype) &&
                         first.weight.qtype == second.weight.qtype;
    require(q4_q4 || q4_q5 || q5_q5 || q4_q6 || q6_q5 || q6_q6 || ternary,
            "input projection: requires Q4/Q4, Q4/Q5, Q5/Q5, Q4/Q6, Q6/Q5, Q6/Q6, or two ternary parents of the same format");
    const auto pair_policy = first.weight.qtype == QType::T5_G128_FP16
                                 ? common_policy(inputs) : LinearPolicy::A16Only;
    return PairedProjectionWeights{first.weight, second.weight, pair_policy,
                                   common_signs(inputs)};
}

} // namespace

bool joins(std::span<const WeightInput> inputs) { return contiguous(concatenate_rows(inputs)); }

SingleProjectionWeight prepare_linear_weight(const WeightInput& input) {
    return single({&input, 1});
}

SingleProjectionWeight prepare_linear_weight(std::span<const WeightInput> rows) {
    return single(rows);
}

SingleProjectionWeight prepare_attn_input_proj_weights(const WeightInput& query,
                                                       const WeightInput& key,
                                                       const WeightInput& value) {
    const auto& q = matrix(query);
    require((q == std::vector<std::uint64_t>{4096, 2048} ||
             q == std::vector<std::uint64_t>{4096, 5120}) &&
                matrix(key) == std::vector<std::uint64_t>{1024, q[1]} &&
                matrix(value) == matrix(key),
            "QKV input projection: unsupported logical geometry");
    const std::array inputs{query, key, value};
    auto result = single(inputs);
    require(result.weight.qtype == QType::Q8_G32_FP16,
            "QKV input projection: native form requires Q8");
    return result;
}

ProjectionWeights prepare_attn_input_proj_weights(const WeightInput& query, const WeightInput& key,
                                                  const WeightInput& gate,
                                                  const WeightInput& value) {
    const std::array inputs{query, key, gate, value};
    return input_projection(inputs, true);
}

ProjectionWeights prepare_gdn_input_proj_weights(const WeightInput& query, const WeightInput& key,
                                                 const WeightInput& value, const WeightInput& z) {
    const std::array inputs{query, key, value, z};
    return input_projection(inputs, false);
}

ProjectionWeights prepare_gdn_gating_proj_weights(const WeightInput& a, const WeightInput& b) {
    const auto& shape = matrix(a);
    require(shape == matrix(b) && shape[0] > 0 && shape[0] % 16 == 0 &&
                shape[1] > 0 && shape[1] % 128 == 0,
            "GDN control: unsupported A/B geometry");
    const std::array inputs{a, b};
    if (contiguous(concatenate_rows(inputs))) {
        auto result = single(inputs);
        require(result.weight.qtype == QType::BF16, "GDN control requires BF16 weights");
        return result;
    }
    const auto first  = prepare_linear_weight(a);
    const auto second = prepare_linear_weight(b);
    require(first.weight.qtype == QType::BF16 && second.weight.qtype == QType::BF16,
            "GDN control requires BF16 weights");
    return PairedProjectionWeights{first.weight, second.weight};
}

SingleProjectionWeight prepare_linear_swiglu_weight(const WeightInput& gate,
                                                    const WeightInput& up) {
    require(matrix(gate) == matrix(up), "SwiGLU gate and up geometry differs");
    const std::array inputs{gate, up};
    return single(inputs);
}

namespace {

// `experts` of the 256 in each bank; the shared expert's inputs, or none for a share without it.
SparseMoeWeights prepare_sparse_moe_banks(const WeightInput& router,
                                          const WeightInput& shared_score,
                                          std::span<const WeightInput> expert_gate_up,
                                          std::span<const WeightInput> expert_down,
                                          std::size_t experts, std::uint64_t width,
                                          const WeightInput* shared_gate,
                                          const WeightInput* shared_up,
                                          const WeightInput* shared_down) {
    require(experts > 0 && expert_gate_up.size() == 2 * experts && expert_down.size() == experts,
            experts == 256 && width == 512 ? "SparseMoe native input requires 256 experts"
                                           : "SparseMoe share inputs differ from its experts");
    require(matrix(router) == std::vector<std::uint64_t>{256, 2048} &&
                matrix(shared_score) == std::vector<std::uint64_t>{1, 2048},
            "SparseMoe router geometry differs");
    for (const auto& input : expert_gate_up) {
        require(matrix(input) == std::vector<std::uint64_t>{width, 2048},
                "SparseMoe gate/up geometry differs");
    }
    for (const auto& input : expert_down) {
        require(matrix(input) == std::vector<std::uint64_t>{2048, width},
                "SparseMoe down geometry differs");
    }
    const bool holds_shared = shared_gate != nullptr;
    require(holds_shared == (shared_up != nullptr) && holds_shared == (shared_down != nullptr) &&
                (holds_shared || experts < 256 || width < 512),
            "SparseMoe shared expert inputs are incomplete");
    if (holds_shared) {
        require(matrix(*shared_gate) == std::vector<std::uint64_t>{width, 2048} &&
                    matrix(*shared_up) == matrix(*shared_gate) &&
                    matrix(*shared_down) == std::vector<std::uint64_t>{2048, width},
                "SparseMoe shared expert geometry differs");
    }
    const std::array router_inputs{router, shared_score};
    const auto router_bank  = single(router_inputs);
    const auto gate_up_bank = single(expert_gate_up);
    const auto down_bank    = single(expert_down);
    const auto shared       = holds_shared ? prepare_linear_swiglu_weight(*shared_gate, *shared_up)
                                           : SingleProjectionWeight{};
    const auto down = holds_shared ? prepare_linear_weight(*shared_down) : SingleProjectionWeight{};
    const auto gate_format = gate_up_bank.weight.qtype;
    const auto down_format = down_bank.weight.qtype;
    // A share without the shared expert takes the profile's Q8 for it.
    const auto shared_format      = holds_shared ? shared.weight.qtype : QType::Q8_G32_FP16;
    const auto shared_down_format = holds_shared ? down.weight.qtype : QType::Q8_G32_FP16;
    const bool groupwise =
        shared_format == QType::Q8_G32_FP16 && shared_down_format == QType::Q8_G32_FP16 &&
        ((gate_format == QType::Q4_G64_FP16 &&
          (down_format == QType::Q5_G64_FP16 || down_format == QType::Q6_G64_FP16)) ||
         (gate_format == QType::Q8_G32_FP16 && down_format == QType::Q8_G32_FP16));
    // The four expert matrices are one profile rather than two independent choices: the prefill
    // route quantises the activation as well as the weight and its gate/up epilogue emits the
    // encoded intermediate `down` consumes, so a mixture with a groupwise codec has no route
    // behind it at any token count.
    const bool nvfp4 = gate_format == QType::NVFP4 && down_format == QType::NVFP4 &&
                       shared_format == QType::NVFP4 && shared_down_format == QType::NVFP4;
    require(router_bank.weight.qtype == QType::BF16 && (groupwise || nvfp4),
            "SparseMoe native bank formats are unsupported");
#if defined(NINFER_SM8X_COMPAT) && !defined(NINFER_SM120_NVFP4)
    // A model prefills, and this profile prefills only through W4A4 on Blackwell tensor cores.
    require(!nvfp4, "SparseMoe NVFP4 expert banks need an sm_120a build");
#endif
    if (nvfp4) {
        // The prefill route of this profile quantises the hidden state to four bits, which is what
        // AllowA4 grants. The groupwise profiles consume it represented and need no permission.
        const auto permits = [](const WeightInput& input) { return allows_a4(input.policy); };
        require(std::all_of(expert_gate_up.begin(), expert_gate_up.end(), permits) &&
                    std::all_of(expert_down.begin(), expert_down.end(), permits) &&
                    permits(*shared_gate) && permits(*shared_up) && permits(*shared_down),
                "SparseMoe NVFP4 requires AllowA4 on every expert input");
    }
    return {router_bank.weight, gate_up_bank.weight, down_bank.weight, shared.weight, down.weight};
}

} // namespace

SparseMoeWeights
prepare_sparse_moe_weights(const WeightInput& router, const WeightInput& shared_score,
                           std::span<const WeightInput> expert_gate_up,
                           std::span<const WeightInput> expert_down, const WeightInput& shared_gate,
                           const WeightInput& shared_up, const WeightInput& shared_down) {
    return prepare_sparse_moe_banks(router, shared_score, expert_gate_up, expert_down, 256, 512,
                                    &shared_gate, &shared_up, &shared_down);
}

SparseMoeWeights prepare_sparse_moe_share_weights(
    const WeightInput& router, const WeightInput& shared_score,
    std::span<const WeightInput> expert_gate_up, std::span<const WeightInput> expert_down,
    const WeightInput* shared_gate, const WeightInput* shared_up, const WeightInput* shared_down) {
    // A share holds some of the experts whole, or every expert's half of its intermediate
    // columns.
    const std::uint64_t width = expert_down.empty() ? 0 : matrix(expert_down.front()).back();
    require(!expert_down.empty() && expert_down.size() <= 256 &&
                (width == 256 || (width == 512 && expert_down.size() < 256)),
            "SparseMoe share holds some of the 256 experts, or half of each one's columns");
    const auto out =
        prepare_sparse_moe_banks(router, shared_score, expert_gate_up, expert_down,
                                 expert_down.size(), width, shared_gate, shared_up, shared_down);
    require(out.routed_gate_up.qtype != QType::NVFP4,
            "SparseMoe NVFP4 expert banks have no expert-parallel share");
    return out;
}

} // namespace ninfer::ops

// LoRA adapter and control vector loading (frontend/steering.h) against a small hybrid model
// geometry. The GGUF files are written here byte for byte in llama.cpp's layouts.
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/frontend/steering.h"
#include "runtime/engine/model_instance.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace q = ninfer::models::qwen3_5;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

constexpr std::uint32_t kHidden = 16;

// Three Gated DeltaNet layers then one attention layer, the Qwen3.5 hybrid period, with a dense
// FFN: attention output reads 32 columns, the GDN output 16 and the FFN down projection 24.
q::TextConfig model() {
    q::TextConfig text;
    text.hidden_size       = kHidden;
    text.vocab_size        = 1000;
    text.num_hidden_layers = 4;
    text.layer_types       = {q::MixerKind::LinearAttention, q::MixerKind::LinearAttention,
                              q::MixerKind::LinearAttention, q::MixerKind::FullAttention};
    text.full_attention_layers   = 1;
    text.linear_attention_layers = 3;
    text.attention = q::AttentionConfig{.num_attention_heads = 4, .num_key_value_heads = 2,
                                        .head_dim = 8};
    text.gdn = q::GdnConfig{.linear_num_key_heads   = 2,
                            .linear_key_head_dim    = 4,
                            .linear_num_value_heads = 4,
                            .linear_value_head_dim  = 4,
                            .linear_conv_kernel_dim = 4};
    text.ffn = q::DenseConfig{.intermediate_size = 24};
    return text;
}

constexpr std::uint32_t kF32  = 0;
constexpr std::uint32_t kF16  = 1;
constexpr std::uint32_t kBF16 = 30;

struct GgufTensor {
    std::string name;
    std::vector<std::uint64_t> ne;
    std::uint32_t type = kF32;
    std::vector<std::uint8_t> bytes;
};

using GgufValue = std::variant<std::string, float, std::uint32_t>;

struct Gguf {
    std::vector<std::pair<std::string, GgufValue>> values;
    std::vector<GgufTensor> tensors;
};

GgufTensor f32(std::string name, std::vector<std::uint64_t> ne, const std::vector<float>& values) {
    GgufTensor tensor{std::move(name), std::move(ne), kF32, {}};
    tensor.bytes.resize(values.size() * 4);
    std::memcpy(tensor.bytes.data(), values.data(), tensor.bytes.size());
    return tensor;
}

// Values exact in both 16-bit formats.
GgufTensor half(std::string name, std::vector<std::uint64_t> ne, const std::vector<float>& values,
                std::uint32_t type) {
    GgufTensor tensor{std::move(name), std::move(ne), type, {}};
    for (const float value : values) {
        std::uint16_t bits = 0;
        if (type == kBF16) {
            bits = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16U);
        } else {
            const std::uint32_t f = std::bit_cast<std::uint32_t>(value);
            if ((f & 0x7fffffffU) != 0) {
                const std::uint32_t exponent = ((f >> 23U) & 0xffU) - 112U;
                bits = static_cast<std::uint16_t>(((f >> 16U) & 0x8000U) | (exponent << 10U) |
                                                  ((f >> 13U) & 0x3ffU));
            }
        }
        tensor.bytes.push_back(static_cast<std::uint8_t>(bits & 0xffU));
        tensor.bytes.push_back(static_cast<std::uint8_t>(bits >> 8U));
    }
    return tensor;
}

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

void put_u64(std::vector<std::uint8_t>& out, std::uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        out.push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

void put_string(std::vector<std::uint8_t>& out, const std::string& text) {
    put_u64(out, text.size());
    out.insert(out.end(), text.begin(), text.end());
}

std::filesystem::path write(const std::filesystem::path& dir, const std::string& name,
                            const Gguf& gguf) {
    std::vector<std::uint8_t> out;
    put_u32(out, 0x46554747U);
    put_u32(out, 3);
    put_u64(out, gguf.tensors.size());
    put_u64(out, gguf.values.size());
    for (const auto& [key, value] : gguf.values) {
        put_string(out, key);
        if (const auto* text = std::get_if<std::string>(&value)) {
            put_u32(out, 8);
            put_string(out, *text);
        } else if (const auto* number = std::get_if<float>(&value)) {
            put_u32(out, 6);
            put_u32(out, std::bit_cast<std::uint32_t>(*number));
        } else {
            put_u32(out, 4);
            put_u32(out, std::get<std::uint32_t>(value));
        }
    }
    std::uint64_t offset = 0;
    for (const GgufTensor& tensor : gguf.tensors) {
        put_string(out, tensor.name);
        put_u32(out, static_cast<std::uint32_t>(tensor.ne.size()));
        for (const std::uint64_t extent : tensor.ne) { put_u64(out, extent); }
        put_u32(out, tensor.type);
        put_u64(out, offset);
        offset += (tensor.bytes.size() + 31) / 32 * 32;
    }
    for (const GgufTensor& tensor : gguf.tensors) {
        out.resize((out.size() + 31) / 32 * 32);
        out.insert(out.end(), tensor.bytes.begin(), tensor.bytes.end());
    }
    const std::filesystem::path path = dir / (name + ".gguf");
    std::ofstream(path, std::ios::binary)
        .write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
    return path;
}

// B = -r for r_i = (i + 1) / 4: |B|^2 = 93.5.
std::vector<float> removed(float factor = 1.0F) {
    std::vector<float> values(kHidden);
    for (std::uint32_t index = 0; index < kHidden; ++index) {
        values[index] = -0.25F * factor * static_cast<float>(index + 1);
    }
    return values;
}
constexpr double kRemovedNorm2 = 93.5;

void add_site(Gguf& gguf, const std::string& base, std::uint64_t input, bool embedding,
              const std::vector<float>& b, std::uint32_t b_type = kF32, std::uint64_t rank = 1) {
    const std::vector<std::uint64_t> a_ne =
        embedding ? std::vector<std::uint64_t>{rank, input} : std::vector<std::uint64_t>{input, rank};
    gguf.tensors.push_back(
        f32(base + ".lora_a", a_ne, std::vector<float>(input * rank, 0.01F)));
    const std::vector<std::uint64_t> b_ne{rank, kHidden};
    gguf.tensors.push_back(b_type == kF32 ? f32(base + ".lora_b", b_ne, b)
                                          : half(base + ".lora_b", b_ne, b, b_type));
}

// A rank-one adapter removing r at the embedding, layer 0's GDN output, layer 1's FFN output and
// layer 3's attention output, its lora_b in each of the three stored float formats.
Gguf valid_adapter() {
    Gguf gguf;
    gguf.values = {{"general.architecture", std::string("qwen35")},
                   {"general.type", std::string("adapter")},
                   {"adapter.type", std::string("lora")},
                   // The AtomicChat adapters' own key, which llama.cpp does not read.
                   {"adapter.lora_alpha", 1.0F}};
    add_site(gguf, "token_embd.weight", 1000, true, removed());
    add_site(gguf, "blk.0.ssm_out.weight", 16, false, removed(), kF16);
    add_site(gguf, "blk.1.ffn_down.weight", 24, false, removed(), kBF16);
    add_site(gguf, "blk.3.attn_output.weight", 32, false, removed());
    return gguf;
}

bool close(double actual, double expected) {
    return std::abs(actual - expected) <= 1e-6 * std::max(1.0, std::abs(expected));
}

q::Steering load(const std::vector<ninfer::LoraAdapterSource>& adapters,
                 const std::vector<ninfer::ControlVectorSource>& vectors = {},
                 const std::optional<ninfer::ControlVectorLayers>& layers = std::nullopt,
                 const q::TextConfig& text = model()) {
    return q::load_steering(adapters, vectors, layers, text);
}

int expect_rejected(const std::function<void()>& action, const std::string& fragment,
                    const std::string& message) {
    try {
        action();
    } catch (const std::invalid_argument& error) {
        if (std::string(error.what()).find(fragment) != std::string::npos) { return 0; }
        std::cerr << "FAIL: " << message << " (refused with: " << error.what() << ")\n";
        return 1;
    }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int expect_adapter_rejected(const std::filesystem::path& path, const std::string& fragment,
                            const std::string& message) {
    return expect_rejected([&] { (void)load({{.path = path.string()}}); }, fragment, message);
}

GgufTensor direction(std::uint32_t layer, const std::vector<float>& values) {
    return f32("direction." + std::to_string(layer), {values.size()}, values);
}

Gguf control_vector(std::vector<GgufTensor> tensors) {
    Gguf gguf;
    gguf.values  = {{"general.architecture", std::string("controlvector")},
                    {"controlvector.model_hint", std::string("qwen35")},
                    {"controlvector.layer_count", static_cast<std::uint32_t>(tensors.size())}};
    gguf.tensors = std::move(tensors);
    return gguf;
}

std::vector<float> ramp(float offset, float step) {
    std::vector<float> values(kHidden);
    for (std::uint32_t index = 0; index < kHidden; ++index) {
        values[index] = offset + step * static_cast<float>(index);
    }
    return values;
}

} // namespace

int main() {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "ninfer_steering_loader_test";
    std::filesystem::create_directories(dir);
    int failures = 0;

    const std::filesystem::path adapter_path = write(dir, "adapter", valid_adapter());
    {
        const q::Steering steering = load({{.path = adapter_path.string(), .scale = 0.75F}});
        failures += check(steering.adapters.size() == 1 && steering.layer_vectors.empty() &&
                              steering.steered_layers.empty() && !steering.empty(),
                          "one adapter did not load as one direction without control vectors");
        const q::DirectionAdapter& adapter = steering.adapters.front();
        failures += check(adapter.default_scale == 0.75F, "the adapter lost its default scale");
        const std::vector<float> b = removed();
        bool unit                  = true;
        for (std::uint32_t index = 0; index < kHidden; ++index) {
            unit = unit && close(adapter.direction[index], b[index] / std::sqrt(kRemovedNorm2));
        }
        failures += check(unit, "the direction is not lora_b over its norm");
        // Without adapter.lora.alpha llama.cpp scales B A by the request's scale alone, so a site
        // removes |B|^2 times the component along r.
        const std::vector<float>& factors = adapter.site_factors;
        failures += check(factors.size() == q::steering_sites(4), "the site table is not 1 + 2L");
        for (std::uint32_t site = 0; site < factors.size(); ++site) {
            const bool present = site == q::kEmbeddingSteeringSite ||
                                 site == q::mixer_steering_site(0) ||
                                 site == q::ffn_steering_site(1) ||
                                 site == q::mixer_steering_site(3);
            failures += check(close(factors[site], present ? kRemovedNorm2 : 0.0),
                              "site " + std::to_string(site) + " has factor " +
                                  std::to_string(factors[site]));
        }

        const q::Steering again = load({{.path = adapter_path.string(), .scale = 0.75F}});
        failures += check(again.adapters.front().digest == adapter.digest,
                          "the same file gave two digests");
    }
    {
        // llama.cpp's alpha: the factor becomes alpha / rank times |B|^2.
        Gguf alpha = valid_adapter();
        alpha.values.emplace_back("adapter.lora.alpha", 2.0F);
        const q::Steering steering = load({{.path = write(dir, "alpha", alpha).string()}});
        failures += check(close(steering.adapters.front().site_factors[q::ffn_steering_site(1)],
                                2.0 * kRemovedNorm2),
                          "adapter.lora.alpha did not scale the site factor");
        failures += check(steering.adapters.front().digest !=
                              load({{.path = adapter_path.string()}}).adapters.front().digest,
                          "two different files gave one digest");
    }

    // Refusals: anything but a rank-one adapter removing one direction from this model's sites.
    {
        Gguf c = valid_adapter();
        c.tensors.clear();
        add_site(c, "blk.0.ssm_out.weight", 16, false, std::vector<float>(2 * kHidden, 0.5F),
                 kF32, 2);
        failures += expect_adapter_rejected(write(dir, "rank2", c), "rank 2",
                                            "a rank-two adapter was accepted");
    }
    {
        Gguf c = valid_adapter();
        add_site(c, "blk.2.ssm_out.weight", 16, false, removed(2.0F));
        failures += expect_adapter_rejected(write(dir, "differs", c), "different lora_b",
                                            "an adapter whose sites differ was accepted");
    }
    {
        Gguf c = valid_adapter();
        c.tensors.clear();
        add_site(c, "blk.0.ssm_out.weight", 16, false, std::vector<float>(kHidden, 0.0F));
        failures += expect_adapter_rejected(write(dir, "zero", c), "is zero",
                                            "a zero lora_b was accepted");
    }
    {
        Gguf c       = valid_adapter();
        c.values[0].second = std::string("llama");
        failures += expect_adapter_rejected(write(dir, "arch", c), "'llama'",
                                            "an adapter for another architecture was accepted");
    }
    {
        Gguf c       = valid_adapter();
        c.values[2].second = std::string("control_vector");
        failures += expect_adapter_rejected(write(dir, "type", c), "is not a LoRA adapter",
                                            "a file that is not a LoRA adapter was accepted");
    }
    {
        Gguf c = valid_adapter();
        c.values.emplace_back("adapter.alora.invocation_tokens", 7U);
        failures += expect_adapter_rejected(write(dir, "alora", c), "activated LoRA",
                                            "an activated LoRA was accepted");
    }
    {
        Gguf c = valid_adapter();
        c.tensors.clear();
        add_site(c, "blk.0.ssm_out.weight", 17, false, removed());
        failures += expect_adapter_rejected(write(dir, "width", c), "do not fit the model",
                                            "a lora_a of the wrong input width was accepted");
    }
    {
        Gguf c = valid_adapter();
        add_site(c, "blk.1.attn_output.weight", 32, false, removed());
        failures += expect_adapter_rejected(write(dir, "kind", c), "Gated DeltaNet layer",
                                            "an attention output on a GDN layer was accepted");
    }
    {
        Gguf c = valid_adapter();
        add_site(c, "blk.0.attn_q.weight", 16, false, removed());
        failures += expect_adapter_rejected(write(dir, "q", c), "does not apply",
                                            "a query projection adapter was accepted");
    }
    {
        Gguf c = valid_adapter();
        add_site(c, "blk.4.ffn_down.weight", 24, false, removed());
        failures += expect_adapter_rejected(write(dir, "layer", c), "names no layer",
                                            "an adapter past the last layer was accepted");
    }
    {
        Gguf c = valid_adapter();
        c.tensors.pop_back();
        failures += expect_adapter_rejected(write(dir, "pair", c), "lacks its lora_a or its lora_b",
                                            "an unpaired lora_a was accepted");
    }
    {
        q::TextConfig moe = model();
        moe.ffn = q::MoeConfig{.num_experts = 4, .num_experts_per_tok = 2,
                               .moe_intermediate_size = 8, .shared_expert_intermediate_size = 8};
        Gguf c = valid_adapter();
        c.values[0].second = std::string("qwen35moe");
        const auto path    = write(dir, "moe", c);
        failures += expect_rejected(
            [&] { (void)load({{.path = path.string()}}, {}, std::nullopt, moe); }, "MoE",
            "a dense FFN adapter was accepted on an MoE model");
    }
    failures += expect_adapter_rejected(dir / "missing.gguf", "cannot open",
                                        "a missing adapter file was accepted");
    failures += expect_rejected(
        [&] {
            (void)load(std::vector<ninfer::LoraAdapterSource>(
                ninfer::kMaximumLoraAdapters + 1, {.path = adapter_path.string()}));
        },
        "at most", "more adapters than a request can name were accepted");

    // Control vectors: direction.N is added after layer N, files summed by their scales.
    const auto vector_a = write(dir, "cv_a", control_vector({direction(1, ramp(0.0F, 1.0F)),
                                                             direction(3, ramp(1.0F, 0.0F))}));
    const auto vector_b = write(dir, "cv_b", control_vector({direction(3, ramp(0.0F, 1.0F))}));
    const std::vector<ninfer::ControlVectorSource> both{{.path = vector_a.string(), .scale = 2.0F},
                                                        {.path = vector_b.string(), .scale = 0.5F}};
    {
        const q::Steering steering = load({}, both);
        failures += check(steering.adapters.empty() && !steering.empty(),
                          "control vectors alone did not load");
        failures += check(steering.steered_layers == std::vector<std::uint8_t>{0, 1, 0, 1},
                          "the steered layers are not those the files name");
        bool sums = steering.layer_vectors.size() == 4 * kHidden;
        for (std::uint32_t index = 0; sums && index < kHidden; ++index) {
            sums = steering.layer_vectors[index] == 0.0F &&
                   steering.layer_vectors[kHidden + index] == 2.0F * static_cast<float>(index) &&
                   steering.layer_vectors[2 * kHidden + index] == 0.0F &&
                   steering.layer_vectors[3 * kHidden + index] ==
                       2.0F + 0.5F * static_cast<float>(index);
        }
        failures += check(sums, "the layer rows are not the scaled sums of the files");

        const q::Steering ranged = load({}, both, ninfer::ControlVectorLayers{.first = 2, .last = 3});
        failures += check(ranged.steered_layers == std::vector<std::uint8_t>{0, 0, 0, 1} &&
                              ranged.layer_vectors[kHidden + 5] == 0.0F,
                          "the layer range kept a layer outside it");
        failures += check(ranged.control_digest != steering.control_digest,
                          "the layer range did not change the control vectors' digest");
        std::vector<ninfer::ControlVectorSource> rescaled = both;
        rescaled[1].scale                                 = 0.25F;
        failures += check(load({}, rescaled).control_digest != steering.control_digest,
                          "a scale did not change the control vectors' digest");
        failures += check(load({}, both).control_digest == steering.control_digest,
                          "the same control vectors gave two digests");
        failures += expect_rejected(
            [&] { (void)load({}, both, ninfer::ControlVectorLayers{.first = 2, .last = 2}); },
            "holds no layer", "a layer range that steers nothing was accepted");
    }
    const auto expect_vector_rejected = [&](const Gguf& gguf, const std::string& name,
                                            const std::string& fragment,
                                            const std::string& message) {
        const auto path = write(dir, name, gguf);
        return expect_rejected([&] { (void)load({}, {{.path = path.string()}}); }, fragment,
                               message);
    };
    failures += expect_vector_rejected(control_vector({direction(0, ramp(0.0F, 1.0F))}), "cv0",
                                       "direction.N tensor, N >= 1",
                                       "a control vector for layer 0 was accepted");
    failures += expect_vector_rejected(control_vector({direction(4, ramp(0.0F, 1.0F))}), "cv4",
                                       "past the model",
                                       "a control vector past the last layer was accepted");
    failures += expect_vector_rejected(
        control_vector({f32("direction.1", {kHidden - 1}, std::vector<float>(kHidden - 1, 1.0F))}),
        "cv_width", "the model's residual", "a control vector of the wrong width was accepted");
    failures += expect_vector_rejected(
        control_vector({half("direction.1", {kHidden}, ramp(0.0F, 1.0F), kF16)}), "cv_f16",
        "F32 tensor", "an F16 control vector was accepted");
    failures += expect_vector_rejected(control_vector({}), "cv_empty", "holds no direction",
                                       "an empty control vector was accepted");

    // A request's scales: llama.cpp's `lora` field and the cache identity they give its prompt.
    {
        Gguf second_file = valid_adapter();
        second_file.values.emplace_back("adapter.lora.alpha", 3.0F);
        const auto second = write(dir, "second", second_file);
        const q::Steering steering =
            load({{.path = adapter_path.string(), .scale = 1.5F}, {.path = second.string()}});
        failures += check(q::resolve_lora_scales(steering, std::nullopt) ==
                              std::vector<float>{1.5F, 1.0F},
                          "an unset lora field did not take the default scales");
        failures += check(q::resolve_lora_scales(steering, std::vector<ninfer::LoraScale>{
                                                               {.id = 1, .scale = 0.5F}}) ==
                              std::vector<float>{0.0F, 0.5F},
                          "an adapter the lora field leaves out did not run at scale 0");
        failures += check(q::resolve_lora_scales(steering, std::vector<ninfer::LoraScale>{}) ==
                              std::vector<float>{0.0F, 0.0F},
                          "an empty lora field did not turn every adapter off");
        failures += check(q::resolve_lora_scales(steering,
                                                 std::vector<ninfer::LoraScale>{
                                                     {.id = 0, .scale = 0.5F},
                                                     {.id = 0, .scale = -1.0F}}) ==
                              std::vector<float>{-1.0F, 0.0F},
                          "a repeated id did not take its last scale");
        failures += expect_rejected(
            [&] {
                (void)q::resolve_lora_scales(
                    steering, std::vector<ninfer::LoraScale>{{.id = 2, .scale = 1.0F}});
            },
            "not a loaded adapter", "an id past the loaded adapters was accepted");
        failures += expect_rejected(
            [&] {
                (void)q::resolve_lora_scales(
                    steering, std::vector<ninfer::LoraScale>{{.id = 0, .scale = NAN}});
            },
            "not finite", "a NaN scale was accepted");

        using Key       = std::array<std::uint64_t, 2>;
        const Key zero  = q::lora_identity(steering, std::vector<float>{0.0F, 0.0F});
        const Key one   = q::lora_identity(steering, std::vector<float>{1.5F, 0.0F});
        const Key other = q::lora_identity(steering, std::vector<float>{1.25F, 0.0F});
        const Key pair  = q::lora_identity(steering, std::vector<float>{1.5F, 0.5F});
        failures += check(zero == Key{}, "scales of zero changed the prompt's cache identity");
        failures += check(one != Key{} && one != other && one != pair,
                          "different scales gave one cache identity");
        failures += check(one == q::lora_identity(steering, std::vector<float>{1.5F, 0.0F}),
                          "the same scales gave two cache identities");
        // The edits add up, so the order the adapters were loaded in is not part of the identity.
        const q::Steering swapped =
            load({{.path = second.string()}, {.path = adapter_path.string()}});
        failures += check(q::lora_identity(swapped, std::vector<float>{0.5F, 1.5F}) == pair,
                          "the adapters' load order changed the cache identity");
        // An adapter at scale zero is no edit: an Engine that does not load it computes the same.
        const q::Steering alone = load({{.path = adapter_path.string()}});
        failures += check(q::lora_identity(alone, std::vector<float>{1.5F}) == one,
                          "an adapter at scale zero changed the cache identity");
        failures += expect_rejected(
            [&] { (void)q::lora_identity(steering, std::vector<float>{1.0F}); }, "do not match",
            "scales for a different adapter count were accepted");
    }

    // Engine options: the adapter count and a layer range without vectors.
    {
        const auto rejected = [](ninfer::EngineOptions options) {
            try {
                (void)ninfer::runtime::normalize_engine_options(std::move(options));
            } catch (const std::invalid_argument&) { return true; }
            return false;
        };
        ninfer::EngineOptions plain;
        ninfer::EngineOptions many = plain;
        many.lora_adapters.assign(ninfer::kMaximumLoraAdapters + 1, {.path = "a.gguf"});
        failures += check(rejected(many), "more than eight adapters were accepted");
        ninfer::EngineOptions range = plain;
        range.control_vector_layers = ninfer::ControlVectorLayers{.first = 1, .last = 2};
        failures += check(rejected(range), "a layer range without control vectors was accepted");
        ninfer::EngineOptions steered = plain;
        steered.lora_adapters.push_back({.path = "a.gguf"});
        steered.control_vectors.push_back({.path = "v.gguf"});
        steered.control_vector_layers = ninfer::ControlVectorLayers{.first = 1, .last = 2};
        failures += check(!rejected(steered), "an adapter with control vectors was refused");
    }

    std::filesystem::remove_all(dir);
    if (failures != 0) {
        std::cerr << failures << " steering loader checks failed\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}

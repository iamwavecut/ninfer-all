#include "models/qwen3_5/frontend/steering.h"

#include "models/qwen3_5/frontend/digest.h"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace ninfer::models::qwen3_5 {
namespace {

class SteeringError {
public:
    SteeringError(std::string_view kind, const std::filesystem::path& path)
        : prefix_(std::string(kind) + " " + path.string() + ": ") {}

    [[noreturn]] void fail(const std::string& message) const {
        throw std::invalid_argument(prefix_ + message);
    }

    void require(bool condition, const std::string& message) const {
        if (!condition) { fail(message); }
    }

private:
    std::string prefix_;
};

std::vector<std::uint8_t> read_file(const std::filesystem::path& path, const SteeringError& error) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    error.require(in.good(), "cannot open the file");
    const std::streamoff size = in.tellg();
    error.require(size >= 0, "cannot size the file");
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(bytes.data()), size);
    error.require(in.good() || in.eof(), "cannot read the file");
    return bytes;
}

// GGUF value and tensor types this reader meets (ggml.h, gguf.h).
enum GgufType : std::uint32_t {
    kGgufU8     = 0,
    kGgufI8     = 1,
    kGgufU16    = 2,
    kGgufI16    = 3,
    kGgufU32    = 4,
    kGgufI32    = 5,
    kGgufF32    = 6,
    kGgufBool   = 7,
    kGgufString = 8,
    kGgufArray  = 9,
    kGgufU64    = 10,
    kGgufI64    = 11,
    kGgufF64    = 12,
};
constexpr std::uint32_t kGgmlF32  = 0;
constexpr std::uint32_t kGgmlF16  = 1;
constexpr std::uint32_t kGgmlBF16 = 30;

struct GgufTensor {
    std::string name;
    std::array<std::uint64_t, 4> ne{1, 1, 1, 1};
    std::uint32_t dims   = 0;
    std::uint32_t type   = 0;
    std::uint64_t offset = 0;
};

// The parts of a GGUF file the steering inputs use: string and numeric metadata, and the tensor
// directory with each tensor's absolute file offset. Arrays are skipped.
struct GgufFile {
    std::vector<std::uint8_t> bytes;
    std::map<std::string, std::string, std::less<>> strings;
    std::map<std::string, double, std::less<>> numbers;
    std::map<std::string, std::uint32_t, std::less<>> types;
    std::vector<GgufTensor> tensors;
};

class Cursor {
public:
    Cursor(std::span<const std::uint8_t> bytes, const SteeringError& error)
        : bytes_(bytes), error_(error) {}

    template <class T>
    T read() {
        need(sizeof(T));
        T value;
        std::memcpy(&value, bytes_.data() + position_, sizeof(T));
        position_ += sizeof(T);
        return value;
    }

    std::string string() {
        const auto size = read<std::uint64_t>();
        need(size);
        std::string value(reinterpret_cast<const char*>(bytes_.data() + position_),
                          static_cast<std::size_t>(size));
        position_ += static_cast<std::size_t>(size);
        return value;
    }

    void skip(std::uint64_t size) {
        need(size);
        position_ += static_cast<std::size_t>(size);
    }

    [[nodiscard]] std::size_t position() const noexcept { return position_; }

private:
    void need(std::uint64_t size) const {
        error_.require(size <= bytes_.size() - position_, "GGUF data runs past the end of the file");
    }

    std::span<const std::uint8_t> bytes_;
    const SteeringError& error_;
    std::size_t position_ = 0;
};

std::uint64_t scalar_bytes(std::uint32_t type, const SteeringError& error) {
    switch (type) {
    case kGgufU8:
    case kGgufI8:
    case kGgufBool: return 1;
    case kGgufU16:
    case kGgufI16: return 2;
    case kGgufU32:
    case kGgufI32:
    case kGgufF32: return 4;
    case kGgufU64:
    case kGgufI64:
    case kGgufF64: return 8;
    default: error.fail("unknown GGUF value type " + std::to_string(type));
    }
}

void skip_value(Cursor& cursor, std::uint32_t type, const SteeringError& error, int depth = 0) {
    if (type == kGgufString) {
        (void)cursor.string();
        return;
    }
    if (type != kGgufArray) {
        cursor.skip(scalar_bytes(type, error));
        return;
    }
    error.require(depth < 4, "GGUF arrays nest too deeply");
    const auto element = cursor.read<std::uint32_t>();
    const auto count   = cursor.read<std::uint64_t>();
    if (element == kGgufString || element == kGgufArray) {
        for (std::uint64_t index = 0; index < count; ++index) {
            skip_value(cursor, element, error, depth + 1);
        }
        return;
    }
    const std::uint64_t size = scalar_bytes(element, error);
    error.require(count <= std::numeric_limits<std::uint64_t>::max() / size,
                  "a GGUF array size overflows");
    cursor.skip(count * size);
}

double read_number(Cursor& cursor, std::uint32_t type, const SteeringError& error) {
    switch (type) {
    case kGgufU8: return cursor.read<std::uint8_t>();
    case kGgufI8: return cursor.read<std::int8_t>();
    case kGgufU16: return cursor.read<std::uint16_t>();
    case kGgufI16: return cursor.read<std::int16_t>();
    case kGgufU32: return cursor.read<std::uint32_t>();
    case kGgufI32: return cursor.read<std::int32_t>();
    case kGgufF32: return cursor.read<float>();
    case kGgufBool: return cursor.read<std::uint8_t>() != 0 ? 1.0 : 0.0;
    case kGgufU64: return static_cast<double>(cursor.read<std::uint64_t>());
    case kGgufI64: return static_cast<double>(cursor.read<std::int64_t>());
    case kGgufF64: return cursor.read<double>();
    default: error.fail("unknown GGUF value type " + std::to_string(type));
    }
}

GgufFile read_gguf(const std::filesystem::path& path, const SteeringError& error) {
    GgufFile file;
    file.bytes = read_file(path, error);
    Cursor cursor(file.bytes, error);
    const auto magic = cursor.read<std::uint32_t>();
    error.require(magic == 0x46554747U, "is not a GGUF file");
    const auto version = cursor.read<std::uint32_t>();
    error.require(version == 2 || version == 3,
                  "GGUF version " + std::to_string(version) + " is not supported");
    const auto tensor_count = cursor.read<std::uint64_t>();
    const auto value_count  = cursor.read<std::uint64_t>();
    error.require(tensor_count <= file.bytes.size() && value_count <= file.bytes.size(),
                  "GGUF counts exceed the file");
    for (std::uint64_t index = 0; index < value_count; ++index) {
        std::string key  = cursor.string();
        const auto type  = cursor.read<std::uint32_t>();
        file.types[key] = type;
        if (type == kGgufString) {
            file.strings[key] = cursor.string();
        } else if (type == kGgufArray) {
            skip_value(cursor, type, error);
        } else {
            file.numbers[key] = read_number(cursor, type, error);
        }
    }
    file.tensors.reserve(static_cast<std::size_t>(tensor_count));
    for (std::uint64_t index = 0; index < tensor_count; ++index) {
        GgufTensor tensor;
        tensor.name = cursor.string();
        tensor.dims = cursor.read<std::uint32_t>();
        error.require(tensor.dims >= 1 && tensor.dims <= 4,
                      "tensor '" + tensor.name + "' has " + std::to_string(tensor.dims) +
                          " dimensions");
        for (std::uint32_t dim = 0; dim < tensor.dims; ++dim) {
            tensor.ne[dim] = cursor.read<std::uint64_t>();
        }
        tensor.type   = cursor.read<std::uint32_t>();
        tensor.offset = cursor.read<std::uint64_t>();
        file.tensors.push_back(std::move(tensor));
    }
    std::uint64_t alignment = 32;
    if (const auto found = file.numbers.find("general.alignment"); found != file.numbers.end()) {
        alignment = static_cast<std::uint64_t>(found->second);
    }
    error.require(alignment != 0 && std::has_single_bit(alignment),
                  "general.alignment is not a power of two");
    const std::uint64_t data = (cursor.position() + alignment - 1) / alignment * alignment;
    for (GgufTensor& tensor : file.tensors) {
        constexpr std::uint64_t kElementLimit = 1ULL << 40U;
        std::uint64_t elements                = 1;
        for (const std::uint64_t extent : tensor.ne) {
            error.require(extent != 0 && extent <= kElementLimit &&
                              elements <= kElementLimit / extent,
                          "tensor '" + tensor.name + "' has an invalid shape");
            elements *= extent;
        }
        std::uint64_t element_bytes = 0;
        if (tensor.type == kGgmlF32) { element_bytes = 4; }
        if (tensor.type == kGgmlF16 || tensor.type == kGgmlBF16) { element_bytes = 2; }
        error.require(element_bytes != 0, "tensor '" + tensor.name + "' is ggml type " +
                                              std::to_string(tensor.type) +
                                              "; F32, F16 and BF16 are read");
        error.require(tensor.offset <= file.bytes.size() &&
                          data <= file.bytes.size() - tensor.offset &&
                          elements * element_bytes <= file.bytes.size() - data - tensor.offset,
                      "tensor '" + tensor.name + "' runs past the end of the file");
        tensor.offset += data;
    }
    return file;
}

std::uint64_t elements(const GgufTensor& tensor) {
    return tensor.ne[0] * tensor.ne[1] * tensor.ne[2] * tensor.ne[3];
}

float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign     = (bits & 0x8000U) << 16U;
    const std::uint32_t exponent = (bits >> 10U) & 0x1fU;
    std::uint32_t mantissa       = bits & 0x3ffU;
    std::uint32_t out            = 0;
    if (exponent == 0x1fU) {
        out = sign | 0x7f800000U | (mantissa << 13U);
    } else if (exponent != 0) {
        out = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    } else if (mantissa != 0) {
        std::uint32_t shift = 0;
        while ((mantissa & 0x400U) == 0) {
            mantissa <<= 1U;
            ++shift;
        }
        out = sign | ((113U - shift) << 23U) | ((mantissa & 0x3ffU) << 13U);
    } else {
        out = sign;
    }
    return std::bit_cast<float>(out);
}

std::vector<float> tensor_values(const GgufFile& file, const GgufTensor& tensor) {
    const auto count = static_cast<std::size_t>(elements(tensor));
    std::vector<float> values(count);
    const std::uint8_t* data = file.bytes.data() + tensor.offset;
    for (std::size_t index = 0; index < count; ++index) {
        if (tensor.type == kGgmlF32) {
            std::memcpy(&values[index], data + 4 * index, 4);
        } else {
            std::uint16_t bits = 0;
            std::memcpy(&bits, data + 2 * index, 2);
            values[index] = tensor.type == kGgmlF16
                                ? half_to_float(bits)
                                : std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
        }
    }
    return values;
}

std::string shape_text(const GgufTensor& tensor) {
    std::string text = "[";
    for (std::uint32_t dim = 0; dim < tensor.dims; ++dim) {
        if (dim != 0) { text += ", "; }
        text += std::to_string(tensor.ne[dim]);
    }
    return text + "]";
}

std::string text_value(const GgufFile& file, std::string_view key) {
    const auto found = file.strings.find(key);
    return found == file.strings.end() ? std::string() : found->second;
}

bool finite(float value) { return std::isfinite(value); }

// A layer index written as decimal digits, nothing else.
std::optional<std::uint32_t> parse_index(std::string_view text) {
    std::uint32_t value = 0;
    const auto* end     = text.data() + text.size();
    const auto [stop, error] = std::from_chars(text.data(), end, value);
    if (text.empty() || error != std::errc() || stop != end) { return std::nullopt; }
    return value;
}

// Which site an adapter tensor (its name without .lora_a / .lora_b) writes, and the width its
// lora_a reads: the vocabulary for the embedding.
struct AdapterSite {
    std::uint32_t site  = 0;
    std::uint64_t input = 0;
};

AdapterSite adapter_site(std::string_view base, const TextConfig& text, const SteeringError& error) {
    if (base == "token_embd.weight") {
        return {kEmbeddingSteeringSite, text.vocab_size};
    }
    const std::string name(base);
    error.require(base.starts_with("blk."), "tensor '" + name +
                                                "' is not a site this engine applies: only "
                                                "token_embd, attn_output, ssm_out and ffn_down");
    const std::string_view rest = base.substr(4);
    const std::size_t dot       = rest.find('.');
    const auto layer =
        dot == std::string_view::npos ? std::nullopt : parse_index(rest.substr(0, dot));
    error.require(layer.has_value() && *layer < text.num_hidden_layers,
                  "tensor '" + name + "' names no layer of the model");
    const std::string_view kind = rest.substr(dot + 1);
    const bool full = text.layer_types.at(*layer) == MixerKind::FullAttention;
    if (kind == "attn_output.weight") {
        error.require(full && text.attention.has_value(),
                      "tensor '" + name + "' writes a full-attention output, but layer " +
                          std::to_string(*layer) + " is a Gated DeltaNet layer");
        return {mixer_steering_site(*layer), text.attention->query_width()};
    }
    if (kind == "ssm_out.weight") {
        error.require(!full && text.gdn.has_value(),
                      "tensor '" + name + "' writes a Gated DeltaNet output, but layer " +
                          std::to_string(*layer) + " is a full-attention layer");
        return {mixer_steering_site(*layer), text.gdn->value_width()};
    }
    if (kind == "ffn_down.weight") {
        const auto* dense = std::get_if<DenseConfig>(&text.ffn);
        error.require(dense != nullptr,
                      "tensor '" + name + "' is a dense FFN, but the model's FFN layers are MoE");
        return {ffn_steering_site(*layer), dense->intermediate_size};
    }
    error.fail("tensor '" + name +
               "' is a LoRA site this engine does not apply: it runs adapters that remove one "
               "direction from what token_embd, attn_output, ssm_out and ffn_down write");
}

DirectionAdapter load_adapter(const LoraAdapterSource& source, const TextConfig& text) {
    const SteeringError error("LoRA adapter", source.path);
    error.require(finite(source.scale), "its scale is not finite");
    const GgufFile file = read_gguf(source.path, error);
    error.require(text_value(file, "general.type") == "adapter" &&
                      text_value(file, "adapter.type") == "lora",
                  "is not a LoRA adapter (general.type adapter, adapter.type lora)");
    error.require(!file.types.contains("adapter.alora.invocation_tokens"),
                  "is an activated LoRA, which this engine does not run");
    const std::string expected =
        std::holds_alternative<MoeConfig>(text.ffn) ? "qwen35moe" : "qwen35";
    const std::string architecture = text_value(file, "general.architecture");
    error.require(architecture == expected, "is an adapter for '" + architecture +
                                                "', and the model is '" + expected + "'");
    float alpha = 0.0F;
    if (const auto found = file.numbers.find("adapter.lora.alpha"); found != file.numbers.end()) {
        error.require(file.types.at("adapter.lora.alpha") == kGgufF32,
                      "adapter.lora.alpha is not an F32 value");
        alpha = static_cast<float>(found->second);
        error.require(finite(alpha), "adapter.lora.alpha is not finite");
    }

    struct Pair {
        const GgufTensor* a = nullptr;
        const GgufTensor* b = nullptr;
    };
    std::map<std::string, Pair, std::less<>> pairs;
    for (const GgufTensor& tensor : file.tensors) {
        const std::string_view name = tensor.name;
        if (name.ends_with(".lora_a")) {
            pairs[std::string(name.substr(0, name.size() - 7))].a = &tensor;
        } else if (name.ends_with(".lora_b")) {
            pairs[std::string(name.substr(0, name.size() - 7))].b = &tensor;
        } else {
            error.fail("tensor '" + tensor.name + "' is neither a lora_a nor a lora_b");
        }
    }
    error.require(!pairs.empty(), "holds no LoRA tensors");

    DirectionAdapter adapter;
    adapter.path          = source.path;
    adapter.default_scale = source.scale;
    adapter.site_factors.assign(steering_sites(text.num_hidden_layers), 0.0F);
    std::vector<float> first;
    std::string first_name;
    for (const auto& [base, pair] : pairs) {
        error.require(pair.a != nullptr && pair.b != nullptr,
                      "'" + base + "' lacks its lora_a or its lora_b");
        const AdapterSite site = adapter_site(base, text, error);
        const GgufTensor& a    = *pair.a;
        const GgufTensor& b    = *pair.b;
        const bool embedding   = site.site == kEmbeddingSteeringSite;
        // llama.cpp's layouts: B [rank, hidden]; A [input, rank], the embedding's [rank, vocab].
        const std::uint64_t rank = b.ne[0];
        const bool shaped =
            a.ne[2] == 1 && a.ne[3] == 1 && b.ne[2] == 1 && b.ne[3] == 1 &&
            b.ne[1] == text.hidden_size &&
            (embedding ? a.ne[0] == rank && a.ne[1] == site.input
                       : a.ne[0] == site.input && a.ne[1] == rank);
        error.require(shaped, "'" + base + "' has lora_a " + shape_text(a) + " and lora_b " +
                                  shape_text(b) + ", which do not fit the model");
        error.require(rank == 1, "'" + base + "' has rank " + std::to_string(rank) +
                                     "; this engine runs rank-one adapters that remove one "
                                     "direction, and a general LoRA is not supported");
        std::vector<float> values = tensor_values(file, b);
        double norm2              = 0.0;
        for (const float value : values) {
            error.require(finite(value), "'" + base + "' lora_b holds a value that is not finite");
            norm2 += static_cast<double>(value) * value;
        }
        error.require(norm2 > 0.0, "'" + base + "' lora_b is zero");
        if (first.empty()) {
            first      = values;
            first_name = base;
        } else {
            // Every site removes the same direction r (B = -r): an adapter whose sites differ
            // is a general LoRA.
            float largest = 0.0F;
            float differs = 0.0F;
            for (std::size_t index = 0; index < values.size(); ++index) {
                largest = std::max(largest, std::abs(first[index]));
                differs = std::max(differs, std::abs(values[index] - first[index]));
            }
            error.require(differs <= 1e-6F * largest,
                          "'" + base + "' and '" + first_name +
                              "' have different lora_b; this engine runs adapters whose every "
                              "site removes the same direction, and a general LoRA is not "
                              "supported");
        }
        const double per_rank = alpha != 0.0F ? static_cast<double>(alpha) / rank : 1.0;
        adapter.site_factors[site.site] = static_cast<float>(per_rank * norm2);
    }
    double norm2 = 0.0;
    for (const float value : first) { norm2 += static_cast<double>(value) * value; }
    const double inverse = 1.0 / std::sqrt(norm2);
    adapter.direction.resize(first.size());
    for (std::size_t index = 0; index < first.size(); ++index) {
        adapter.direction[index] = static_cast<float>(first[index] * inverse);
    }
    adapter.digest = frontend::sha256(std::span<const std::uint8_t>(file.bytes));
    return adapter;
}

} // namespace

Steering load_steering(const std::vector<LoraAdapterSource>& adapters,
                       const std::vector<ControlVectorSource>& vectors,
                       const std::optional<ControlVectorLayers>& layers, const TextConfig& text) {
    if (adapters.size() > kMaximumLoraAdapters) {
        throw std::invalid_argument("at most " + std::to_string(kMaximumLoraAdapters) +
                                    " LoRA adapters can be loaded");
    }
    if (layers && (layers->first == 0 || layers->first > layers->last)) {
        throw std::invalid_argument("the control vector layer range must be 1 <= first <= last");
    }
    Steering steering;
    steering.adapters.reserve(adapters.size());
    for (const LoraAdapterSource& source : adapters) {
        steering.adapters.push_back(load_adapter(source, text));
    }
    if (vectors.empty()) { return steering; }

    const std::uint32_t hidden = text.hidden_size;
    steering.layer_vectors.assign(std::size_t(text.num_hidden_layers) * hidden, 0.0F);
    steering.steered_layers.assign(text.num_hidden_layers, 0);
    std::vector<std::uint8_t> identity;
    for (const ControlVectorSource& source : vectors) {
        const SteeringError error("control vector", source.path);
        error.require(finite(source.scale), "its scale is not finite");
        const GgufFile file = read_gguf(source.path, error);
        error.require(!file.tensors.empty(), "holds no direction tensors");
        for (const GgufTensor& tensor : file.tensors) {
            const std::string_view name = tensor.name;
            const auto layer =
                name.starts_with("direction.") ? parse_index(name.substr(10)) : std::nullopt;
            error.require(layer.has_value() && *layer != 0,
                          "tensor '" + tensor.name + "' is not a direction.N tensor, N >= 1");
            error.require(*layer < text.num_hidden_layers,
                          "tensor '" + tensor.name + "' names a layer past the model's " +
                              std::to_string(text.num_hidden_layers));
            error.require(tensor.type == kGgmlF32 && tensor.dims == 1,
                          "tensor '" + tensor.name + "' is not a one-dimensional F32 tensor");
            error.require(tensor.ne[0] == hidden, "tensor '" + tensor.name + "' has " +
                                                      std::to_string(tensor.ne[0]) +
                                                      " values and the model's residual " +
                                                      std::to_string(hidden));
            if (layers && (*layer < layers->first || *layer > layers->last)) { continue; }
            const std::vector<float> values = tensor_values(file, tensor);
            float* row = steering.layer_vectors.data() + std::size_t(*layer) * hidden;
            for (std::uint32_t index = 0; index < hidden; ++index) {
                error.require(finite(values[index]),
                              "tensor '" + tensor.name + "' holds a value that is not finite");
                row[index] += source.scale * values[index];
            }
            steering.steered_layers[*layer] = 1;
        }
        const auto digest = frontend::sha256(std::span<const std::uint8_t>(file.bytes));
        identity.insert(identity.end(), digest.begin(), digest.end());
        const auto scale = std::bit_cast<std::array<std::uint8_t, 4>>(source.scale);
        identity.insert(identity.end(), scale.begin(), scale.end());
    }
    if (layers) {
        const auto range = std::bit_cast<std::array<std::uint8_t, 8>>(
            std::array<std::uint32_t, 2>{layers->first, layers->last});
        identity.insert(identity.end(), range.begin(), range.end());
    }
    steering.control_digest = frontend::sha256(std::span<const std::uint8_t>(identity));
    if (std::none_of(steering.steered_layers.begin(), steering.steered_layers.end(),
                     [](std::uint8_t steered) { return steered != 0; })) {
        throw std::invalid_argument(
            "the control vector layer range holds no layer the control vectors name");
    }
    return steering;
}

std::vector<float> resolve_lora_scales(const Steering& steering,
                                       const std::optional<std::vector<LoraScale>>& lora) {
    std::vector<float> scales(steering.adapters.size(), 0.0F);
    if (!lora) {
        for (std::size_t index = 0; index < scales.size(); ++index) {
            scales[index] = steering.adapters[index].default_scale;
        }
        return scales;
    }
    for (const LoraScale& entry : *lora) {
        if (entry.id >= scales.size()) {
            throw std::invalid_argument("lora id " + std::to_string(entry.id) +
                                        " is not a loaded adapter (" +
                                        std::to_string(scales.size()) + " loaded)");
        }
        if (!std::isfinite(entry.scale)) {
            throw std::invalid_argument("lora id " + std::to_string(entry.id) +
                                        " has a scale that is not finite");
        }
        scales[entry.id] = entry.scale;
    }
    return scales;
}

std::array<std::uint64_t, 2> lora_identity(const Steering& steering, std::span<const float> scales) {
    if (scales.size() != steering.adapters.size()) {
        throw std::invalid_argument("lora scales do not match the loaded adapters");
    }
    // The adapters' edits add up, so the identity is the sorted set of (file, scale) pairs.
    std::vector<std::array<std::uint8_t, 36>> entries;
    for (std::size_t index = 0; index < scales.size(); ++index) {
        if (scales[index] == 0.0F) { continue; }
        std::array<std::uint8_t, 36> entry{};
        std::copy(steering.adapters[index].digest.begin(), steering.adapters[index].digest.end(),
                  entry.begin());
        const auto scale = std::bit_cast<std::array<std::uint8_t, 4>>(scales[index]);
        std::copy(scale.begin(), scale.end(), entry.begin() + 32);
        entries.push_back(entry);
    }
    if (entries.empty()) { return {}; }
    std::sort(entries.begin(), entries.end());
    std::vector<std::uint8_t> bytes;
    bytes.reserve(entries.size() * 36);
    for (const auto& entry : entries) { bytes.insert(bytes.end(), entry.begin(), entry.end()); }
    const auto digest = frontend::sha256(std::span<const std::uint8_t>(bytes));
    std::array<std::uint64_t, 2> out{};
    std::memcpy(out.data(), digest.data(), sizeof(out));
    if (out[0] == 0 && out[1] == 0) { out[0] = 1; }
    return out;
}

} // namespace ninfer::models::qwen3_5

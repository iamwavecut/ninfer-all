#include "models/qwen3_5/frontend/processor.h"

#include "media/decode/decode.h"
#include "models/qwen3_5/frontend/digest.h"
#include "models/qwen3_5/frontend/media_cache.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iterator>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {
namespace {

[[noreturn]] void throw_decode_error(const media::decode::Error& error) {
    switch (error.kind()) {
    case media::decode::ErrorKind::BudgetExceeded:
        throw ProcessorError(ProcessorErrorKind::BudgetExceeded, error.what());
    case media::decode::ErrorKind::InvalidInput:
        throw ProcessorError(ProcessorErrorKind::InvalidMedia, error.what());
    }
    throw std::logic_error("unknown media decode error kind");
}

using Clock = std::chrono::steady_clock;

constexpr int kPatch                              = 16;
constexpr int kTemporal                           = 2;
constexpr int kMerge                              = 2;
constexpr int kFactor                             = kPatch * kMerge;
constexpr int kPatchFeatures                      = 3 * kTemporal * kPatch * kPatch;
constexpr std::uint64_t kMinimumRawPatchesPerItem = kMerge * kMerge;
constexpr std::string_view kImagePad              = "<|image_pad|>";
constexpr std::string_view kVideoPad              = "<|video_pad|>";
constexpr std::string_view kVisionStart           = "<|vision_start|>";
constexpr std::string_view kVisionEnd             = "<|vision_end|>";

// Gemma 4: an image's soft tokens between <|image> and <image|>; a video frame's after its
// mm:ss time, between the same two.
constexpr std::string_view kGemmaBoi = "<|image>";
constexpr std::string_view kGemmaEoi = "<image|>";

struct Size {
    int h = 0;
    int w = 0;
};

struct Prepared {
    VisionItem item;
    std::shared_ptr<qwen3_5::PreparedMediaPayload> payload;
};

static_assert(kPatchFeatures == static_cast<int>(kPreparedVisionPatchFeatures));

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::invalid_argument(std::string(label) + " overflow");
    }
    return a * b;
}

int round_even(double value) { return static_cast<int>(std::nearbyint(value)); }

Size smart_resize_image(int height, int width, std::uint64_t min_pixels, std::uint64_t max_pixels) {
    if (height <= 0 || width <= 0 || min_pixels == 0 || max_pixels < min_pixels) {
        throw std::invalid_argument("invalid image resize configuration");
    }
    if (static_cast<double>(std::max(height, width)) / std::min(height, width) > 200.0) {
        throw std::invalid_argument("image aspect ratio must be at most 200");
    }
    int h                    = round_even(static_cast<double>(height) / kFactor) * kFactor;
    int w                    = round_even(static_cast<double>(width) / kFactor) * kFactor;
    const std::uint64_t area = checked_mul(std::max(h, 0), std::max(w, 0), "image area");
    if (area > max_pixels) {
        const double beta = std::sqrt(static_cast<double>(height) * width / max_pixels);
        h = std::max(kFactor, static_cast<int>(std::floor(height / beta / kFactor)) * kFactor);
        w = std::max(kFactor, static_cast<int>(std::floor(width / beta / kFactor)) * kFactor);
    } else if (area < min_pixels) {
        const double beta =
            std::sqrt(static_cast<double>(min_pixels) / (static_cast<double>(height) * width));
        h = static_cast<int>(std::ceil(height * beta / kFactor)) * kFactor;
        w = static_cast<int>(std::ceil(width * beta / kFactor)) * kFactor;
    }
    return {h, w};
}

Size smart_resize_video(int frames, int height, int width, std::uint64_t min_pixels,
                        std::uint64_t max_pixels) {
    if (frames <= 0 || height < kFactor || width < kFactor || min_pixels == 0 ||
        max_pixels < min_pixels) {
        throw std::invalid_argument("invalid video resize configuration");
    }
    if (static_cast<double>(std::max(height, width)) / std::min(height, width) > 200.0) {
        throw std::invalid_argument("video aspect ratio must be at most 200");
    }
    int h                      = round_even(static_cast<double>(height) / kFactor) * kFactor;
    int w                      = round_even(static_cast<double>(width) / kFactor) * kFactor;
    const int padded_frames    = ((frames + kTemporal - 1) / kTemporal) * kTemporal;
    const std::uint64_t volume = checked_mul(
        static_cast<std::uint64_t>(padded_frames),
        checked_mul(static_cast<std::uint64_t>(h), static_cast<std::uint64_t>(w), "video area"),
        "video pixels");
    if (volume > max_pixels) {
        const double beta = std::sqrt(static_cast<double>(frames) * height * width / max_pixels);
        h = std::max(kFactor, static_cast<int>(std::floor(height / beta / kFactor)) * kFactor);
        w = std::max(kFactor, static_cast<int>(std::floor(width / beta / kFactor)) * kFactor);
    } else if (volume < min_pixels) {
        const double beta = std::sqrt(static_cast<double>(min_pixels) /
                                      (static_cast<double>(frames) * height * width));
        h                 = static_cast<int>(std::ceil(height * beta / kFactor)) * kFactor;
        w                 = static_cast<int>(std::ceil(width * beta / kFactor)) * kFactor;
    }
    return {h, w};
}

double cubic(double x) {
    // Torchvision's antialiased bicubic path uses the Keys/Pillow coefficient.
    constexpr double a = -0.5;
    x                  = std::abs(x);
    if (x < 1.0) { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0; }
    if (x < 2.0) { return (((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a); }
    return 0.0;
}

struct Coefficients {
    std::vector<int> starts;
    std::vector<int> offsets;
    std::vector<float> weights;
};

Coefficients coefficients(int input, int output) {
    Coefficients out;
    out.starts.resize(static_cast<std::size_t>(output));
    out.offsets.resize(static_cast<std::size_t>(output + 1));
    const double scale    = static_cast<double>(input) / output;
    const double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
    const double support  = 2.0 * (scale >= 1.0 ? scale : 1.0);
    for (int dst = 0; dst < output; ++dst) {
        const double center = scale * (dst + 0.5);
        const int begin     = std::max(static_cast<int>(center - support + 0.5), 0);
        const int size      = std::min(static_cast<int>(center + support + 0.5), input) - begin;
        out.starts[static_cast<std::size_t>(dst)]  = begin;
        out.offsets[static_cast<std::size_t>(dst)] = static_cast<int>(out.weights.size());
        double sum                                 = 0.0;
        for (int j = 0; j < size; ++j) {
            const double weight = cubic((j + begin - center + 0.5) * invscale);
            out.weights.push_back(static_cast<float>(weight));
            sum += weight;
        }
        if (sum == 0.0) { throw std::runtime_error("bicubic resize produced zero weights"); }
        const int first = out.offsets[static_cast<std::size_t>(dst)];
        for (std::size_t i = static_cast<std::size_t>(first); i < out.weights.size(); ++i) {
            out.weights[i] = static_cast<float>(out.weights[i] / sum);
        }
    }
    out.offsets[static_cast<std::size_t>(output)] = static_cast<int>(out.weights.size());
    return out;
}

void resize_bicubic(media::decode::Image& input, Size size, const PreparationControl& control) {
    if (input.width == size.w && input.height == size.h) { return; }
    const Coefficients horizontal = coefficients(input.width, size.w);
    const Coefficients vertical   = coefficients(input.height, size.h);
    std::vector<std::uint8_t> temp(static_cast<std::size_t>(input.height) * size.w * 3);
    for (int y = 0; y < input.height; ++y) {
        if (y % 16 == 0) { check_preparation_control(control); }
        for (int x = 0; x < size.w; ++x) {
            const int first        = horizontal.offsets[static_cast<std::size_t>(x)];
            const int last         = horizontal.offsets[static_cast<std::size_t>(x + 1)];
            const int source_begin = horizontal.starts[static_cast<std::size_t>(x)];
            std::array<float, 3> value{};
            const std::uint8_t* source =
                input.rgb.data() + (static_cast<std::size_t>(y) * input.width + source_begin) * 3;
            for (int i = first; i < last; ++i, source += 3) {
                const float weight = horizontal.weights[static_cast<std::size_t>(i)];
                value[0] += weight * source[0];
                value[1] += weight * source[1];
                value[2] += weight * source[2];
            }
            std::uint8_t* destination =
                temp.data() + (static_cast<std::size_t>(y) * size.w + x) * 3;
            for (int c = 0; c < 3; ++c) {
                destination[c] =
                    static_cast<std::uint8_t>(std::clamp(round_even(value[c]), 0, 255));
            }
        }
    }

    media::decode::Image out;
    out.width  = size.w;
    out.height = size.h;
    out.rgb.resize(static_cast<std::size_t>(size.h) * size.w * 3);
    for (int y = 0; y < size.h; ++y) {
        if (y % 16 == 0) { check_preparation_control(control); }
        const int first        = vertical.offsets[static_cast<std::size_t>(y)];
        const int last         = vertical.offsets[static_cast<std::size_t>(y + 1)];
        const int source_begin = vertical.starts[static_cast<std::size_t>(y)];
        for (int x = 0; x < size.w; ++x) {
            std::array<float, 3> value{};
            const std::uint8_t* source =
                temp.data() + (static_cast<std::size_t>(source_begin) * size.w + x) * 3;
            for (int i = first; i < last; ++i, source += static_cast<std::size_t>(size.w) * 3) {
                const float weight = vertical.weights[static_cast<std::size_t>(i)];
                value[0] += weight * source[0];
                value[1] += weight * source[1];
                value[2] += weight * source[2];
            }
            std::uint8_t* destination =
                out.rgb.data() + (static_cast<std::size_t>(y) * size.w + x) * 3;
            for (int c = 0; c < 3; ++c) {
                destination[c] =
                    static_cast<std::uint8_t>(std::clamp(round_even(value[c]), 0, 255));
            }
        }
    }
    input = std::move(out);
}

std::uint16_t to_bf16(float value) noexcept {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

const std::array<std::uint16_t, 256>& normalization_lut() {
    static const std::array<std::uint16_t, 256> values = [] {
        std::array<std::uint16_t, 256> out{};
        for (std::size_t value = 0; value < out.size(); ++value) {
            out[value] = to_bf16(static_cast<float>(value) / 127.5f - 1.0f);
        }
        return out;
    }();
    return values;
}

// DeepSeek-V4.1's values (x / 255 - 0.5) / 0.5 in FP32, as image_processor.py computes them.
const std::array<std::uint16_t, 256>& deepseek_lut() {
    static const std::array<std::uint16_t, 256> values = [] {
        std::array<std::uint16_t, 256> out{};
        for (std::size_t value = 0; value < out.size(); ++value) {
            out[value] = to_bf16((static_cast<float>(value) / 255.0f - 0.5f) / 0.5f);
        }
        return out;
    }();
    return values;
}

// Inkling 2's values (x - 127.5) / 127.5 in FP32, the release's fused rescale and normalization
// as llama.cpp computes it; its padding is the raw value -1.
const std::array<std::uint16_t, 256>& inkling_lut() {
    static const std::array<std::uint16_t, 256> values = [] {
        std::array<std::uint16_t, 256> out{};
        for (std::size_t value = 0; value < out.size(); ++value) {
            out[value] = to_bf16((static_cast<float>(value) - 127.5f) / 127.5f);
        }
        return out;
    }();
    return values;
}

std::uint16_t inkling_pad_value() { return to_bf16((-1.0f - 127.5f) / 127.5f); }

// PIL's bicubic filter (a = -0.5, support 2) and its fixed-point coefficients of one axis: per
// output sample its first input sample, their count and `ksize` coefficients (Resample.c).
constexpr int kPilPrecision = 22;

double pil_bicubic(double x) {
    constexpr double a = -0.5;
    if (x < 0.0) { x = -x; }
    if (x < 1.0) { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1; }
    if (x < 2.0) { return (((x - 5) * x + 8) * x - 4) * a; }
    return 0.0;
}

struct PilAxis {
    int ksize = 0;
    std::vector<int> first, count;
    std::vector<std::int32_t> weights; // [out][ksize]
};

PilAxis pil_axis(int in_size, int out_size) {
    const float in0 = 0.0f, in1 = static_cast<float>(in_size);
    const double scale       = static_cast<double>(in1 - in0) / out_size;
    const double filterscale = scale < 1.0 ? 1.0 : scale;
    const double support     = 2.0 * filterscale;
    PilAxis axis;
    axis.ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    axis.first.resize(static_cast<std::size_t>(out_size));
    axis.count.resize(static_cast<std::size_t>(out_size));
    axis.weights.assign(static_cast<std::size_t>(out_size) * axis.ksize, 0);
    std::vector<double> k(static_cast<std::size_t>(axis.ksize));
    const double inv = 1.0 / filterscale;
    for (int xx = 0; xx < out_size; ++xx) {
        const double center = in0 + (xx + 0.5) * scale;
        double ww           = 0.0;
        int xmin            = static_cast<int>(center - support + 0.5);
        if (xmin < 0) { xmin = 0; }
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > in_size) { xmax = in_size; }
        xmax -= xmin;
        for (int x = 0; x < xmax; ++x) {
            const double w = pil_bicubic((x + xmin - center + 0.5) * inv);
            k[static_cast<std::size_t>(x)] = w;
            ww += w;
        }
        if (ww != 0.0) {
            for (int x = 0; x < xmax; ++x) { k[static_cast<std::size_t>(x)] /= ww; }
        }
        for (int x = 0; x < axis.ksize; ++x) {
            const double w = x < xmax ? k[static_cast<std::size_t>(x)] : 0.0;
            axis.weights[static_cast<std::size_t>(xx) * axis.ksize + x] = static_cast<std::int32_t>(
                w < 0 ? -0.5 + w * (1 << kPilPrecision) : 0.5 + w * (1 << kPilPrecision));
        }
        axis.first[static_cast<std::size_t>(xx)] = xmin;
        axis.count[static_cast<std::size_t>(xx)] = xmax;
    }
    return axis;
}

std::uint8_t pil_clip(std::int32_t sum) {
    return static_cast<std::uint8_t>(std::clamp(sum >> kPilPrecision, 0, 255));
}

// One pass: along x (each row of `rows`) or along y (each column), from in_size samples to the
// axis' out_size, RGB.
std::vector<std::uint8_t> pil_pass(std::span<const std::uint8_t> rgb, int width, int height,
                                   int out_size, bool horizontal) {
    const int in_size = horizontal ? width : height;
    const PilAxis axis = pil_axis(in_size, out_size);
    const int out_w = horizontal ? out_size : width, out_h = horizontal ? height : out_size;
    std::vector<std::uint8_t> out(static_cast<std::size_t>(out_w) * out_h * 3);
    for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
            const int o = horizontal ? x : y;
            const std::int32_t* k = axis.weights.data() + static_cast<std::size_t>(o) * axis.ksize;
            const int first = axis.first[static_cast<std::size_t>(o)];
            const int count = axis.count[static_cast<std::size_t>(o)];
            for (int c = 0; c < 3; ++c) {
                std::int32_t sum = 1 << (kPilPrecision - 1);
                for (int i = 0; i < count; ++i) {
                    const std::size_t at =
                        horizontal ? (static_cast<std::size_t>(y) * width + first + i) * 3 + c
                                   : (static_cast<std::size_t>(first + i) * width + x) * 3 + c;
                    sum += static_cast<std::int32_t>(rgb[at]) * k[i];
                }
                out[(static_cast<std::size_t>(y) * out_w + x) * 3 + c] = pil_clip(sum);
            }
        }
    }
    return out;
}

void append_patch(const std::vector<const media::decode::Image*>& frames, int grid_y, int grid_x,
                  std::span<std::uint16_t> out, std::size_t& cursor) {
    if (cursor > out.size() || out.size() - cursor < kPatchFeatures) {
        throw std::logic_error("Vision patch writer exceeded its allocation");
    }
    const auto& lut            = normalization_lut();
    std::uint16_t* destination = out.data() + cursor;
    std::size_t local          = 0;
    for (int channel = 0; channel < 3; ++channel) {
        for (int temporal = 0; temporal < kTemporal; ++temporal) {
            const media::decode::Image& frame = *frames[static_cast<std::size_t>(temporal)];
            for (int y = 0; y < kPatch; ++y) {
                const std::uint8_t* source =
                    frame.rgb.data() +
                    (static_cast<std::size_t>(grid_y * kPatch + y) * frame.width +
                     grid_x * kPatch) *
                        3 +
                    channel;
                for (int x = 0; x < kPatch; ++x) {
                    destination[local++] = lut[source[static_cast<std::size_t>(x) * 3]];
                }
            }
        }
    }
    cursor += local;
}

void add_budget(PreprocessStats& stats, const VisionItem& item, const ProcessorOptions& options);
void enforce_media_item_resource_limits(const PreprocessStats& stats);
void enforce_media_resource_limits(const PreprocessStats& stats, const ProcessorOptions& options);

// Miss builders run concurrently. Claim their aggregate extent before allocating the retained
// patch payload so an invalid prompt cannot fill the live-byte account and leave another worker
// waiting for memory that this same request will never release.
class ConcurrentMediaBudget {
public:
    explicit ConcurrentMediaBudget(const ProcessorOptions& options) : options_(options) {}

    void claim(const VisionItem& item) {
        std::lock_guard lock(mutex_);
        add_budget(stats_, item, options_);
        enforce_media_resource_limits(stats_, options_);
    }

private:
    const ProcessorOptions& options_;
    std::mutex mutex_;
    PreprocessStats stats_;
};

Prepared prepare_image(std::span<const std::uint8_t> bytes, const ProcessorOptions& options,
                       const media::decode::Policy& policy, MediaPreprocessCache& cache,
                       ConcurrentMediaBudget& request_budget, const PreparationControl& control) {
    media::decode::Image image = media::decode::decode_image(bytes, policy);
    const Size size = smart_resize_image(image.height, image.width, options.image_min_pixels,
                                         options.image_max_pixels);
    const int gh    = size.h / kPatch;
    const int gw    = size.w / kPatch;
    Prepared out;
    out.item.modality = Modality::Image;
    out.item.grid     = {1, gh, gw};
    PreprocessStats item_stats;
    add_budget(item_stats, out.item, options);
    enforce_media_item_resource_limits(item_stats);
    request_budget.claim(out.item);
    const std::size_t elements = static_cast<std::size_t>(gh) * gw * kPatchFeatures;
    out.payload                = cache.allocate_payload(elements, control);
    resize_bicubic(image, size, control);
    std::size_t cursor = 0;
    const std::vector<const media::decode::Image*> frames{&image, &image};
    for (int block_y = 0; block_y < gh / kMerge; ++block_y) {
        check_preparation_control(control);
        for (int block_x = 0; block_x < gw / kMerge; ++block_x) {
            for (int merge_y = 0; merge_y < kMerge; ++merge_y) {
                for (int merge_x = 0; merge_x < kMerge; ++merge_x) {
                    append_patch(frames, block_y * kMerge + merge_y, block_x * kMerge + merge_x,
                                 out.payload->mutable_span(), cursor);
                }
            }
        }
    }
    if (cursor != elements) { throw std::logic_error("Vision patch writer left a short payload"); }
    return out;
}

std::vector<double> video_timestamps(std::span<const int> indices, int temporal_groups,
                                     double fps) {
    std::vector<int> padded(indices.begin(), indices.end());
    if (padded.size() % kTemporal != 0) { padded.push_back(padded.back()); }
    if (temporal_groups < 0 ||
        static_cast<std::size_t>(temporal_groups) > padded.size() / kTemporal) {
        throw std::logic_error("sampled video indices do not cover the temporal grid");
    }
    std::vector<double> timestamps;
    timestamps.reserve(static_cast<std::size_t>(temporal_groups));
    for (int t = 0; t < temporal_groups; ++t) {
        timestamps.push_back(static_cast<double>(padded[2 * t] + padded[2 * t + 1]) / (2.0 * fps));
    }
    return timestamps;
}

Prepared prepare_video(std::span<const std::uint8_t> bytes, const ProcessorOptions& options,
                       const media::decode::Policy& policy, MediaPreprocessCache& cache,
                       ConcurrentMediaBudget& request_budget, const PreparationControl& control) {
    media::decode::Video video = media::decode::decode_video(
        bytes, policy, options.video_fps, options.video_min_frames, options.video_max_frames);
    const Size size =
        smart_resize_video(static_cast<int>(video.frames.size()), video.height, video.width,
                           options.video_min_pixels, options.video_max_pixels);
    const bool pad_temporal = video.frames.size() % kTemporal != 0;
    const int gt            = static_cast<int>((video.frames.size() + kTemporal - 1) / kTemporal);
    const int gh            = size.h / kPatch;
    const int gw            = size.w / kPatch;
    Prepared out;
    out.item.modality = Modality::Video;
    out.item.grid     = {gt, gh, gw};
    PreprocessStats item_stats;
    add_budget(item_stats, out.item, options);
    enforce_media_item_resource_limits(item_stats);
    request_budget.claim(out.item);
    const std::size_t elements = static_cast<std::size_t>(gt) * gh * gw * kPatchFeatures;
    out.payload                = cache.allocate_payload(elements, control);
    for (media::decode::Image& frame : video.frames) { resize_bicubic(frame, size, control); }
    if (pad_temporal) { video.frames.push_back(video.frames.back()); }
    out.item.timestamps = video_timestamps(video.indices, gt, video.fps);
    std::size_t cursor  = 0;
    for (int t = 0; t < gt; ++t) {
        check_preparation_control(control);
        const std::vector<const media::decode::Image*> frames{
            &video.frames[static_cast<std::size_t>(2 * t)],
            &video.frames[static_cast<std::size_t>(2 * t + 1)]};
        for (int block_y = 0; block_y < gh / kMerge; ++block_y) {
            for (int block_x = 0; block_x < gw / kMerge; ++block_x) {
                for (int merge_y = 0; merge_y < kMerge; ++merge_y) {
                    for (int merge_x = 0; merge_x < kMerge; ++merge_x) {
                        append_patch(frames, block_y * kMerge + merge_y, block_x * kMerge + merge_x,
                                     out.payload->mutable_span(), cursor);
                    }
                }
            }
        }
    }
    if (cursor != elements) { throw std::logic_error("Vision patch writer left a short payload"); }
    return out;
}

// ---- Gemma 4 -----------------------------------------------------------------------------------

// transformers' get_aspect_ratio_preserving_size: the largest size, aspect kept, whose sides are
// multiples of pooling * patch and whose patches fit max_patches (a small image grows to it).
Size gemma_size(int height, int width, std::uint64_t max_patches, int patch, int pooling) {
    if (height <= 0 || width <= 0 || max_patches == 0) {
        throw std::invalid_argument("invalid Gemma media resize configuration");
    }
    const double target_px = static_cast<double>(max_patches) * patch * patch;
    const double factor    = std::sqrt(target_px / (static_cast<double>(height) * width));
    const int side         = pooling * patch;
    int h = static_cast<int>(std::floor(factor * height / side)) * side;
    int w = static_cast<int>(std::floor(factor * width / side)) * side;
    if (h == 0 && w == 0) {
        throw ProcessorError(ProcessorErrorKind::InvalidMedia,
                             "Gemma media resizes to a 0 x 0 picture");
    }
    const int max_side = static_cast<int>(max_patches / std::uint64_t(pooling * pooling)) * side;
    if (h == 0) {
        h = side;
        w = std::min(static_cast<int>(std::floor(static_cast<double>(width) / height)) * side,
                     max_side);
    } else if (w == 0) {
        w = side;
        h = std::min(static_cast<int>(std::floor(static_cast<double>(height) / width)) * side,
                     max_side);
    }
    if (static_cast<double>(h) * w > target_px) {
        throw ProcessorError(ProcessorErrorKind::InvalidMedia,
                             "Gemma media resize exceeds its patch budget");
    }
    return {h, w};
}

// A DeepSeek-V4.1 image's span budget: the bound, a quarter of it at low detail.
std::uint32_t deepseek_part_budget(const ChatPart& part, const DeepSeekMediaOptions& d) {
    return part.media.image_detail == ImageDetail::Low ? std::max(d.max_tokens / 4, 5U)
                                                       : d.max_tokens;
}

// An Inkling 2 image's cell budget: the bound, a quarter of it at low detail.
std::uint32_t inkling_part_budget(const ChatPart& part, const InklingMediaOptions& k) {
    return part.media.image_detail == ImageDetail::Low ? std::max(k.max_tokens / 4, 1U)
                                                       : k.max_tokens;
}

int ceil_div(int a, int b) { return (a + b - 1) / b; }

// The prompt positions of one media run: Gemma's pooled tokens, DeepSeek's span, Inkling's cells,
// Qwen's merged tokens.
std::size_t media_run_tokens(const VisionItem& item, const ProcessorOptions& options) {
    if (options.inkling) {
        const int m = options.inkling->merge;
        return static_cast<std::size_t>(item.grid.h / m) * (item.grid.w / m);
    }
    if (options.deepseek) {
        const int m = options.deepseek->merge;
        return DeepSeekImagePlan{0, 0, ceil_div(item.grid.h, m), ceil_div(item.grid.w, m)}
            .span_tokens();
    }
    const int pooling = options.gemma ? options.gemma->pooling : kMerge;
    return static_cast<std::size_t>(item.grid.h / pooling) * (item.grid.w / pooling);
}

MediaPads media_pads(const ProcessorOptions& options) {
    return options.inkling    ? kInklingMediaPads
           : options.deepseek ? kDeepSeekMediaPads
           : options.gemma    ? kGemmaMediaPads
                              : kQwenMediaPads;
}

void refuse_image_only_video(std::span<ChatPart* const> parts, const ProcessorOptions& options) {
    if (!options.deepseek && !options.inkling) { return; }
    for (const ChatPart* part : parts) {
        if (part->kind != ChatPartKind::Image) {
            throw ProcessorError(ProcessorErrorKind::InvalidMedia,
                                 options.inkling ? "Inkling 2 reads images, not video"
                                                 : "DeepSeek-V4.1 reads images, not video");
        }
    }
}

// Inkling 2's patches of `rgb` [height][width][3] under `plan` into `destination`: resized first
// when the plan says so, then per patch of the row-major grid its channel, row and column values,
// those past the image the padding.
void write_inkling_patches(std::span<const std::uint8_t> rgb, int width, int height,
                           const InklingImagePlan& plan, const InklingMediaOptions& k,
                           std::uint16_t* destination, const PreparationControl* control) {
    std::vector<std::uint8_t> resized;
    if (plan.width != width || plan.height != height) {
        resized = pil_bicubic_resize(rgb, width, height, plan.width, plan.height);
        rgb     = resized;
    }
    const auto& lut           = inkling_lut();
    const std::uint16_t pad   = inkling_pad_value();
    const int p               = k.patch;
    for (int gy = 0; gy < plan.grid_h; ++gy) {
        if (control != nullptr) { check_preparation_control(*control); }
        for (int gx = 0; gx < plan.grid_w; ++gx) {
            for (int c = 0; c < 3; ++c) {
                for (int py = 0; py < p; ++py) {
                    const int y = gy * p + py;
                    for (int px = 0; px < p; ++px) {
                        const int x    = gx * p + px;
                        *destination++ = y < plan.height && x < plan.width
                                             ? lut[rgb[(std::size_t(y) * plan.width + x) * 3 + c]]
                                             : pad;
                    }
                }
            }
        }
    }
}

Prepared prepare_inkling_image(std::span<const std::uint8_t> bytes, std::uint32_t budget,
                               const ProcessorOptions& options, const media::decode::Policy& policy,
                               MediaPreprocessCache& cache, ConcurrentMediaBudget& request_budget,
                               const PreparationControl& control) {
    const InklingMediaOptions& k     = *options.inkling;
    const media::decode::Image image = media::decode::decode_image(bytes, policy);
    const InklingImagePlan plan      = inkling_image_plan(image.width, image.height, k, budget);
    Prepared out;
    out.item.modality = Modality::Image;
    out.item.grid     = {1, plan.grid_h, plan.grid_w};
    PreprocessStats item_stats;
    add_budget(item_stats, out.item, options);
    enforce_media_item_resource_limits(item_stats);
    request_budget.claim(out.item);
    check_preparation_control(control);
    const std::size_t elements =
        static_cast<std::size_t>(plan.grid_h) * plan.grid_w * 3 * k.patch * k.patch;
    out.payload = cache.allocate_payload(elements, control);
    write_inkling_patches(image.rgb, image.width, image.height, plan, k,
                          out.payload->mutable_span().data(), &control);
    return out;
}

VisionItem inspect_inkling_item(const ChatPart& part, const ProcessorOptions& options,
                                const media::decode::Policy& policy) {
    const InklingMediaOptions& k         = *options.inkling;
    const media::decode::ImageInfo image = media::decode::inspect_image(part.media.bytes, policy);
    const InklingImagePlan plan =
        inkling_image_plan(image.width, image.height, k, inkling_part_budget(part, k));
    VisionItem item;
    item.modality = Modality::Image;
    item.grid     = {1, plan.grid_h, plan.grid_w};
    return item;
}

Prepared prepare_deepseek_image(std::span<const std::uint8_t> bytes, std::uint32_t budget,
                                const ProcessorOptions& options, const media::decode::Policy& policy,
                                MediaPreprocessCache& cache, ConcurrentMediaBudget& request_budget,
                                const PreparationControl& control) {
    const DeepSeekMediaOptions& d = *options.deepseek;
    const media::decode::Image image = media::decode::decode_image(bytes, policy);
    const DeepSeekImagePlan plan = deepseek_image_plan(image.width, image.height, d, budget);
    const int gh = plan.height / d.patch, gw = plan.width / d.patch;
    Prepared out;
    out.item.modality = Modality::Image;
    out.item.grid     = {1, gh, gw};
    PreprocessStats item_stats;
    add_budget(item_stats, out.item, options);
    enforce_media_item_resource_limits(item_stats);
    request_budget.claim(out.item);
    check_preparation_control(control);
    const std::vector<std::uint8_t> rgb =
        deepseek_pad(image.rgb, image.width, image.height, plan.width, plan.height);
    const std::size_t features = std::size_t(3) * d.patch * d.patch;
    const std::size_t elements = static_cast<std::size_t>(gh) * gw * features;
    out.payload                = cache.allocate_payload(elements, control);
    const auto& lut            = deepseek_lut();
    std::uint16_t* destination = out.payload->mutable_span().data();
    for (int y = 0; y < gh; ++y) {
        check_preparation_control(control);
        for (int x = 0; x < gw; ++x) {
            for (int c = 0; c < 3; ++c) {
                for (int py = 0; py < d.patch; ++py) {
                    const std::uint8_t* source =
                        rgb.data() +
                        (static_cast<std::size_t>(y * d.patch + py) * plan.width + x * d.patch) * 3 + c;
                    for (int px = 0; px < d.patch; ++px) { *destination++ = lut[source[px * 3]]; }
                }
            }
        }
    }
    return out;
}

VisionItem inspect_deepseek_item(const ChatPart& part, const ProcessorOptions& options,
                                 const media::decode::Policy& policy) {
    const DeepSeekMediaOptions& d        = *options.deepseek;
    const media::decode::ImageInfo image = media::decode::inspect_image(part.media.bytes, policy);
    const DeepSeekImagePlan plan =
        deepseek_image_plan(image.width, image.height, d, deepseek_part_budget(part, d));
    VisionItem item;
    item.modality = Modality::Image;
    item.grid     = {1, plan.height / d.patch, plan.width / d.patch};
    return item;
}

// The soft-token budget of one media part: a video frame's, or an image's by its detail (low the
// smallest, high the server's bound, auto the release's default), never above the bound.
std::uint32_t gemma_part_budget(const ChatPart& part, const GemmaMediaOptions& g) {
    std::uint64_t wanted = g.image_tokens;
    if (part.kind == ChatPartKind::Video) {
        wanted = g.video_tokens;
    } else if (part.media.image_detail == ImageDetail::Low) {
        wanted = kGemmaSoftTokenBudgets.front();
    } else if (part.media.image_detail == ImageDetail::High) {
        wanted = g.max_tokens;
    }
    return gemma_budget_at_most(std::min<std::uint64_t>(wanted, g.max_tokens));
}

std::uint64_t gemma_max_patches(std::uint32_t budget, const GemmaMediaOptions& g) {
    return std::uint64_t(budget) * std::uint64_t(g.pooling) * std::uint64_t(g.pooling);
}

// One patch (channel, y, x) of `frame` at patch (grid_y, grid_x), as 2 p / 255 - 1 in BF16.
void append_gemma_patch(const media::decode::Image& frame, int patch, int grid_y, int grid_x,
                        std::span<std::uint16_t> out, std::size_t& cursor) {
    const std::size_t features = std::size_t(3) * patch * patch;
    if (cursor > out.size() || out.size() - cursor < features) {
        throw std::logic_error("Vision patch writer exceeded its allocation");
    }
    const auto& lut            = normalization_lut();
    std::uint16_t* destination = out.data() + cursor;
    std::size_t local          = 0;
    for (int channel = 0; channel < 3; ++channel) {
        for (int y = 0; y < patch; ++y) {
            const std::uint8_t* source =
                frame.rgb.data() +
                (static_cast<std::size_t>(grid_y * patch + y) * frame.width + grid_x * patch) * 3 +
                channel;
            for (int x = 0; x < patch; ++x) {
                destination[local++] = lut[source[static_cast<std::size_t>(x) * 3]];
            }
        }
    }
    cursor += local;
}

// One frame's patches, row by row of the patch grid.
void append_gemma_frame(const media::decode::Image& frame, int patch, int gh, int gw,
                        std::span<std::uint16_t> out, std::size_t& cursor,
                        const PreparationControl& control) {
    for (int y = 0; y < gh; ++y) {
        check_preparation_control(control);
        for (int x = 0; x < gw; ++x) { append_gemma_patch(frame, patch, y, x, out, cursor); }
    }
}

Prepared prepare_gemma_image(std::span<const std::uint8_t> bytes, std::uint32_t budget,
                             const ProcessorOptions& options, const media::decode::Policy& policy,
                             MediaPreprocessCache& cache, ConcurrentMediaBudget& request_budget,
                             const PreparationControl& control) {
    const GemmaMediaOptions& g = *options.gemma;
    media::decode::Image image = media::decode::decode_image(bytes, policy);
    const Size size =
        gemma_size(image.height, image.width, gemma_max_patches(budget, g), g.patch, g.pooling);
    const int gh = size.h / g.patch;
    const int gw = size.w / g.patch;
    Prepared out;
    out.item.modality = Modality::Image;
    out.item.grid     = {1, gh, gw};
    PreprocessStats item_stats;
    add_budget(item_stats, out.item, options);
    enforce_media_item_resource_limits(item_stats);
    request_budget.claim(out.item);
    const std::size_t elements = static_cast<std::size_t>(gh) * gw * 3 * g.patch * g.patch;
    out.payload                = cache.allocate_payload(elements, control);
    resize_bicubic(image, size, control);
    std::size_t cursor = 0;
    append_gemma_frame(image, g.patch, gh, gw, out.payload->mutable_span(), cursor, control);
    if (cursor != elements) { throw std::logic_error("Vision patch writer left a short payload"); }
    return out;
}

std::vector<double> gemma_timestamps(std::span<const int> indices, double fps) {
    std::vector<double> out;
    out.reserve(indices.size());
    for (const int index : indices) { out.push_back(static_cast<double>(index) / fps); }
    return out;
}

Prepared prepare_gemma_video(std::span<const std::uint8_t> bytes, std::uint32_t budget,
                             const ProcessorOptions& options, const media::decode::Policy& policy,
                             MediaPreprocessCache& cache, ConcurrentMediaBudget& request_budget,
                             const PreparationControl& control) {
    const GemmaMediaOptions& g = *options.gemma;
    media::decode::Video video = media::decode::decode_video(
        bytes, policy, 1.0, 1, g.frames, media::decode::FrameSpacing::Floor);
    const Size size =
        gemma_size(video.height, video.width, gemma_max_patches(budget, g), g.patch, g.pooling);
    const int gt = static_cast<int>(video.frames.size());
    const int gh = size.h / g.patch;
    const int gw = size.w / g.patch;
    Prepared out;
    out.item.modality = Modality::Video;
    out.item.grid     = {gt, gh, gw};
    PreprocessStats item_stats;
    add_budget(item_stats, out.item, options);
    enforce_media_item_resource_limits(item_stats);
    request_budget.claim(out.item);
    const std::size_t elements =
        static_cast<std::size_t>(gt) * gh * gw * 3 * g.patch * g.patch;
    out.payload = cache.allocate_payload(elements, control);
    std::size_t cursor = 0;
    for (media::decode::Image& frame : video.frames) {
        resize_bicubic(frame, size, control);
        append_gemma_frame(frame, g.patch, gh, gw, out.payload->mutable_span(), cursor, control);
    }
    if (cursor != elements) { throw std::logic_error("Vision patch writer left a short payload"); }
    out.item.timestamps = gemma_timestamps(video.indices, video.fps);
    return out;
}

VisionItem inspect_gemma_item(const ChatPart& part, const ProcessorOptions& options,
                              const media::decode::Policy& policy) {
    const GemmaMediaOptions& g  = *options.gemma;
    const std::uint32_t budget  = gemma_part_budget(part, g);
    const std::uint64_t patches = gemma_max_patches(budget, g);
    VisionItem item;
    if (part.kind == ChatPartKind::Image) {
        const media::decode::ImageInfo image = media::decode::inspect_image(part.media.bytes, policy);
        const Size size = gemma_size(image.height, image.width, patches, g.patch, g.pooling);
        item.modality   = Modality::Image;
        item.grid       = {1, size.h / g.patch, size.w / g.patch};
        return item;
    }
    const media::decode::VideoInfo video = media::decode::inspect_video(
        part.media.bytes, policy, 1.0, 1, g.frames, media::decode::FrameSpacing::Floor);
    const Size size = gemma_size(video.height, video.width, patches, g.patch, g.pooling);
    item.modality   = Modality::Video;
    item.grid       = {video.sampled_frames, size.h / g.patch, size.w / g.patch};
    item.timestamps = gemma_timestamps(video.indices, video.fps);
    return item;
}

// mm:ss, as the release's processor prints a frame's time.
std::string gemma_time(double seconds) {
    const auto whole = static_cast<long long>(std::floor(seconds));
    char text[32];
    std::snprintf(text, sizeof(text), "%02lld:%02lld", whole / 60, whole % 60);
    return text;
}

std::vector<ChatPart*> media_parts(std::vector<ChatMessage>& messages) {
    std::vector<ChatPart*> out;
    for (ChatMessage& message : messages) {
        if (message.role == ChatRole::System || message.role == ChatRole::Developer) {
            if (message.has_media()) {
                throw std::invalid_argument(
                    "system and developer messages cannot contain images or videos");
            }
            continue;
        }
        if (message.role != ChatRole::User && message.role != ChatRole::Assistant &&
            message.role != ChatRole::Tool) {
            throw std::invalid_argument("unsupported chat role value");
        }
        for (ChatPart& part : message.parts) {
            if (part.kind != ChatPartKind::Text) { out.push_back(&part); }
        }
    }
    return out;
}

std::size_t validate_media_inputs(std::span<ChatPart* const> parts,
                                  const ProcessorOptions& options) {
    const std::uint64_t maximum_items_from_extents =
        std::min(options.max_raw_patches / kMinimumRawPatchesPerItem, options.max_vision_tokens);
    if (std::cmp_greater(parts.size(), maximum_items_from_extents)) {
        throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                             "minimum Vision grids exceed processor extent budget");
    }
    std::size_t remaining = options.max_encoded_media_bytes;
    for (const ChatPart* part : parts) {
        if (part->media.bytes.size() > remaining) {
            throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                                 "request media bytes exceed processor budget");
        }
        remaining -= part->media.bytes.size();
    }
    return options.max_encoded_media_bytes - remaining;
}

VisionItem inspect_image_item(std::span<const std::uint8_t> bytes, const ProcessorOptions& options,
                              const media::decode::Policy& policy) {
    const media::decode::ImageInfo image = media::decode::inspect_image(bytes, policy);
    const Size size = smart_resize_image(image.height, image.width, options.image_min_pixels,
                                         options.image_max_pixels);
    VisionItem item;
    item.modality = Modality::Image;
    item.grid     = {1, size.h / kPatch, size.w / kPatch};
    return item;
}

// A low-detail image's bound: the area of a 512 x 512 picture.
constexpr std::uint64_t kLowDetailMergedTokens = 256;

// The options one media part is counted and prepared with: a low-detail image is bounded further.
ProcessorOptions part_options(const ChatPart& part, const ProcessorOptions& options) {
    ProcessorOptions out = options;
    if (part.kind == ChatPartKind::Image && part.media.image_detail == ImageDetail::Low) {
        bound_merged_tokens(out, kLowDetailMergedTokens);
    }
    return out;
}

// The identity of a part's prepared media: its bytes' digest, kept apart for a low-detail image,
// whose Vision tokens differ from the same bytes seen at full detail.
Sha256Digest part_digest(const ChatPart& part, const Sha256Digest& bytes_digest,
                         const ProcessorOptions& options) {
    if (options.inkling) {
        // Inkling 2's cells depend on the budget the part is seen at.
        std::string keyed = "ninfer inkling budget " +
                            std::to_string(inkling_part_budget(part, *options.inkling)) + "\n";
        keyed.append(reinterpret_cast<const char*>(bytes_digest.data()), bytes_digest.size());
        return sha256(std::string_view(keyed));
    }
    if (options.deepseek) {
        // DeepSeek-V4.1's span depends on the budget the part is seen at.
        std::string keyed = "ninfer deepseek budget " +
                            std::to_string(deepseek_part_budget(part, *options.deepseek)) + "\n";
        keyed.append(reinterpret_cast<const char*>(bytes_digest.data()), bytes_digest.size());
        return sha256(std::string_view(keyed));
    }
    if (options.gemma) {
        // Gemma 4's soft tokens depend on the budget the part is seen at.
        std::string keyed =
            "ninfer gemma budget " + std::to_string(gemma_part_budget(part, *options.gemma)) + "\n";
        keyed.append(reinterpret_cast<const char*>(bytes_digest.data()), bytes_digest.size());
        return sha256(std::string_view(keyed));
    }
    if (part.kind != ChatPartKind::Image || part.media.image_detail != ImageDetail::Low) {
        return bytes_digest;
    }
    std::string keyed = "ninfer image detail low\n";
    keyed.append(reinterpret_cast<const char*>(bytes_digest.data()), bytes_digest.size());
    return sha256(std::string_view(keyed));
}

void enforce_image_resize_policy(const ChatPart& part, const ProcessorOptions& options,
                                 const media::decode::Policy& policy) {
    if (part.kind != ChatPartKind::Image ||
        part.media.image_resize_policy != ImageResizePolicy::RejectOversized) {
        return;
    }
    const media::decode::ImageInfo image = media::decode::inspect_image(part.media.bytes, policy);
    if (options.inkling) {
        const InklingMediaOptions& k = *options.inkling;
        const InklingImagePlan plan  = inkling_image_plan(
            image.width, image.height, k, std::numeric_limits<std::uint32_t>::max());
        if (plan.cells(k.merge) > inkling_part_budget(part, k)) {
            throw ProcessorError(ProcessorErrorKind::InvalidMedia,
                                 "image exceeds its Vision budget and oversized_image is set to "
                                 "'error'");
        }
        return;
    }
    if (options.deepseek) {
        const DeepSeekMediaOptions& d = *options.deepseek;
        const std::uint32_t budget    = deepseek_part_budget(part, d);
        if (deepseek_image_plan(image.width, image.height, d,
                                std::numeric_limits<std::uint32_t>::max())
                .span_tokens() > budget) {
            throw ProcessorError(ProcessorErrorKind::InvalidMedia,
                                 "image exceeds its Vision budget and oversized_image is set to "
                                 "'error'");
        }
        return;
    }
    if (options.gemma) {
        const GemmaMediaOptions& g = *options.gemma;
        const double budget_px     = static_cast<double>(gemma_max_patches(
                                     gemma_part_budget(part, g), g)) * g.patch * g.patch;
        if (static_cast<double>(image.height) * image.width > budget_px) {
            throw ProcessorError(ProcessorErrorKind::InvalidMedia,
                                 "image exceeds its Vision budget and oversized_image is set to "
                                 "'error'");
        }
        return;
    }
    const int aligned_h = round_even(static_cast<double>(image.height) / kFactor) * kFactor;
    const int aligned_w = round_even(static_cast<double>(image.width) / kFactor) * kFactor;
    const std::uint64_t area =
        checked_mul(std::max(aligned_h, 0), std::max(aligned_w, 0), "image area");
    if (area > options.image_max_pixels) {
        throw ProcessorError(ProcessorErrorKind::InvalidMedia,
                             "image exceeds the native Vision geometry and oversized_image is "
                             "set to 'error'");
    }
}

VisionItem inspect_video_item(std::span<const std::uint8_t> bytes, const ProcessorOptions& options,
                              const media::decode::Policy& policy) {
    const media::decode::VideoInfo video = media::decode::inspect_video(
        bytes, policy, options.video_fps, options.video_min_frames, options.video_max_frames);
    const Size size = smart_resize_video(video.sampled_frames, video.height, video.width,
                                         options.video_min_pixels, options.video_max_pixels);
    const int gt    = (video.sampled_frames + kTemporal - 1) / kTemporal;
    VisionItem item;
    item.modality   = Modality::Video;
    item.grid       = {gt, size.h / kPatch, size.w / kPatch};
    item.timestamps = video_timestamps(video.indices, gt, video.fps);
    return item;
}

void append_repeated(std::string& out, std::string_view value, std::uint64_t count) {
    if (count > (std::numeric_limits<std::size_t>::max() - out.size()) / value.size()) {
        throw std::invalid_argument("vision placeholder expansion is too large");
    }
    for (std::uint64_t i = 0; i < count; ++i) { out += value; }
}

void append_media_token_run(std::string& out, std::string_view pad, std::uint64_t count,
                            Modality modality, std::size_t item_index, std::size_t frame_index,
                            std::vector<MediaTokenRunByteSpec>& runs) {
    const std::size_t begin = out.size();
    append_repeated(out, pad, count);
    runs.push_back(MediaTokenRunByteSpec{
        .bytes       = text::ByteSpan{begin, out.size()},
        .modality    = modality,
        .item_index  = item_index,
        .frame_index = frame_index,
    });
}

void append_placeholder(std::string& out, const VisionItem& item, std::size_t item_index,
                        std::vector<MediaTokenRunByteSpec>& runs) {
    const std::uint64_t frame_tokens =
        static_cast<std::uint64_t>(item.grid.h / kMerge) * (item.grid.w / kMerge);
    if (item.modality == Modality::Image) {
        append_media_token_run(out, kImagePad, frame_tokens, Modality::Image, item_index, 0, runs);
        return;
    }
    if (item.timestamps.size() != static_cast<std::size_t>(item.grid.t)) {
        throw std::logic_error("video timestamp count does not match grid");
    }
    for (std::size_t frame = 0; frame < item.timestamps.size(); ++frame) {
        std::ostringstream time;
        time << '<' << std::fixed << std::setprecision(1) << item.timestamps[frame] << " seconds>";
        out += time.str();
        out += kVisionStart;
        append_media_token_run(out, kVideoPad, frame_tokens, Modality::Video, item_index, frame,
                               runs);
        out += kVisionEnd;
    }
}

void append_gemma_placeholder(std::string& out, const VisionItem& item, std::size_t item_index,
                              int pooling, std::vector<MediaTokenRunByteSpec>& runs) {
    const std::uint64_t frame_tokens =
        static_cast<std::uint64_t>(item.grid.h / pooling) * (item.grid.w / pooling);
    if (item.modality == Modality::Image) {
        out += kGemmaBoi;
        append_media_token_run(out, kGemmaMediaPads.image, frame_tokens, Modality::Image,
                               item_index, 0, runs);
        out += kGemmaEoi;
        return;
    }
    if (item.timestamps.size() != static_cast<std::size_t>(item.grid.t)) {
        throw std::logic_error("video timestamp count does not match grid");
    }
    for (std::size_t frame = 0; frame < item.timestamps.size(); ++frame) {
        if (frame != 0) { out += ' '; }
        out += gemma_time(item.timestamps[frame]);
        out += ' ';
        out += kGemmaBoi;
        append_media_token_run(out, kGemmaMediaPads.video, frame_tokens, Modality::Video,
                               item_index, frame, runs);
        out += kGemmaEoi;
    }
}

void append_deepseek_placeholder(std::string& out, const VisionItem& item, std::size_t item_index,
                                 const ProcessorOptions& options,
                                 std::vector<MediaTokenRunByteSpec>& runs) {
    append_media_token_run(out, kDeepSeekMediaPads.image, media_run_tokens(item, options),
                           Modality::Image, item_index, 0, runs);
}

struct PlaceholderExpansion {
    std::size_t source_begin = 0;
    std::size_t source_end   = 0;
    std::size_t rendered_end = 0;
};

std::size_t map_expanded_boundary(std::size_t boundary, std::size_t source_size,
                                  std::span<const PlaceholderExpansion> expansions,
                                  std::string_view kind) {
    if (boundary > source_size) {
        throw std::logic_error(std::string(kind) + " byte offset exceeds rendered chat");
    }
    const auto first_incomplete =
        std::upper_bound(expansions.begin(), expansions.end(), boundary,
                         [](std::size_t value, const PlaceholderExpansion& expansion) {
                             return value < expansion.source_end;
                         });
    if (first_incomplete != expansions.end() && first_incomplete->source_begin < boundary) {
        throw std::logic_error(std::string(kind) + " intersects a media placeholder");
    }
    if (first_incomplete == expansions.begin()) { return boundary; }
    const PlaceholderExpansion& completed = *std::prev(first_incomplete);
    return completed.rendered_end + boundary - completed.source_end;
}

RenderedChat expand_placeholders(RenderedChat rendered, const std::vector<VisionItem>& items,
                                 const ProcessorOptions& options) {
    const MediaPads pads = media_pads(options);
    if (!rendered.media_token_runs.empty()) {
        throw std::logic_error("rendered chat media placeholders were already expanded");
    }
    if (rendered.media_placeholders.size() != items.size()) {
        throw std::invalid_argument("chat media count does not match rendered placeholders");
    }
    std::string source = std::move(rendered.text);
    std::string expanded;
    expanded.reserve(source.size());
    std::vector<PlaceholderExpansion> expansions;
    expansions.reserve(items.size());
    std::size_t source_cursor = 0;
    for (std::size_t index = 0; index < items.size(); ++index) {
        const VisionItem& item                     = items[index];
        const MediaPlaceholderByteSpec placeholder = rendered.media_placeholders[index];
        if (placeholder.item_index != index || placeholder.modality != item.modality) {
            throw std::invalid_argument("chat media order does not match rendered placeholders");
        }
        const std::string_view needle = item.modality == Modality::Image ? pads.image : pads.video;
        if (placeholder.bytes.begin < source_cursor || placeholder.bytes.end > source.size() ||
            placeholder.bytes.end - placeholder.bytes.begin != needle.size() ||
            source.compare(placeholder.bytes.begin, needle.size(), needle) != 0) {
            throw std::logic_error("rendered media placeholder metadata does not match its text");
        }
        expanded.append(source, source_cursor, placeholder.bytes.begin - source_cursor);
        if (options.inkling) {
            append_media_token_run(expanded, kInklingMediaPads.image,
                                   media_run_tokens(item, options), Modality::Image, index, 0,
                                   rendered.media_token_runs);
        } else if (options.deepseek) {
            append_deepseek_placeholder(expanded, item, index, options, rendered.media_token_runs);
        } else if (options.gemma) {
            append_gemma_placeholder(expanded, item, index, options.gemma->pooling,
                                     rendered.media_token_runs);
        } else {
            append_placeholder(expanded, item, index, rendered.media_token_runs);
        }
        const std::size_t source_end = placeholder.bytes.end;
        expansions.push_back(PlaceholderExpansion{.source_begin = placeholder.bytes.begin,
                                                  .source_end   = source_end,
                                                  .rendered_end = expanded.size()});
        source_cursor = source_end;
    }
    expanded.append(source, source_cursor, source.size() - source_cursor);

    const auto map_boundary = [&](std::size_t boundary, std::string_view kind) {
        return map_expanded_boundary(boundary, source.size(), expansions, kind);
    };
    for (auto& span : rendered.literal_spans) {
        span.begin = map_boundary(span.begin, "literal span");
        span.end   = map_boundary(span.end, "literal span");
    }
    if (rendered.rewrite_checkpoint) {
        rendered.rewrite_checkpoint->offset =
            map_boundary(rendered.rewrite_checkpoint->offset, "rewrite checkpoint");
    }
    for (std::size_t& boundary : rendered.rewrite_execution_boundaries) {
        boundary = map_boundary(boundary, "rewrite execution boundary");
    }
    for (std::optional<std::size_t>& boundary : rendered.message_boundaries) {
        if (boundary) { *boundary = map_boundary(*boundary, "message boundary"); }
    }
    for (std::optional<std::size_t>& boundary : rendered.cache_boundaries) {
        if (boundary) { *boundary = map_boundary(*boundary, "cache boundary"); }
    }
    rendered.media_placeholders.clear();
    rendered.text = std::move(expanded);
    return rendered;
}

void add_budget(PreprocessStats& stats, const VisionItem& item, const ProcessorOptions& options) {
    const std::uint64_t spatial =
        checked_mul(static_cast<std::uint64_t>(item.grid.h), item.grid.w, "vision spatial grid");
    const std::uint64_t patches =
        checked_mul(static_cast<std::uint64_t>(item.grid.t), spatial, "vision raw patches");
    const std::uint64_t per_token =
        options.gemma ? std::uint64_t(options.gemma->pooling) * options.gemma->pooling
                      : std::uint64_t(kMerge * kMerge);
    stats.raw_patches += patches;
    stats.vision_tokens += options.deepseek || options.inkling ? media_run_tokens(item, options)
                                                               : patches / per_token;
    stats.attention_pairs += checked_mul(static_cast<std::uint64_t>(item.grid.t),
                                         checked_mul(spatial, spatial, "vision attention pairs"),
                                         "vision attention pairs");
}

void enforce_media_item_resource_limits(const PreprocessStats& stats) {
    if (stats.raw_patches > kMaximumVisionItemRawPatches) {
        throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                             "single media item raw patches exceed Vision execution capacity");
    }
    if (stats.vision_tokens > kMaximumVisionItemTokens) {
        throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                             "single media item tokens exceed Vision execution capacity");
    }
}

void enforce_media_resource_limits(const PreprocessStats& stats, const ProcessorOptions& options) {
    if (stats.raw_patches > options.max_raw_patches) {
        throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                             "vision raw patches exceed processor budget");
    }
    if (stats.vision_tokens > options.max_vision_tokens) {
        throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                             "vision tokens exceed processor budget");
    }
}

void assign_positions(ProcessedInput& output,
                      std::span<const EncodedChat::MediaTokenRun> media_runs) {
    const std::size_t length = output.input_ids.size();
    output.positions.assign(length * 3, 0);
    output.token_types.assign(length, 0);
    auto set = [&](int axis, std::size_t index, std::int32_t value) {
        output.positions[static_cast<std::size_t>(axis) * length + index] = value;
    };
    std::vector<std::size_t> next_frame(output.vision_items.size(), 0);
    std::int32_t current   = 0;
    std::int32_t maximum   = 0;
    std::size_t cursor     = 0;
    const auto assign_text = [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            const std::int32_t position = current + static_cast<std::int32_t>(index - begin);
            for (int axis = 0; axis < 3; ++axis) { set(axis, index, position); }
            maximum = std::max(maximum, position);
        }
        current += static_cast<std::int32_t>(end - begin);
    };

    for (const EncodedChat::MediaTokenRun& run : media_runs) {
        if (run.tokens.begin < cursor || run.tokens.begin > length || run.tokens.count == 0 ||
            run.tokens.count > length - run.tokens.begin) {
            throw std::logic_error("encoded media token runs are not ordered disjoint spans");
        }
        assign_text(cursor, run.tokens.begin);
        if (run.item_index >= output.vision_items.size()) {
            throw std::logic_error("encoded media token run references no Vision item");
        }
        VisionItem& item = output.vision_items[run.item_index];
        if (item.modality != run.modality || run.frame_index != next_frame[run.item_index]) {
            throw std::logic_error("encoded media token run identity is inconsistent");
        }
        const std::size_t expected_frames = item.modality == Modality::Image ? 1 : item.grid.t;
        if (run.frame_index >= expected_frames) {
            throw std::logic_error("encoded media token run exceeds its temporal grid");
        }
        ++next_frame[run.item_index];

        const int gh               = item.grid.h / kMerge;
        const int gw               = item.grid.w / kMerge;
        const std::size_t expected = static_cast<std::size_t>(gh) * gw;
        if (run.tokens.count != expected) {
            throw std::invalid_argument("vision placeholder run does not match media grid");
        }
        item.token_spans.push_back(run.tokens);
        const std::size_t end = run.tokens.begin + run.tokens.count;
        std::fill(output.token_types.begin() + static_cast<std::ptrdiff_t>(run.tokens.begin),
                  output.token_types.begin() + static_cast<std::ptrdiff_t>(end),
                  static_cast<std::uint8_t>(run.modality));
        std::size_t index = run.tokens.begin;
        for (int y = 0; y < gh; ++y) {
            for (int x = 0; x < gw; ++x, ++index) {
                set(0, index, current);
                set(1, index, current + y);
                set(2, index, current + x);
                maximum = std::max({maximum, current, current + y, current + x});
            }
        }
        current += std::max(gh, gw);
        cursor = end;
    }
    assign_text(cursor, length);

    for (std::size_t index = 0; index < output.vision_items.size(); ++index) {
        const VisionItem& item     = output.vision_items[index];
        const std::size_t expected = item.modality == Modality::Image ? 1 : item.grid.t;
        if (next_frame[index] != expected || item.token_spans.size() != expected) {
            throw std::invalid_argument("media grid count does not match placeholder runs");
        }
    }
    output.rope_delta = maximum + 1 - static_cast<std::int32_t>(length);
}

// Gemma 4's positions are the token indices on every axis; media runs set the token types and the
// items' spans.
void assign_gemma_positions(ProcessedInput& output,
                            std::span<const EncodedChat::MediaTokenRun> media_runs,
                            const ProcessorOptions& options) {
    const std::size_t length = output.input_ids.size();
    output.positions.assign(length * 3, 0);
    output.token_types.assign(length, 0);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t index = 0; index < length; ++index) {
            output.positions[static_cast<std::size_t>(axis) * length + index] =
                static_cast<std::int32_t>(index);
        }
    }
    std::vector<std::size_t> next_frame(output.vision_items.size(), 0);
    std::size_t cursor = 0;
    for (const EncodedChat::MediaTokenRun& run : media_runs) {
        if (run.tokens.begin < cursor || run.tokens.begin > length || run.tokens.count == 0 ||
            run.tokens.count > length - run.tokens.begin) {
            throw std::logic_error("encoded media token runs are not ordered disjoint spans");
        }
        if (run.item_index >= output.vision_items.size()) {
            throw std::logic_error("encoded media token run references no Vision item");
        }
        VisionItem& item = output.vision_items[run.item_index];
        if (item.modality != run.modality || run.frame_index != next_frame[run.item_index]) {
            throw std::logic_error("encoded media token run identity is inconsistent");
        }
        const std::size_t expected_frames = item.modality == Modality::Image ? 1 : item.grid.t;
        if (run.frame_index >= expected_frames) {
            throw std::logic_error("encoded media token run exceeds its temporal grid");
        }
        ++next_frame[run.item_index];
        const std::size_t expected = media_run_tokens(item, options);
        if (run.tokens.count != expected) {
            throw std::invalid_argument("vision placeholder run does not match media grid");
        }
        item.token_spans.push_back(run.tokens);
        const std::size_t end = run.tokens.begin + run.tokens.count;
        std::fill(output.token_types.begin() + static_cast<std::ptrdiff_t>(run.tokens.begin),
                  output.token_types.begin() + static_cast<std::ptrdiff_t>(end),
                  static_cast<std::uint8_t>(run.modality));
        cursor = end;
    }
    for (std::size_t index = 0; index < output.vision_items.size(); ++index) {
        const VisionItem& item     = output.vision_items[index];
        const std::size_t expected = item.modality == Modality::Image ? 1 : item.grid.t;
        if (next_frame[index] != expected || item.token_spans.size() != expected) {
            throw std::invalid_argument("media grid count does not match placeholder runs");
        }
    }
    output.rope_delta = 0;
}

int media_token_id(const Tokenizer& tokenizer, std::string_view text) {
    const std::vector<int> ids = tokenizer.encode(text);
    if (ids.size() != 1) {
        throw std::invalid_argument("Vision placeholder must encode to one tokenizer token: " +
                                    std::string(text));
    }
    return ids.front();
}

} // namespace

std::string PreprocessStats::summary() const {
    std::ostringstream out;
    out << "media=" << media_items << " media_bytes=" << media_bytes << " patches=" << raw_patches
        << " vision_tokens=" << vision_tokens << " attention_pairs=" << attention_pairs
        << " prompt_tokens=" << prompt_tokens << " patch_bytes=" << patch_bytes;
    return out.str();
}

std::span<const std::int32_t> ProcessedInput::position_axis(int axis) const {
    if (axis < 0 || axis >= 3 || positions.size() != input_ids.size() * 3) {
        throw std::out_of_range("invalid processed position axis");
    }
    return std::span<const std::int32_t>(positions).subspan(
        static_cast<std::size_t>(axis) * input_ids.size(), input_ids.size());
}

DeepSeekImagePlan deepseek_image_plan(int width, int height, const DeepSeekMediaOptions& options,
                                      std::uint32_t budget) {
    if (width <= 0 || height <= 0 || budget < 4) {
        throw ProcessorError(ProcessorErrorKind::InvalidMedia, "DeepSeek-V4.1 image plan");
    }
    // plan_image_grid, safe_resize and solve_resize_ratio in Python's float arithmetic.
    const int p = options.patch, r = options.merge;
    double w = width, h = height;
    if (w * h < double(options.min_pixels)) {
        const double ratio = std::pow(double(options.min_pixels) / (w * h), 0.5);
        w                  = std::trunc(w * ratio);
        h                  = std::trunc(h * ratio);
    }
    const auto grid = [&](int bh, int bw) {
        return DeepSeekImagePlan{bh, bw, ceil_div(bh / p, r), ceil_div(bw / p, r)};
    };
    DeepSeekImagePlan plan = grid(int(std::ceil(h / p)) * p, int(std::ceil(w / p)) * p);
    if (plan.span_tokens() > budget) {
        const double aspect = h / w;
        const double max_w  = std::sqrt((double(budget) - 2) / aspect + 0.25) - 0.5;
        const double max_h  = max_w * aspect;
        const int cell      = p * r;
        if (max_w < 1.0) {
            plan = grid(int((budget - 2) / 2) * cell, cell);
        } else if (max_h < 1.0) {
            plan = grid(cell, int(budget - 3) * cell);
        } else {
            const double beta = std::min(double(std::int64_t(std::floor(max_w)) * cell) / w,
                                         double(std::int64_t(std::floor(max_h)) * cell) / h);
            plan = grid(int(std::floor(h * beta / p)) * p, int(std::floor(w * beta / p)) * p);
        }
        if (plan.span_tokens() > budget || plan.height <= 0 || plan.width <= 0) {
            throw ProcessorError(ProcessorErrorKind::InvalidMedia,
                                 "DeepSeek-V4.1 image does not fit its token budget");
        }
    }
    return plan;
}

InklingImagePlan inkling_image_plan(int width, int height, const InklingMediaOptions& options,
                                    std::uint32_t budget) {
    if (width <= 0 || height <= 0 || budget == 0) {
        throw ProcessorError(ProcessorErrorKind::InvalidMedia, "Inkling 2 image plan");
    }
    const int m = options.merge, align = options.patch * m;
    InklingImagePlan plan{width, height, ceil_div(width, align) * m, ceil_div(height, align) * m};
    if (plan.cells(m) <= budget) { return plan; }
    // llama.cpp's calc_size_preserved_ratio at this alignment and budget, in its FP32 arithmetic:
    // the sides rounded to whole cells, and past the budget each scaled down by the same factor to
    // whole cells below it.
    const auto round_cells = [&](int side) {
        return std::max(align, int(std::round(float(side) / float(align))) * align);
    };
    const auto floor_cells = [&](float side) {
        return std::max(align, int(std::floor(side / float(align))) * align);
    };
    const std::int64_t max_pixels = std::int64_t(budget) * align * align;
    int w = round_cells(width), h = round_cells(height);
    if (std::int64_t(w) * h > max_pixels) {
        const float beta = std::sqrt(float(height) * float(width) / float(max_pixels));
        h                = floor_cells(float(height) / beta);
        w                = floor_cells(float(width) / beta);
    }
    plan = {w, h, w / options.patch, h / options.patch};
    if (plan.cells(m) > budget) {
        throw ProcessorError(ProcessorErrorKind::InvalidMedia,
                             "Inkling 2 image does not fit its token budget");
    }
    return plan;
}

std::vector<std::uint16_t> inkling_patches(std::span<const std::uint8_t> rgb, int width,
                                           int height, const InklingImagePlan& plan,
                                           const InklingMediaOptions& options) {
    if (rgb.size() != std::size_t(width) * height * 3) {
        throw std::invalid_argument("inkling_patches: invalid sizes");
    }
    std::vector<std::uint16_t> out(std::size_t(plan.grid_h) * plan.grid_w * 3 * options.patch *
                                   options.patch);
    write_inkling_patches(rgb, width, height, plan, options, out.data(), nullptr);
    return out;
}

std::vector<std::uint8_t> pil_bicubic_resize(std::span<const std::uint8_t> rgb, int width,
                                             int height, int out_width, int out_height) {
    if (rgb.size() != std::size_t(width) * height * 3 || out_width <= 0 || out_height <= 0) {
        throw std::invalid_argument("pil_bicubic_resize: invalid sizes");
    }
    const bool horizontal = out_width != width, vertical = out_height != height;
    // Resample.c runs the vertical pass first only when the height shrinks more than twice as
    // much as the width; each pass rounds to 8 bits.
    const bool horizontal_first =
        !((height - out_height) > 0 && (height - out_height) > (width - out_width) * 2);
    std::vector<std::uint8_t> out(rgb.begin(), rgb.end());
    if (horizontal && horizontal_first) { out = pil_pass(out, width, height, out_width, true); }
    if (vertical) {
        out = pil_pass(out, horizontal && horizontal_first ? out_width : width, height, out_height,
                       false);
    }
    if (horizontal && !horizontal_first) { out = pil_pass(out, width, out_height, out_width, true); }
    return out;
}

std::vector<std::uint8_t> deepseek_pad(std::span<const std::uint8_t> rgb, int width, int height,
                                       int out_width, int out_height) {
    // ImageOps.contain: the size within (out_width, out_height) of the image's aspect, Python's
    // round() halves to even.
    int w = out_width, h = out_height;
    const double image_ratio = double(width) / height, box_ratio = double(out_width) / out_height;
    if (image_ratio != box_ratio) {
        if (image_ratio > box_ratio) {
            h = int(std::nearbyint(double(height) / width * out_width));
        } else {
            w = int(std::nearbyint(double(width) / height * out_height));
        }
    }
    std::vector<std::uint8_t> resized =
        std::min(w, h) == 0 ? std::vector<std::uint8_t>{}
                            : pil_bicubic_resize(rgb, width, height, w, h);
    if (w == out_width && h == out_height) { return resized; }
    std::vector<std::uint8_t> out(std::size_t(out_width) * out_height * 3, 127);
    const int x0 = w != out_width ? int(std::nearbyint((out_width - w) * 0.5)) : 0;
    const int y0 = w != out_width ? 0 : int(std::nearbyint((out_height - h) * 0.5));
    for (int y = 0; y < h; ++y) {
        std::copy_n(resized.begin() + std::ptrdiff_t(y) * w * 3, std::size_t(w) * 3,
                    out.begin() + (std::ptrdiff_t(y + y0) * out_width + x0) * 3);
    }
    return out;
}

std::uint32_t gemma_budget_at_most(std::uint64_t bound) {
    std::uint32_t out = 0;
    for (const std::uint32_t budget : kGemmaSoftTokenBudgets) {
        if (budget <= bound) { out = budget; }
    }
    if (out == 0) {
        throw std::invalid_argument("a Gemma 4 media budget is at least " +
                                    std::to_string(kGemmaSoftTokenBudgets.front()) +
                                    " soft tokens");
    }
    return out;
}

EncodedChat encode_rendered_chat(const Tokenizer& tokenizer, const RenderedChat& rendered,
                                 std::size_t maximum_tokens, MediaPads pads) {
    if (!rendered.media_placeholders.empty()) {
        throw std::logic_error("rendered chat contains unexpanded media placeholders");
    }
    EncodedChat encoded;
    std::vector<std::size_t> byte_boundaries;
    byte_boundaries.reserve((rendered.rewrite_checkpoint ? 1U : 0U) +
                            rendered.rewrite_execution_boundaries.size() +
                            rendered.message_boundaries.size() + rendered.cache_boundaries.size() +
                            rendered.media_token_runs.size() * 2U);
    if (rendered.rewrite_checkpoint) {
        byte_boundaries.push_back(rendered.rewrite_checkpoint->offset);
    }
    byte_boundaries.insert(byte_boundaries.end(), rendered.rewrite_execution_boundaries.begin(),
                           rendered.rewrite_execution_boundaries.end());
    for (const std::optional<std::size_t> boundary : rendered.message_boundaries) {
        if (boundary) { byte_boundaries.push_back(*boundary); }
    }
    for (const std::optional<std::size_t> boundary : rendered.cache_boundaries) {
        if (boundary) { byte_boundaries.push_back(*boundary); }
    }
    for (const MediaTokenRunByteSpec& run : rendered.media_token_runs) {
        byte_boundaries.push_back(run.bytes.begin);
        byte_boundaries.push_back(run.bytes.end);
    }

    BoundaryEncodedText tokenized = tokenizer.encode_with_boundaries(
        rendered.text, byte_boundaries, EncodeOptions{.max_tokens = maximum_tokens},
        rendered.literal_spans);
    encoded.input_ids = std::move(tokenized.input_ids);
    if (encoded.input_ids.size() == maximum_tokens) { return encoded; }
    std::size_t boundary_index = 0;
    const auto to_frontier     = [](std::size_t frontier, std::string_view kind) {
        if (frontier > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(std::string(kind) + " token frontier exceeds uint32");
        }
        return static_cast<std::uint32_t>(frontier);
    };
    if (rendered.rewrite_checkpoint) {
        const TokenBoundaryResult& boundary = tokenized.boundaries.at(boundary_index++);
        if (boundary.exact_frontier && *boundary.exact_frontier > 0) {
            encoded.rewrite_checkpoint = RewriteCheckpointSpec{
                .kind     = rendered.rewrite_checkpoint->kind,
                .frontier = to_frontier(*boundary.exact_frontier, "rewrite checkpoint")};
        }
    }
    encoded.rewrite_execution_frontiers.reserve(rendered.rewrite_execution_boundaries.size());
    for (std::size_t remaining = rendered.rewrite_execution_boundaries.size(); remaining != 0;
         --remaining) {
        const TokenBoundaryResult& result = tokenized.boundaries.at(boundary_index++);
        const std::optional<std::uint32_t> frontier =
            result.exact_frontier ? std::optional<std::uint32_t>(to_frontier(
                                        *result.exact_frontier, "rewrite execution boundary"))
                                  : std::nullopt;
        if (frontier && *frontier != 0 &&
            (encoded.rewrite_execution_frontiers.empty() ||
             encoded.rewrite_execution_frontiers.back() != *frontier)) {
            encoded.rewrite_execution_frontiers.push_back(*frontier);
        }
    }
    encoded.message_boundaries.resize(rendered.message_boundaries.size());
    for (std::size_t index = 0; index < rendered.message_boundaries.size(); ++index) {
        if (rendered.message_boundaries[index]) {
            const TokenBoundaryResult& boundary = tokenized.boundaries.at(boundary_index++);
            if (boundary.exact_frontier) {
                encoded.message_boundaries[index] =
                    to_frontier(*boundary.exact_frontier, "message boundary");
            }
        }
    }
    encoded.cache_boundaries.resize(rendered.cache_boundaries.size());
    for (std::size_t index = 0; index < rendered.cache_boundaries.size(); ++index) {
        if (rendered.cache_boundaries[index]) {
            const TokenBoundaryResult& boundary = tokenized.boundaries.at(boundary_index++);
            if (boundary.exact_frontier)
                encoded.cache_boundaries[index] =
                    to_frontier(*boundary.exact_frontier, "cache boundary");
        }
    }
    encoded.media_token_runs.reserve(rendered.media_token_runs.size());
    for (const MediaTokenRunByteSpec& run : rendered.media_token_runs) {
        const TokenBoundaryResult& begin = tokenized.boundaries.at(boundary_index++);
        const TokenBoundaryResult& end   = tokenized.boundaries.at(boundary_index++);
        if (!begin.exact_frontier || !end.exact_frontier ||
            *begin.exact_frontier >= *end.exact_frontier) {
            throw std::logic_error("media token run is not an exact nonempty token span");
        }
        const std::size_t token_begin = *begin.exact_frontier;
        const std::size_t token_end   = *end.exact_frontier;
        const int expected_token =
            media_token_id(tokenizer, run.modality == Modality::Image ? pads.image : pads.video);
        if (!std::all_of(encoded.input_ids.begin() + static_cast<std::ptrdiff_t>(token_begin),
                         encoded.input_ids.begin() + static_cast<std::ptrdiff_t>(token_end),
                         [expected_token](int token) { return token == expected_token; })) {
            throw std::logic_error("media token run contains a non-media token");
        }
        encoded.media_token_runs.push_back(EncodedChat::MediaTokenRun{
            .tokens      = TokenSpan{.begin = token_begin, .count = token_end - token_begin},
            .modality    = run.modality,
            .item_index  = run.item_index,
            .frame_index = run.frame_index,
        });
    }
    if (boundary_index != tokenized.boundaries.size()) {
        throw std::logic_error("rendered token boundary result count changed during encoding");
    }
    return encoded;
}

Processor::Processor(const Tokenizer& tokenizer, const CompiledChatTemplate& chat_template,
                     ProcessorOptions options, std::shared_ptr<MediaPreprocessCache> media_cache)
    : tokenizer_(tokenizer), chat_template_(chat_template), options_(std::move(options)),
      media_cache_(std::move(media_cache)) {
    if (options_.max_encoded_media_bytes == 0 || options_.max_decoded_pixels == 0 ||
        options_.max_decoded_video_pixels == 0 || options_.max_raw_patches == 0 ||
        options_.max_vision_tokens == 0 || options_.image_min_pixels == 0 ||
        options_.image_max_pixels < options_.image_min_pixels || options_.video_min_pixels == 0 ||
        options_.video_max_pixels < options_.video_min_pixels || !(options_.video_fps > 0.0) ||
        options_.video_min_frames <= 0 || options_.video_max_frames < options_.video_min_frames ||
        options_.max_video_source_frames < options_.video_max_frames ||
        !(options_.max_video_duration_seconds > 0.0)) {
        throw std::invalid_argument("processor budgets must be positive");
    }
    if (!media_cache_) { throw std::invalid_argument("processor media cache must not be null"); }
    if (options_.inkling) {
        (void)media_token_id(tokenizer_, kInklingMediaPads.image);
        return;
    }
    if (options_.deepseek) {
        (void)media_token_id(tokenizer_, kDeepSeekMediaPads.image);
        return;
    }
    if (options_.gemma) {
        for (const std::string_view token :
             {kGemmaMediaPads.image, kGemmaMediaPads.video, kGemmaBoi, kGemmaEoi}) {
            (void)media_token_id(tokenizer_, token);
        }
        return;
    }
    (void)media_token_id(tokenizer_, kImagePad);
    (void)media_token_id(tokenizer_, kVideoPad);
}

std::size_t Processor::count_tokens(std::vector<ChatMessage> messages,
                                    ChatRenderOptions render_options,
                                    const PreparationControl& control) const {
    check_preparation_control(control);
    const std::vector<ChatPart*> parts = media_parts(messages);
    refuse_image_only_video(parts, options_);
    (void)validate_media_inputs(parts, options_);
    RenderedChat rendered = chat_template_.render(messages, std::move(render_options), control);
    MediaPreparationPermit request_permit = media_cache_->acquire_request(control);
    const media::decode::Policy policy{
        .max_bytes                  = options_.max_encoded_media_bytes,
        .max_decoded_pixels         = options_.max_decoded_pixels,
        .max_decoded_video_pixels   = options_.max_decoded_video_pixels,
        .max_video_source_frames    = options_.max_video_source_frames,
        .max_video_duration_seconds = options_.max_video_duration_seconds,
        .checkpoint                 = [&control] { check_preparation_control(control); },
    };
    std::vector<VisionItem> items;
    items.reserve(parts.size());
    PreprocessStats stats;
    try {
        for (const ChatPart* part : parts) {
            check_preparation_control(control);
            const ProcessorOptions options = part_options(*part, options_);
            enforce_image_resize_policy(*part, options, policy);
            VisionItem item = options_.inkling    ? inspect_inkling_item(*part, options_, policy)
                              : options_.deepseek ? inspect_deepseek_item(*part, options_, policy)
                              : options_.gemma  ? inspect_gemma_item(*part, options_, policy)
                              : part->kind == ChatPartKind::Image
                                  ? inspect_image_item(part->media.bytes, options, policy)
                                  : inspect_video_item(part->media.bytes, options, policy);
            PreprocessStats item_stats;
            add_budget(item_stats, item, options_);
            enforce_media_item_resource_limits(item_stats);
            add_budget(stats, item, options_);
            enforce_media_resource_limits(stats, options_);
            items.push_back(std::move(item));
        }
    } catch (const media::decode::Error& error) { throw_decode_error(error); }
    rendered = expand_placeholders(std::move(rendered), items, options_);
    const std::size_t count =
        encode_rendered_chat(tokenizer_, rendered, std::numeric_limits<std::size_t>::max(),
                             media_pads(options_))
            .input_ids.size();
    check_preparation_control(control, "tokenization");
    return count;
}

void bound_merged_tokens(ProcessorOptions& options, std::uint64_t merged_tokens) {
    if (merged_tokens == 0) { return; }
    const auto factor_pixels =
        static_cast<std::uint64_t>(kFactor) * static_cast<std::uint64_t>(kFactor);
    if (merged_tokens > std::numeric_limits<std::uint64_t>::max() / (factor_pixels * kTemporal)) {
        return;
    }
    const std::uint64_t image_pixels = merged_tokens * factor_pixels;
    const std::uint64_t video_pixels = image_pixels * static_cast<std::uint64_t>(kTemporal);
    options.image_max_pixels         = std::min(options.image_max_pixels, image_pixels);
    options.video_max_pixels         = std::min(options.video_max_pixels, video_pixels);
    options.image_min_pixels         = std::min(options.image_min_pixels, options.image_max_pixels);
    options.video_min_pixels         = std::min(options.video_min_pixels, options.video_max_pixels);
}

ProcessedInput Processor::process(std::vector<ChatMessage> messages,
                                  ChatRenderOptions render_options,
                                  const PreparationControl& control,
                                  std::size_t maximum_prompt_tokens) const {
    check_preparation_control(control);
    const std::vector<ChatPart*> parts = media_parts(messages);
    refuse_image_only_video(parts, options_);
    const std::size_t media_bytes      = validate_media_inputs(parts, options_);
    RenderedChat rendered = chat_template_.render(messages, std::move(render_options), control);
    const std::size_t encode_limit =
        maximum_prompt_tokens == std::numeric_limits<std::size_t>::max()
            ? maximum_prompt_tokens
            : maximum_prompt_tokens + 1U;
    double preliminary_tokenize_seconds = 0.0;
    if (maximum_prompt_tokens != std::numeric_limits<std::size_t>::max()) {
        const auto preliminary_started = Clock::now();
        const std::size_t preliminary_tokens =
            tokenizer_
                .encode_with_boundaries(rendered.text, {},
                                        EncodeOptions{.max_tokens = encode_limit},
                                        rendered.literal_spans)
                .input_ids.size();
        preliminary_tokenize_seconds =
            std::chrono::duration<double>(Clock::now() - preliminary_started).count();
        check_preparation_control(control, "tokenization");
        if (preliminary_tokens > maximum_prompt_tokens) {
            throw ProcessorError(ProcessorErrorKind::ContextLengthExceeded,
                                 "prepared prompt exceeds Engine max_context " +
                                     std::to_string(maximum_prompt_tokens));
        }
    }
    MediaPreparationPermit request_permit = media_cache_->acquire_request(control);
    std::atomic<bool> stop_preparation{false};
    const PreparationControl worker_control{
        .deadline     = control.deadline,
        .cancellation = CancellationView(
            [&stop_preparation] { return stop_preparation.load(std::memory_order_relaxed); }),
    };
    const media::decode::Policy policy{
        .max_bytes                  = options_.max_encoded_media_bytes,
        .max_decoded_pixels         = options_.max_decoded_pixels,
        .max_decoded_video_pixels   = options_.max_decoded_video_pixels,
        .max_video_source_frames    = options_.max_video_source_frames,
        .max_video_duration_seconds = options_.max_video_duration_seconds,
        .checkpoint = [&worker_control] { check_preparation_control(worker_control); },
    };
    try {
        for (const ChatPart* part : parts) {
            enforce_image_resize_policy(*part, part_options(*part, options_), policy);
        }
    } catch (const media::decode::Error& error) { throw_decode_error(error); }
    ProcessedInput output;
    std::vector<VisionItem> items;
    items.reserve(parts.size());
    PreprocessStats stats;
    stats.media_items      = parts.size();
    stats.media_bytes      = media_bytes;
    stats.tokenize_seconds = preliminary_tokenize_seconds;

    std::vector<PendingMedia> pending_items;
    pending_items.reserve(parts.size());
    ConcurrentMediaBudget request_budget(options_);
    std::exception_ptr preparation_error;
    const auto media_phase_started = Clock::now();
    for (ChatPart* part : parts) {
        try {
            check_preparation_control(control);
            const auto digest = part_digest(
                *part,
                sha256(part->media.bytes, [&control] { check_preparation_control(control); }),
                options_);
            const ChatPartKind kind = part->kind;
            const MediaCacheKey key{
                .digest   = digest,
                .modality = kind == ChatPartKind::Image ? Modality::Image : Modality::Video,
            };
            PendingMedia pending = media_cache_->begin_prepare(
                key, worker_control,
                [this, part, kind, digest, options = part_options(*part, options_), &policy,
                 &request_budget, &worker_control]() {
                    Prepared built;
                    if (options.inkling) {
                        built = prepare_inkling_image(
                            part->media.bytes, inkling_part_budget(*part, *options.inkling),
                            options, policy, *media_cache_, request_budget, worker_control);
                    } else if (options.deepseek) {
                        built = prepare_deepseek_image(
                            part->media.bytes, deepseek_part_budget(*part, *options.deepseek),
                            options, policy, *media_cache_, request_budget, worker_control);
                    } else if (options.gemma) {
                        const std::uint32_t budget = gemma_part_budget(*part, *options.gemma);
                        built = kind == ChatPartKind::Image
                                    ? prepare_gemma_image(part->media.bytes, budget, options,
                                                          policy, *media_cache_, request_budget,
                                                          worker_control)
                                    : prepare_gemma_video(part->media.bytes, budget, options,
                                                          policy, *media_cache_, request_budget,
                                                          worker_control);
                    } else {
                        built = kind == ChatPartKind::Image
                                    ? prepare_image(part->media.bytes, options, policy,
                                                    *media_cache_, request_budget, worker_control)
                                    : prepare_video(part->media.bytes, options, policy,
                                                    *media_cache_, request_budget, worker_control);
                    }
                    built.item.content_digest = digest;
                    return PreparedMedia{std::move(built.item), std::move(built.payload)};
                });
            pending_items.push_back(std::move(pending));
        } catch (...) {
            preparation_error = std::current_exception();
            stop_preparation.store(true, std::memory_order_relaxed);
            break;
        }
    }
    MediaCacheRequestStats cache_stats;
    std::size_t patch_cursor = 0;
    const std::size_t patch_features =
        options_.inkling    ? std::size_t(3) * options_.inkling->patch * options_.inkling->patch
        : options_.deepseek ? std::size_t(3) * options_.deepseek->patch * options_.deepseek->patch
        : options_.gemma  ? std::size_t(3) * options_.gemma->patch * options_.gemma->patch
                          : std::size_t(kPatchFeatures);
    for (PendingMedia& pending : pending_items) {
        PreparedMedia media;
        try {
            media = media_cache_->await(pending, preparation_error ? PreparationControl{} : control,
                                        cache_stats);
        } catch (...) {
            if (!preparation_error) {
                preparation_error = std::current_exception();
                stop_preparation.store(true, std::memory_order_relaxed);
            }
            MediaCacheRequestStats discarded;
            try {
                (void)media_cache_->await(pending, {}, discarded);
            } catch (...) {}
            continue;
        }
        if (preparation_error) { continue; }
        if (!media.payload || media.payload->patch_elements % patch_features != 0) {
            preparation_error = std::make_exception_ptr(
                std::logic_error("preprocessed patch buffer is not row aligned"));
            stop_preparation.store(true, std::memory_order_relaxed);
            continue;
        }
        VisionItem item  = media.item;
        item.patch_begin = patch_cursor;
        item.patch_count = media.payload->patch_elements / patch_features;
        patch_cursor += item.patch_count;
        try {
            PreprocessStats item_stats;
            add_budget(item_stats, item, options_);
            enforce_media_item_resource_limits(item_stats);
            add_budget(stats, item, options_);
            enforce_media_resource_limits(stats, options_);
        } catch (...) {
            preparation_error = std::current_exception();
            stop_preparation.store(true, std::memory_order_relaxed);
            continue;
        }
        stats.patch_bytes += media.payload->patch_elements * sizeof(std::uint16_t);
        items.push_back(std::move(item));
        output.media_payloads.push_back(std::move(media.payload));
    }
    for (ChatPart* part : parts) { std::vector<std::uint8_t>().swap(part->media.bytes); }
    if (preparation_error) {
        try {
            std::rethrow_exception(preparation_error);
        } catch (const media::decode::Error& error) { throw_decode_error(error); }
    }
    if (patch_cursor != stats.raw_patches || output.media_payloads.size() != items.size()) {
        throw std::logic_error("preprocessed patch count does not match processor budget");
    }
    const double media_preprocess_seconds =
        std::chrono::duration<double>(Clock::now() - media_phase_started).count();
    request_permit.reset();

    check_preparation_control(control);
    rendered                    = expand_placeholders(std::move(rendered), items, options_);
    const auto tokenize_started = Clock::now();
    EncodedChat encoded         = encode_rendered_chat(
        tokenizer_, rendered, encode_limit, media_pads(options_));
    stats.tokenize_seconds +=
        std::chrono::duration<double>(Clock::now() - tokenize_started).count();
    check_preparation_control(control, "tokenization");
    if (encoded.input_ids.size() > maximum_prompt_tokens) {
        throw ProcessorError(ProcessorErrorKind::ContextLengthExceeded,
                             "prepared prompt exceeds Engine max_context " +
                                 std::to_string(maximum_prompt_tokens));
    }
    output.starts_in_reasoning         = rendered.starts_in_reasoning;
    output.input_ids                   = std::move(encoded.input_ids);
    output.rewrite_checkpoint          = encoded.rewrite_checkpoint;
    output.rewrite_execution_frontiers = std::move(encoded.rewrite_execution_frontiers);
    output.message_boundaries          = std::move(encoded.message_boundaries);
    output.cache_boundaries            = std::move(encoded.cache_boundaries);
    stats.prompt_tokens                = output.input_ids.size();
    enforce_media_resource_limits(stats, options_);

    stats.media_cache_hits              = cache_stats.hits;
    stats.media_cache_misses            = cache_stats.misses;
    stats.media_singleflight_waits      = cache_stats.singleflight_waits;
    stats.built_patch_bytes             = cache_stats.built_patch_bytes;
    stats.reused_patch_bytes            = cache_stats.reused_patch_bytes;
    stats.media_preprocess_seconds      = media_preprocess_seconds;
    stats.media_preprocess_work_seconds = cache_stats.build_seconds;
    output.vision_items                 = std::move(items);
    if (options_.gemma || options_.deepseek || options_.inkling) {
        assign_gemma_positions(output, encoded.media_token_runs, options_);
    } else {
        assign_positions(output, encoded.media_token_runs);
    }
    check_preparation_control(control);
    output.stats = stats;
    return output;
}

} // namespace ninfer::models::qwen3_5::frontend

// Inkling 2's vision path against a real artifact (skipped without one):
//   NINFER_INKLING_VISION_ARTIFACT  a .ninfer converted with --components text,vision
//   NINFER_INKLING_VISION_IMAGE     an image file
//   NINFER_INKLING_VISION_DIR       a directory: the test writes the frontend's patches there
//                                   (patches.bin, BF16 words, and grid.json) and each tower's span
//                                   rows (device.bin, overlay.bin: FP32 [span][text hidden]); when
//                                   reference.bin is there too (tools/reference/inkling2_vision.py
//                                   --mmproj <the source mmproj> --patches patches.bin --grid W H
//                                   --out reference.bin: FP64 projector rows, raw), the towers'
//                                   rows are checked against it
// The frontend prepares a user turn holding the image: a span of one <|unused_200054|> token per
// 2 x 2 cell of its patch grid within the release's 4096 cells, and the item's patches are the
// grid's. The resident tower then encodes them, and the tower whose matrices stream from pinned
// memory gives the same rows bit for bit. Against the FP64 oracle, the tower, which rounds its
// matrices' inputs, Q, K, V, the attention probabilities and a Q8_0 matrix's decoded rows to BF16
// (its products stay FP32), keeps every row's cosine at 0.98 or more and the relative RMS error
// within 3%. The experts stay in the artifact's files (disk residency): the tower needs none of
// them.

#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/device_buffer.h"
#include "models/inkling2/model.h"
#include "models/inkling2/vision.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"

#include <cuda_bf16.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace ink  = ninfer::models::inkling2;
namespace qwen = ninfer::models::qwen3_5;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + path.string()); }
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// Against the FP64 oracle: the smallest per-row cosine and the relative RMS error.
void compare(const std::vector<float>& got, const std::vector<double>& want, int width,
             const std::string& label, double min_cosine, double max_relative) {
    if (got.size() != want.size()) {
        expect(false, label + ": row count differs from the reference");
        return;
    }
    double err = 0.0, norm = 0.0, worst = 1.0;
    for (std::size_t t = 0; t < got.size() / width; ++t) {
        double dot = 0.0, a = 0.0, b = 0.0;
        for (int c = 0; c < width; ++c) {
            const double x = got[t * width + c], y = want[t * width + c];
            dot += x * y;
            a += x * x;
            b += y * y;
            err += (x - y) * (x - y);
            norm += y * y;
        }
        worst = std::min(worst, dot / std::sqrt(a * b));
    }
    const double relative = std::sqrt(err / norm);
    std::cout << label << ": worst row cosine " << worst << ", relative RMS error " << relative
              << '\n';
    expect(worst >= min_cosine, label + ": a row's cosine is below " + std::to_string(min_cosine));
    expect(relative <= max_relative, label + ": relative RMS error above " +
                                         std::to_string(max_relative));
}

} // namespace

int main() {
    const char* artifact_path = std::getenv("NINFER_INKLING_VISION_ARTIFACT");
    const char* image_path    = std::getenv("NINFER_INKLING_VISION_IMAGE");
    const char* dir_value     = std::getenv("NINFER_INKLING_VISION_DIR");
    if (artifact_path == nullptr || image_path == nullptr || dir_value == nullptr) {
        std::cout << "SKIP: set NINFER_INKLING_VISION_ARTIFACT, _IMAGE and _DIR\n";
        return 77;
    }
    const std::filesystem::path dir(dir_value);
    DeviceContext device(0);
    const artifact::Reader reader(artifact_path);
    const auto load = [&](ink::VisionPlacement placement) {
        return ink::load_model(reader,
                               {.artifact = artifact_path,
                                .experts  = ExpertResidency::Disk,
                                .vision   = placement},
                               device);
    };
    auto model = load(ink::VisionPlacement::Device);
    const ink::VisionConfig vc = *model->vision_config();

    auto frontend = qwen::make_frontend(model->resources(),
                                        {.architecture   = models::Architecture::Inkling2,
                                         .vision_enabled = true,
                                         .max_context    = 16384});
    PromptInput input;
    ChatMessage user;
    user.role = ChatRole::User;
    MessagePart image;
    image.kind        = MessagePartKind::Media;
    image.media.kind  = MediaKind::Image;
    image.media.bytes = read_bytes(image_path);
    user.parts.push_back(std::move(image));
    MessagePart text;
    text.text = "Describe this image.";
    user.parts.push_back(std::move(text));
    input.messages.push_back(std::move(user));
    const auto prompt = frontend.prepare(std::move(input));
    const auto& data  = qwen::PreparedPromptAccess::view(prompt);

    expect(data.vision_items.size() == 1, "one vision item");
    if (data.vision_items.size() != 1) { return 1; }
    const auto& item = data.vision_items.front();
    const int gw = item.grid.width, gh = item.grid.height;
    const std::uint32_t span_rows = ink::VisionTower::span_rows(gw, gh);
    std::cout << "image: patch grid " << gw << " x " << gh << ", a span of " << span_rows
              << " positions\n";
    expect(gw % 2 == 0 && gh % 2 == 0, "whole cells");
    expect(span_rows <= vc.max_image_tokens, "within the release's budget");
    expect(item.token_spans.size() == 1 && item.token_spans[0].count == span_rows,
           "one span of the image's positions");
    const auto ids  = std::span<const TokenId>(data.token_ids);
    const auto span = item.token_spans.front();
    bool placeholders = span.begin + span.count <= ids.size();
    for (std::size_t i = 0; placeholders && i < span.count; ++i) {
        placeholders = ids[span.begin + i] == TokenId(vc.image_token_id);
    }
    expect(placeholders, "the span is image tokens");

    const auto patches = data.media_payloads.front()->span();
    expect(patches.size() == std::size_t(gw) * gh * vc.patch_features(), "the item's patches");
    std::filesystem::create_directories(dir);
    {
        std::ofstream out(dir / "patches.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(patches.data()), std::streamsize(patches.size() * 2));
        std::ofstream grid(dir / "grid.json");
        grid << nlohmann::json{{"w", gw}, {"h", gh}, {"span", span_rows}}.dump() << '\n';
    }
    const std::vector<std::uint16_t> host_patches(patches.begin(), patches.end());
    const int width = int(vc.text_hidden_size);
    model.reset();

    // One tower over a fresh load of the artifact with the given placement (each load holds the
    // dense text weights too, so one at a time): the span's rows as FP32 values, saved as
    // <name>.bin.
    const auto encode = [&](ink::VisionPlacement placement, const char* name) {
        auto loaded = load(placement);
        const ink::VisionTower tower(*loaded, std::uint32_t(gw * gh));
        expect(tower.streamed() == (placement == ink::VisionPlacement::Pinned),
               std::string(name) + ": streamed only from pinned matrices");
        WorkspaceArena workspace(tower.workspace_bytes());
        DeviceBuffer d_patches(host_patches.size() * 2), d_out(std::size_t(span_rows) * width * 2);
        d_patches.copy_from_host(host_patches.data(), host_patches.size() * 2);
        tower.encode(static_cast<const __nv_bfloat16*>(d_patches.p), gw, gh,
                     static_cast<__nv_bfloat16*>(d_out.p), workspace, device.stream);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        std::vector<__nv_bfloat16> raw(std::size_t(span_rows) * width);
        d_out.copy_to_host(raw.data(), raw.size() * 2);
        std::vector<float> values(raw.size());
        for (std::size_t i = 0; i < raw.size(); ++i) { values[i] = __bfloat162float(raw[i]); }
        std::ofstream(dir / (std::string(name) + ".bin"), std::ios::binary)
            .write(reinterpret_cast<const char*>(values.data()), std::streamsize(values.size() * 4));
        return values;
    };
    const auto resident = encode(ink::VisionPlacement::Device, "device");
    // The streamed tower runs the same kernels over the same bytes.
    const auto overlay = encode(ink::VisionPlacement::Pinned, "overlay");
    expect(overlay == resident, "the streamed tower's rows equal the resident tower's");
    expect(std::all_of(resident.begin(), resident.end(), [](float v) { return std::isfinite(v); }),
           "finite rows");

    const auto reference_path = dir / "reference.bin";
    if (!std::filesystem::exists(reference_path)) {
        std::cout << "patches written to " << dir << "; run tools/reference/inkling2_vision.py "
                     "and save reference.bin to compare\n";
    } else {
        const auto bytes = read_bytes(reference_path);
        std::vector<double> reference(bytes.size() / sizeof(double));
        std::memcpy(reference.data(), bytes.data(), reference.size() * sizeof(double));
        compare(resident, reference, width, "device tower", 0.98, 0.03);
    }
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " inkling2 vision real\n";
    return failures == 0 ? 0 : 1;
}

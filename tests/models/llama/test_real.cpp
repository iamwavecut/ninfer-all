// Manual full-model qualification: <artifact> logits <ids.json> <out.f32>, or <artifact> capacity.
// Capacity populates the full host cache with an explicit synthetic state, then executes the
// last legal token and checks the next token is rejected. It does not claim full-prompt quality.
#include "models/llama/executor.h"
#include "artifact/reader.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda_bf16.h>
#include <cstring>
using namespace ninfer;

int main(int argc, char** argv) {
    try {
        if (argc < 3)
            throw std::invalid_argument("artifact logits ids.json out.f32 | artifact capacity");
        DeviceContext device(0);
        artifact::Reader reader(argv[1]);
        auto model = models::llama::load_model(reader, {.artifact = argv[1]}, device);
        device.synchronize();
        models::llama::ExecutorOptions options;
        options.max_context =
            std::min<std::uint32_t>(std::string_view(argv[2]) == "capacity-fast" ? 8192 : 262144,
                                    model->config().max_position_embeddings);
        models::llama::Executor executor(*model, device, options);
        if (std::string_view(argv[2]) == "state" || std::string_view(argv[2]) == "state-boundary") {
            if (argc != 4) throw std::invalid_argument("state requires ids.json");
            std::ifstream input(argv[3]);
            nlohmann::json ids;
            input >> ids;
            auto tokens                  = ids.get<std::vector<std::int32_t>>();
            const std::uint32_t base     = std::string_view(argv[2]) == "state-boundary"
                                               ? executor.options().resident_tokens - 1
                                               : 1023;
            constexpr std::uint32_t tail = 17;
            if (tokens.size() < base + tail)
                throw std::invalid_argument(
                    "state input must cover its resident boundary plus 17 tokens");
            for (std::size_t at = 0; at < base; at += options.prefill_chunk) {
                auto n = std::min<std::size_t>(options.prefill_chunk, base - at);
                executor.forward(0, std::span(tokens).subspan(at, n), 0);
            }
            models::llama::SequenceSnapshot snapshot;
            executor.snapshot(0, snapshot);
            const std::size_t page_bytes = 8u << 20;
            auto bytes                   = executor.image_bytes(base);
            PinnedHostBuffer storage(((bytes + page_bytes - 1) / page_bytes) * page_bytes);
            std::vector<std::byte*> pages;
            for (std::size_t at = 0; at < storage.size(); at += page_bytes)
                pages.push_back(static_cast<std::byte*>(storage.data()) + at);
            auto image         = executor.save_image(0, &snapshot, pages, page_bytes);
            const auto capture = [&](int rows = 1) {
                std::vector<std::uint16_t> out(std::size_t(model->config().vocab_size) * rows);
                auto logits = executor.logits(rows);
                CUDA_CHECK(
                    cudaMemcpy(out.data(), logits.data, out.size() * 2, cudaMemcpyDeviceToHost));
                return out;
            };
            auto continuation = std::span(tokens).subspan(base, tail);
            executor.forward(0, continuation, 1);
            auto expected = capture();
            executor.restore(0, snapshot);
            executor.forward(0, continuation, 1);
            if (capture() != expected)
                throw std::runtime_error("snapshot changed continuation logits");
            executor.reset(0);
            executor.load_image(0, image, pages, page_bytes);
            executor.forward(0, continuation, 1);
            if (capture() != expected)
                throw std::runtime_error("host image changed continuation logits");
            executor.restore(0, snapshot);
            executor.forward(0, continuation.first(2), 1);
            auto first = capture();
            executor.restore(0, snapshot);
            executor.forward(0, continuation.subspan(2, 3), 1);
            auto second = capture();
            executor.restore(0, snapshot);
            const std::uint32_t lengths[] = {2, 3};
            executor.forward_branches(0, continuation.first(5), lengths);
            auto branches = capture(2);
            if (executor.position(0) != base ||
                !std::equal(first.begin(), first.end(), branches.begin()) ||
                !std::equal(second.begin(), second.end(), branches.begin() + first.size()))
                throw std::runtime_error("branch read changed logits or prefix position");
            std::cout << "snapshot, host image and branches preserve exact BF16 continuation "
                         "logits across position "
                      << base + 1 << "\n";
        } else if (std::string_view(argv[2]) == "capacity" ||
                   std::string_view(argv[2]) == "capacity-fast") {

            constexpr std::size_t page_bytes = 8u << 20;
            PinnedHostBuffer zero(page_bytes);
            std::memset(zero.data(), 0, page_bytes);
            auto count = executor.image_bytes(options.max_context - 1);
            std::vector<std::byte*> pages((count + page_bytes - 1) / page_bytes,
                                          static_cast<std::byte*>(zero.data()));
            auto start = std::chrono::steady_clock::now();
            executor.load_image(0, {.position = options.max_context - 1}, pages, page_bytes);
            std::cout << "host KV backed " << executor.memory().host_kv_bytes
                      << " bytes at position " << executor.position(0) << std::endl;
            const auto decode_start = std::chrono::steady_clock::now();
            const std::int32_t id   = 0;
            executor.forward(0, {&id, 1}, 1);
            std::cout << "full-cache one-token seconds "
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                       decode_start)
                             .count()
                      << std::endl;
            if (executor.position(0) != options.max_context)
                throw std::runtime_error("wrong final frontier");
            auto logits = executor.logits(1);
            std::vector<__nv_bfloat16> values(model->config().vocab_size);
            CUDA_CHECK(
                cudaMemcpy(values.data(), logits.data, values.size() * 2, cudaMemcpyDeviceToHost));
            for (auto v : values)
                if (!std::isfinite(__bfloat162float(v)))
                    throw std::runtime_error("nonfinite boundary logits");
            bool rejected = false;
            try {
                executor.forward(0, {&id, 1}, 1);
            } catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("accepted token past context capacity");
            std::cout
                << "last legal token accepted; next rejected; elapsed "
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                << " s" << std::endl;
        } else {
            if (argc != 5) throw std::invalid_argument("logits requires ids.json and out.f32");
            std::ifstream input(argv[3]);
            nlohmann::json ids;
            input >> ids;
            auto tokens = ids.get<std::vector<std::int32_t>>();
            for (std::size_t at = 0; at < tokens.size(); at += options.prefill_chunk) {
                auto n = std::min<std::size_t>(options.prefill_chunk, tokens.size() - at);
                executor.forward(0, std::span(tokens).subspan(at, n),
                                 at + n == tokens.size() ? 1 : 0);
            }
            auto logits = executor.logits(1);
            std::vector<__nv_bfloat16> values(model->config().vocab_size);
            std::vector<float> out(values.size());
            CUDA_CHECK(
                cudaMemcpy(values.data(), logits.data, values.size() * 2, cudaMemcpyDeviceToHost));
            for (std::size_t i = 0; i < out.size(); ++i) out[i] = __bfloat162float(values[i]);
            std::ofstream file(argv[4], std::ios::binary);
            file.write(reinterpret_cast<const char*>(out.data()), out.size() * 4);
            std::cout << tokens.size() << " prompt tokens, " << out.size() << " logits saved\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

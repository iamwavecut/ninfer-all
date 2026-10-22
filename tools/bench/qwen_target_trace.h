#pragma once
// Diagnostic-only instrumentation. Included only by the separately applied trace patch;
// no product target includes this file in a normal build. Synchronous copies perturb timing.
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::diagnostic {
inline void target_trace(const std::string& kind,
                         std::initializer_list<std::pair<const char*, Tensor>> values,
                         cudaStream_t stream) {
    const char* root = std::getenv("NINFER_TARGET_TRACE");
    if (!root || !*root) return;
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) return;
    static std::atomic<unsigned> serial{0};
    const auto id = serial++;
    const auto path = std::filesystem::path(root);
    std::filesystem::create_directories(path);
    std::ofstream index(path / "index.jsonl", std::ios::app);
    for (const auto& [name, tensor] : values) {
        const std::string filename = std::to_string(id) + "-" + kind + "-" + name + ".bin";
        std::vector<char> bytes(tensor.bytes());
        if (cudaMemcpyAsync(bytes.data(), tensor.data, bytes.size(), cudaMemcpyDeviceToHost, stream)
                != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess)
            throw std::runtime_error("target trace transfer failed");
        std::ofstream output(path / filename, std::ios::binary);
        output.write(bytes.data(), bytes.size());
        if (!output) throw std::runtime_error("target trace write failed");
        index << "{\"group\":" << id << ",\"kind\":\"" << kind << "\",\"name\":\"" << name
              << "\",\"file\":\"" << filename << "\",\"dtype\":" << int(tensor.dtype)
              << ",\"shape\":[" << tensor.ne[0] << ',' << tensor.ne[1] << ',' << tensor.ne[2]
              << ',' << tensor.ne[3] << "]}\n";
    }
}
} // namespace ninfer::diagnostic

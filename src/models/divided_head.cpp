#include "models/divided_head.h"

#include "core/device_buffer.h"
#include "core/device_wait.h"
#include "core/tensor_split.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models {

namespace {

constexpr std::uint64_t kRowUnit = 256;

} // namespace

bool DividedHead::divides(const Weight& head, std::size_t ranks) {
    return ranks >= 2 && head.input_columns == nullptr && head.qdata != nullptr &&
           std::uint64_t(head.n) > kRowUnit * ranks;
}

DividedHead::DividedHead(const DeviceContext& device, const Weight& head, std::int64_t row_bytes,
                         std::span<const double> speed, std::size_t input_bytes, int columns)
    : devices_(device.device_ids()), columns_(columns), input_bytes_(input_bytes) {
    const std::size_t ranks = device.size();
    if (!divides(head, ranks) || speed.size() != ranks || columns <= 0 || input_bytes == 0 ||
        row_bytes <= 0) {
        throw std::invalid_argument("a divided head needs a GGUF head, a speed per rank and a "
                                    "positive call size");
    }
    const auto vocab = std::uint64_t(head.n);
    rows_            = split_shares(vocab, unit_positions(vocab, kRowUnit), speed);
    ranks_.resize(ranks);
    std::uint64_t widest = 0;
    for (std::size_t r = 0; r < ranks; ++r) {
        const std::uint64_t first = rows_[r], count = rows_[r + 1] - rows_[r];
        widest                    = std::max(widest, count);
        Rank& rank                = ranks_[r];
        rank.piece                = head;
        rank.piece.n              = std::int32_t(count);
        rank.piece.shape[1]       = std::int32_t(count);
        rank.piece.padded_shape[1] = std::int32_t(count);
        rank.piece.payload_bytes  = count * std::uint64_t(row_bytes);
        const auto* rows_begin =
            static_cast<const std::byte*>(head.qdata) + first * std::uint64_t(row_bytes);
        RankBinding bind(device, r);
        if (r == 0) {
            rank.piece.qdata = rows_begin;
            rank.own         = DeviceBuffer(std::size_t(columns) * count * 2);
        } else {
            rank.storage = DeviceBuffer(count * std::uint64_t(row_bytes));
            CUDA_CHECK(ninfer::copy_between_devices_and_wait(
                rank.storage.p, device.rank(r).device, rows_begin, device.rank(0).device,
                count * std::uint64_t(row_bytes)));
            rank.piece.qdata = rank.storage.p;
        }
        rank.piece.payload = rank.piece.qdata;
        rank.plane         = DeviceBuffer(std::size_t(columns) * count * 4);
    }
    exchange_ = std::make_unique<ExpertParallelExchange>(device, input_bytes,
                                                         std::size_t(columns) * widest * 2);
}

DividedHead::~DividedHead() = default;

void DividedHead::project(const void* input, std::size_t input_bytes, int columns,
                          __nv_bfloat16* logits, std::span<const cudaStream_t> streams,
                          const Project& project) {
    if (columns <= 0 || columns > columns_ || input_bytes > input_bytes_ ||
        streams.size() != ranks_.size()) {
        throw std::invalid_argument("a divided head call exceeds its sizes");
    }
    const std::uint64_t vocab = rows_.back();
    const auto part_bytes     = [&](std::size_t r) {
        return std::size_t(columns) * std::size_t(rows_[r + 1] - rows_[r]) * 2;
    };
    const std::size_t slot = exchange_->send(input, input_bytes, streams[0]);
    for (std::size_t r = 1; r < ranks_.size(); ++r) {
        DeviceBinding bind(devices_[r]);
        Rank& rank       = ranks_[r];
        const void* data = exchange_->receive(r, input_bytes, slot, streams[r]);
        project(r, data, rank.piece, columns, static_cast<float*>(rank.plane.p),
                static_cast<__nv_bfloat16*>(exchange_->part(r)), streams[r]);
        exchange_->reply(r, part_bytes(r), slot, streams[r]);
    }
    // Rank 0 projects its rows meanwhile, then places every rank's rows into each column.
    Rank& own = ranks_[0];
    project(0, input, own.piece, columns, static_cast<float*>(own.plane.p),
            static_cast<__nv_bfloat16*>(own.own.p), streams[0]);
    const auto place = [&](std::size_t r, const void* part) {
        const std::size_t row_bytes = std::size_t(rows_[r + 1] - rows_[r]) * 2;
        CUDA_CHECK(cudaMemcpy2DAsync(logits + rows_[r], std::size_t(vocab) * 2, part, row_bytes,
                                     row_bytes, std::size_t(columns), cudaMemcpyDeviceToDevice,
                                     streams[0]));
    };
    place(0, own.own.p);
    std::vector<std::size_t> sizes;
    for (std::size_t r = 1; r < ranks_.size(); ++r) { sizes.push_back(part_bytes(r)); }
    const auto parts = exchange_->collect(sizes, slot, streams[0]);
    for (std::size_t r = 1; r < ranks_.size(); ++r) { place(r, parts[r - 1]); }
}

std::size_t DividedHead::device_bytes(std::size_t rank) const {
    const Rank& r     = ranks_.at(rank);
    std::size_t bytes = r.storage.bytes + r.plane.bytes + r.own.bytes;
    return bytes + exchange_->device_bytes(rank);
}

} // namespace ninfer::models

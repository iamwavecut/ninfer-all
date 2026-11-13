#include "core/device_buffer.h"
#include "core/expert_parallel.h"

#include <stdexcept>

namespace ninfer {

ExpertParallelExchange::ExpertParallelExchange(const DeviceContext& context,
                                               std::size_t input_bytes, std::size_t part_bytes)
    : input_bytes_(input_bytes), part_bytes_(part_bytes) {
    if (context.size() < 2) {
        throw std::invalid_argument("expert parallelism needs at least two ranks");
    }
    if (input_bytes == 0 || part_bytes == 0) {
        throw std::invalid_argument(
            "an expert-parallel exchange carries a positive input and part");
    }
    peers_.resize(context.size());
    for (std::size_t r = 1; r < context.size(); ++r) {
        Peer& peer = peers_[r];
        peer.down  = std::make_unique<StageLink>(
            context, 0, r, StageLinkOptions{.slot_bytes = input_bytes, .slots = 2});
        peer.up = std::make_unique<StageLink>(
            context, r, 0, StageLinkOptions{.slot_bytes = part_bytes, .slots = 2});
        {
            RankBinding bind(context, r);
            peer.input = DeviceBuffer(input_bytes);
            peer.part  = DeviceBuffer(part_bytes);
        }
        RankBinding bind(context, 0);
        peer.received = DeviceBuffer(part_bytes);
    }
}

ExpertParallelExchange::~ExpertParallelExchange() = default;

std::size_t ExpertParallelExchange::send(const void* source, std::size_t bytes,
                                         cudaStream_t stream) {
    if (bytes > input_bytes_) {
        throw std::invalid_argument("expert-parallel input exceeds its slot");
    }
    const std::size_t slot = next_slot_;
    next_slot_             = (next_slot_ + 1) % 2;
    for (std::size_t r = 1; r < peers_.size(); ++r) {
        peers_[r].down->send(source, bytes, slot, stream);
    }
    return slot;
}

void* ExpertParallelExchange::receive(std::size_t rank, std::size_t bytes, std::size_t slot,
                                      cudaStream_t stream) {
    Peer& peer = peers_.at(rank);
    peer.down->recv(peer.input.p, bytes, slot, stream);
    return peer.input.p;
}

void* ExpertParallelExchange::part(std::size_t rank) const { return peers_.at(rank).part.p; }

void ExpertParallelExchange::reply(std::size_t rank, std::size_t bytes, std::size_t slot,
                                   cudaStream_t stream) {
    if (bytes > part_bytes_) {
        throw std::invalid_argument("expert-parallel part exceeds its slot");
    }
    Peer& peer = peers_.at(rank);
    peer.up->send(peer.part.p, bytes, slot, stream);
}

std::vector<void*> ExpertParallelExchange::collect(std::size_t bytes, std::size_t slot,
                                                   cudaStream_t stream) {
    const std::vector<std::size_t> each(peers_.size() - 1, bytes);
    return collect(each, slot, stream);
}

std::vector<void*> ExpertParallelExchange::collect(std::span<const std::size_t> bytes,
                                                   std::size_t slot, cudaStream_t stream) {
    if (bytes.size() + 1 != peers_.size()) {
        throw std::invalid_argument("expert-parallel collect needs a size per peer");
    }
    std::vector<void*> out;
    for (std::size_t r = 1; r < peers_.size(); ++r) {
        if (bytes[r - 1] > part_bytes_) {
            throw std::invalid_argument("expert-parallel part exceeds its slot");
        }
        peers_[r].up->recv(peers_[r].received.p, bytes[r - 1], slot, stream);
        out.push_back(peers_[r].received.p);
    }
    return out;
}

std::size_t ExpertParallelExchange::device_bytes(std::size_t rank) const {
    if (rank == 0) { return (peers_.size() - 1) * part_bytes_; }
    return input_bytes_ + part_bytes_;
}

} // namespace ninfer

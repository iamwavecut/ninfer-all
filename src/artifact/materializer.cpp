#include "artifact/materializer.h"

#include "artifact/framing.h"
#include "artifact/permute_row_split.h"
#include "artifact/transcode.h"
#include "artifact/reader.h"
#include "core/device_wait.h"
#include "core/pinned_memory.h"
#include "core/qtype.h"
#include "core/startup.h"
#include "core/vmm.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <thread>
#include <vector>
#include <string>
#include <tuple>

namespace ninfer::artifact {
namespace {

constexpr std::size_t kSlotBytes        = 64ULL * 1024 * 1024;
constexpr std::size_t kMaximumSlotCount = 4;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw ArtifactError(std::string(operation) + ": " + cudaGetErrorName(status) + ": " +
                            cudaGetErrorString(status));
    }
}

class Slot {
public:
    // cudaMallocHost only guarantees sub-page alignment once other pinned allocations exist in
    // the process (observed on Windows); direct I/O requires page alignment. Over-allocate by one
    // page and align the staging pointer inside the pinned block.
    explicit Slot(std::size_t bytes) : buffer(bytes + kPayloadAlignment) {
        const auto raw     = reinterpret_cast<std::uintptr_t>(buffer.data());
        const auto aligned = (raw + kPayloadAlignment - 1) / kPayloadAlignment * kPayloadAlignment;
        aligned_data       = reinterpret_cast<std::byte*>(aligned);
        check_cuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                   "create weight staging completion event");
    }

    [[nodiscard]] std::byte* data() const noexcept { return aligned_data; }

    ~Slot() {
        if (pending) { (void)ninfer::wait_for_event(event); }
        if (event) { (void)cudaEventDestroy(event); }
    }

    void wait() {
        if (pending) {
            check_cuda(ninfer::wait_for_event(event), "wait for weight staging transfer");
            pending = false;
        }
    }

    // Pinned memory the slot's narrow lines are packed into before their copy, made on first use
    // and free again once the slot's event completes.
    [[nodiscard]] std::byte* packing(std::size_t bytes) {
        if (!packed) { packed.emplace(bytes); }
        return static_cast<std::byte*>(packed->data());
    }

    PinnedHostBuffer buffer;
    std::optional<PinnedHostBuffer> packed;
    std::byte* aligned_data = nullptr;
    cudaEvent_t event       = nullptr;
    bool pending            = false;
};

// Also covers failure between a queued copy and its event record. Destruct before slots.
struct TransferCompletion {
    cudaStream_t stream;
    bool pending = true;

    ~TransferCompletion() {
        if (pending) { (void)ninfer::wait_for_stream(stream); }
    }

    void finish() {
        check_cuda(ninfer::wait_for_stream(stream), "complete weight upload");
        pending = false;
    }
};

struct CopyRange {
    std::size_t file       = 0;
    std::uint64_t begin    = 0;
    std::uint64_t end      = 0;
    std::byte* destination = nullptr;
};

// Binds a pipeline rank for a scope, and does nothing at all for the primary one: rank 0 is
// already current, and leaving the single-device path without even a redundant cudaSetDevice is
// what keeps an offload-free load exactly what it was.
class ScopedRank {
public:
    ScopedRank(DeviceContext& device, std::size_t rank) {
        if (rank != 0) { bound_.emplace(device, rank); }
    }

    ScopedRank(const ScopedRank&)            = delete;
    ScopedRank& operator=(const ScopedRank&) = delete;

private:
    std::optional<ScopedDeviceRank> bound_;
};

struct ReadSpan {
    std::size_t file    = 0;
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

// A range that starts at most this far past a span joins it: reading through the gap costs less
// than another synchronous direct read. A tensor-parallel share of every expert's rows leaves gaps
// of a few hundred KiB, which as separate reads are slow.
constexpr std::uint64_t kSpanGapBytes = 4ULL << 20U;

// Sorts `ranges` by file position and merges them into the aligned file spans that direct reads
// cover; ranges that overlap are a planning error.
std::vector<ReadSpan> direct_spans(std::vector<CopyRange>& ranges) {
    std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
        return std::tie(a.file, a.begin) < std::tie(b.file, b.begin);
    });
    std::vector<ReadSpan> spans;
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        const auto& range = ranges[i];
        if (i && ranges[i - 1].file == range.file && ranges[i - 1].end > range.begin) {
            throw ArtifactError("source ranges overlap");
        }
        const auto begin = range.begin / kPayloadAlignment * kPayloadAlignment;
        if (spans.empty() || spans.back().file != range.file ||
            begin > checked_add(align_up(spans.back().end, kPayloadAlignment, "direct range"),
                                kSpanGapBytes, "direct range")) {
            spans.push_back({range.file, begin, range.end});
        } else {
            spans.back().end = std::max(spans.back().end, range.end);
        }
    }
    return spans;
}

// A range whose stored bytes another range of the same upload pass already reads: copied on the
// device from where that range lands.
struct RepeatedRange {
    const std::byte* source = nullptr;
    std::byte* destination  = nullptr;
    std::uint64_t bytes     = 0;
};

// Sorts `ranges` by file position and takes out each range that lies inside an earlier one, so
// direct reads still read every stored byte once. Several pieces of one stored object on a rank
// need such bytes: every piece of a whole NVFP4 object carries the object's single divisor. Any
// other overlap stays a planning error for direct_spans.
std::vector<RepeatedRange> take_repeated_ranges(std::vector<CopyRange>& ranges) {
    // At one position the longer range comes first, so the shorter one lies inside it.
    std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
        return std::tie(a.file, a.begin, b.end) < std::tie(b.file, b.begin, a.end);
    });
    std::vector<RepeatedRange> repeats;
    std::size_t kept = 0;
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        const CopyRange range = ranges[i];
        // Kept ranges are disjoint and ordered, so only the last one can hold this one.
        if (kept != 0) {
            const CopyRange& cover = ranges[kept - 1];
            if (cover.file == range.file && cover.begin <= range.begin && range.end <= cover.end) {
                repeats.push_back({cover.destination + (range.begin - cover.begin),
                                   range.destination, range.end - range.begin});
                continue;
            }
        }
        ranges[kept++] = range;
    }
    ranges.resize(kept);
    return repeats;
}

// Host-kept objects are read in pieces of this size on this many threads: enough reads in flight
// to keep an NVMe drive busy, and the copies out of staging keep pace on the same threads.
constexpr std::uint64_t kHostPieceBytes = 16ULL << 20U;
constexpr std::size_t kHostReadThreads  = 8;

// An object's stored bytes and where the load keeps them on the host.
struct HostReadTarget {
    std::uint64_t offset   = 0; // logical artifact offset
    std::uint64_t bytes    = 0;
    std::byte* destination = nullptr;
};

struct AlignedDelete {
    void operator()(std::byte* data) const noexcept {
        ::operator delete(data, std::align_val_t{kPayloadAlignment});
    }
};

// Writes every target's stored bytes to its destination. Direct reads skip the page cache, which
// would otherwise hold a second copy of everything the host keeps, and run on several threads:
// each takes the next piece of the merged spans into its own aligned staging buffer and copies the
// targets' bytes out of it. Progress counts from 0 toward `total`. Returns the bytes read.
std::uint64_t read_host_bytes(const Reader& reader, std::span<const HostReadTarget> targets,
                              StartupPhaseScope& phase, std::uint64_t total) {
    std::vector<CopyRange> ranges;
    std::uint64_t expected = 0;
    for (const auto& target : targets) {
        for (const auto& segment : reader.segments(target.offset, target.bytes)) {
            ranges.push_back({segment.file_index, segment.file_offset,
                              checked_add(segment.file_offset, segment.bytes, "host range"),
                              target.destination + segment.destination_offset});
        }
        expected = checked_add(expected, target.bytes, "host bytes");
    }
    struct Piece {
        std::size_t file        = 0;
        std::uint64_t begin     = 0;
        std::uint64_t end       = 0;
        std::size_t first_range = 0; // the first range that ends past `begin`
    };
    std::vector<Piece> pieces;
    std::size_t first = 0;
    for (const auto& span : direct_spans(ranges)) {
        for (auto source = span.begin; source < span.end; source += kHostPieceBytes) {
            while (first < ranges.size() &&
                   std::tie(ranges[first].file, ranges[first].end) <=
                       std::tie(span.file, source)) {
                ++first;
            }
            pieces.push_back({span.file, source, std::min(span.end, source + kHostPieceBytes),
                              first});
        }
    }
    std::atomic<std::size_t> next{0};
    std::atomic<std::uint64_t> copied{0}, read{0};
    std::atomic<bool> stop{false};
    std::mutex mutex;
    std::condition_variable finished_cv;
    std::size_t finished = 0;
    std::exception_ptr error;
    const auto work = [&] {
        try {
            const std::unique_ptr<std::byte, AlignedDelete> staging(static_cast<std::byte*>(
                ::operator new(kHostPieceBytes, std::align_val_t{kPayloadAlignment})));
            for (std::size_t index; !stop && (index = next++) < pieces.size();) {
                const Piece& piece   = pieces[index];
                const auto needed    = piece.end - piece.begin;
                const auto request   = static_cast<std::size_t>(
                    align_up(needed, kPayloadAlignment, "host piece bytes"));
                const auto received  = reader.read_direct(piece.file, piece.begin,
                                                          {staging.get(), request});
                if (received < needed) {
                    throw ArtifactError("direct read ended before the host payload");
                }
                read += received;
                for (auto r = piece.first_range; r < ranges.size() &&
                                                 ranges[r].file == piece.file &&
                                                 ranges[r].begin < piece.end;
                     ++r) {
                    const auto& range = ranges[r];
                    const auto begin  = std::max(piece.begin, range.begin);
                    const auto end    = std::min(piece.end, range.end);
                    if (begin >= end) { continue; }
                    std::memcpy(range.destination + (begin - range.begin),
                                staging.get() + (begin - piece.begin),
                                static_cast<std::size_t>(end - begin));
                    copied += end - begin;
                }
            }
        } catch (...) {
            std::lock_guard lock(mutex);
            if (!error) { error = std::current_exception(); }
            stop = true;
        }
        std::lock_guard lock(mutex);
        ++finished;
        finished_cv.notify_one();
    };
    std::vector<std::thread> threads;
    // Stops and joins the workers on every exit, a failure to start one included.
    struct Join {
        std::vector<std::thread>& threads;
        std::atomic<bool>& stop;
        ~Join() {
            stop = true;
            for (auto& thread : threads) { thread.join(); }
        }
    } join{threads, stop};
    const auto count = std::min<std::size_t>(kHostReadThreads, pieces.size());
    for (std::size_t i = 0; i < count; ++i) { threads.emplace_back(work); }
    {
        std::unique_lock lock(mutex);
        while (!finished_cv.wait_for(lock, std::chrono::milliseconds(200),
                                     [&] { return finished == count; })) {
            lock.unlock();
            phase.progress(copied, total);
            lock.lock();
        }
        if (error) { std::rethrow_exception(error); }
    }
    if (copied != expected) { throw ArtifactError("incomplete Host read"); }
    phase.progress(copied, total);
    return read;
}

// The NVFP4 divisor of the stored object's row `first_row`, from its stored bytes (`host`) or its
// file. A selected piece names the row it starts at: the piece's own geometry places its divisors
// inside the piece, not inside the stored object.
float read_divisor(const Reader& reader, ObjectHandle handle, const WeightGeometry& stored,
                   std::span<const std::byte> host, MaterializationStats& stats,
                   std::uint64_t first_row = 0) {
    if (stored.format != QType::NVFP4) { return 0.0F; }
    const auto bytes = checked_mul(stored.divisor_count, sizeof(float), "weight divisors");
    std::vector<std::byte> words(bytes);
    if (!host.empty()) {
        std::copy_n(host.data() + stored.divisor_offset, bytes, words.data());
    } else {
        const auto& object = reader.directory().tensor(handle);
        reader.read_into(checked_add(object.offset, stored.divisor_offset, "weight divisor"),
                         words);
        stats.read_bytes = checked_add(stats.read_bytes, bytes, "read bytes");
    }
    // Every divisor is read by a kernel, so every one is checked before anything binds.
    for (std::uint64_t index = 0; index < stored.divisor_count; ++index) {
        const auto value = std::bit_cast<float>(read_u32_le(words.data() + index * sizeof(float)));
        if (!std::isfinite(value) || value <= 0) {
            throw ArtifactError(reader.directory().tensor(handle).id +
                                ": invalid NVFP4 weight divisor");
        }
    }
    const std::uint64_t rows = stored.shape.size() == 2 ? stored.shape[0] : 1;
    const std::uint64_t word = first_row / (rows / stored.divisor_count);
    return std::bit_cast<float>(read_u32_le(words.data() + word * sizeof(float)));
}

// Copies of one width whose sources and destinations advance by fixed pitches, as the lines of a
// column cut do (a tensor-parallel share of every expert's columns is millions of lines of a few
// to a few hundred bytes), leave together instead of one call per line: lines narrower than
// kPackedLineBytes that land back to back are packed on the host into the slot's packing memory
// and copied as one block (the copy engine moves such narrow 2D rows at a fraction of the bus
// rate), others as one 2D copy.
class LineBatch {
public:
    static constexpr std::size_t kPackedLineBytes = 4096;

    // The staging slot whose bytes the next copies read, `bytes` large.
    void begin(Slot& slot, std::size_t bytes) {
        slot_       = &slot;
        slot_bytes_ = bytes;
        packed_     = 0;
    }

    // Adds a copy from the staging slot; false when it does not continue the batch.
    bool extend(std::byte* destination, const std::byte* source, std::size_t bytes) {
        if (height_ == 0) {
            destination_ = destination;
            source_      = source;
            width_       = bytes;
            height_      = 1;
            return true;
        }
        if (bytes != width_ || destination <= destination_ || source <= source_) { return false; }
        if (height_ == 1) {
            const auto destination_pitch = static_cast<std::size_t>(destination - destination_);
            const auto source_pitch      = static_cast<std::size_t>(source - source_);
            if (destination_pitch < width_ || source_pitch < width_) { return false; }
            destination_pitch_ = destination_pitch;
            source_pitch_      = source_pitch;
        } else if (destination != destination_ + height_ * destination_pitch_ ||
                   source != source_ + height_ * source_pitch_) {
            return false;
        }
        ++height_;
        return true;
    }

    void flush(cudaStream_t stream) {
        if (height_ == 1) {
            check_cuda(cudaMemcpyAsync(destination_, source_, width_, cudaMemcpyHostToDevice, stream),
                       "upload weight bytes");
        } else if (height_ > 1 && destination_pitch_ == width_ && width_ < kPackedLineBytes) {
            // The packed lines of one slot never exceed the bytes it read.
            std::byte* const packed = slot_->packing(slot_bytes_) + packed_;
            for (std::size_t line = 0; line < height_; ++line) {
                std::memcpy(packed + line * width_, source_ + line * source_pitch_, width_);
            }
            packed_ += height_ * width_;
            check_cuda(cudaMemcpyAsync(destination_, packed, height_ * width_,
                                       cudaMemcpyHostToDevice, stream),
                       "upload packed weight lines");
        } else if (height_ > 1) {
            check_cuda(cudaMemcpy2DAsync(destination_, destination_pitch_, source_, source_pitch_,
                                         width_, height_, cudaMemcpyHostToDevice, stream),
                       "upload weight lines");
        }
        height_ = 0;
    }

private:
    Slot* slot_             = nullptr;
    std::size_t slot_bytes_ = 0;
    std::size_t packed_     = 0;
    std::byte* destination_        = nullptr;
    const std::byte* source_       = nullptr;
    std::size_t width_             = 0;
    std::size_t height_            = 0;
    std::size_t destination_pitch_ = 0;
    std::size_t source_pitch_      = 0;
};

struct UploadTotals {
    std::uint64_t read_bytes         = 0;
    std::uint64_t h2d_bytes          = 0;
    std::uint64_t peak_staging_bytes = 0;
    double seconds                   = 0;
};

// Writes every device object's bytes to `bases[placement.rank] + placement.offset`: transcoded
// objects whole through the host, the rest through direct-I/O staging, one pass per rank. The
// phase's progress counts `done` bytes the load already read before these `total`.
UploadTotals upload_device_bytes(const Reader& reader, std::span<const DevicePlacement> placements,
                                 std::span<std::byte* const> bases, DeviceContext& device,
                                 const StartupObserver& observer, StartupPhaseScope& phase,
                                 std::uint64_t done, std::uint64_t total) {
    UploadTotals totals;
    const std::size_t rank_count = bases.size();
    std::vector<std::vector<CopyRange>> ranges_by_rank(rank_count);
    std::vector<const DevicePlacement*> transcoded;
    std::vector<const DevicePlacement*> rebased;
    // Bytes of selected pieces that no run fills: the alignment between a small piece's planes,
    // zeroed instead of uploaded.
    std::uint64_t zeroed = 0;
    for (const auto& placement : placements) {
        if (placement.rank >= rank_count || bases[placement.rank] == nullptr) {
            throw ArtifactError("device placement names a rank the plan has no arena for");
        }
        std::byte* const destination = bases[placement.rank] + placement.offset;
        if (placement.tail != 0) {
            const ScopedRank bound(device, placement.rank);
            check_cuda(ninfer::fill_and_wait(destination + placement.bytes, 0, placement.tail),
                       "zero a device tail");
        }
        if (placement.transcode) {
            transcoded.push_back(&placement);
            continue;
        }
        if (placement.selection && placement.selection->rebase != 0) {
            rebased.push_back(&placement);
            continue;
        }
        if (placement.selection) {
            const auto& descriptor = reader.directory().tensor(*placement.source);
            const auto runs = selected_runs(reader.geometry(*placement.source), *placement.selection);
            std::uint64_t covered = 0;
            for (const SelectedRun& run : runs) {
                covered = checked_add(covered, run.bytes, "selected bytes");
            }
            if (covered < placement.bytes) {
                const ScopedRank bound(device, placement.rank);
                check_cuda(ninfer::fill_and_wait(destination, 0, placement.bytes),
                           "zero a selected piece");
                zeroed = checked_add(zeroed, placement.bytes - covered, "zeroed bytes");
            }
            for (const SelectedRun& run : runs) {
                for (const auto& segment : reader.segments(
                         checked_add(descriptor.offset, run.source, "selected run"), run.bytes)) {
                    ranges_by_rank[placement.rank].push_back(
                        {segment.file_index, segment.file_offset,
                         checked_add(segment.file_offset, segment.bytes, "copy range"),
                         destination + run.destination + segment.destination_offset});
                }
            }
            continue;
        }
        const auto& descriptor = reader.directory().tensor(placement.object);
        for (const auto& segment : reader.segments(descriptor.offset, descriptor.bytes)) {
            ranges_by_rank[placement.rank].push_back(
                {segment.file_index, segment.file_offset,
                 checked_add(segment.file_offset, segment.bytes, "copy range"),
                 destination + segment.destination_offset});
        }
    }
    // Transcoded objects are read whole, requantized on the host and uploaded synchronously.
    // They are the vocabulary-sized exceptions a startup option asked for, so bounded extra Host
    // memory (one object at a time) is acceptable and the direct-I/O pipeline stays unchanged.
    std::uint64_t transcoded_bytes = 0;
    for (const DevicePlacement* placement : transcoded) {
        const auto source = reader.read_object(placement->object);
        totals.read_bytes = checked_add(totals.read_bytes, source.size(), "transcode read bytes");
        std::vector<std::byte> encoded(static_cast<std::size_t>(placement->bytes));
        transcode_row_split(*placement->transcode,
                            reader.directory().tensor(placement->object).shape, source, encoded);
        const ScopedRank bound(device, placement->rank);
        check_cuda(ninfer::copy_and_wait(bases[placement->rank] + placement->offset, encoded.data(),
                                         encoded.size(), cudaMemcpyHostToDevice),
                   "upload transcoded weight bytes");
        transcoded_bytes = checked_add(transcoded_bytes, encoded.size(), "transcoded bytes");
        phase.progress(done + transcoded_bytes, done + total);
    }
    // Rebased index pieces are a few KiB each: gathered and shifted on the host.
    for (const DevicePlacement* placement : rebased) {
        const auto source = reader.read_object(*placement->source);
        totals.read_bytes = checked_add(totals.read_bytes, source.size(), "rebased read bytes");
        std::vector<std::byte> bytes(static_cast<std::size_t>(placement->bytes));
        for (const SelectedRun& run :
             selected_runs(reader.geometry(*placement->source), *placement->selection)) {
            std::memcpy(bytes.data() + run.destination, source.data() + run.source, run.bytes);
        }
        for (std::size_t at = 0; at + sizeof(std::int32_t) <= bytes.size(); at += sizeof(std::int32_t)) {
            std::int32_t value;
            std::memcpy(&value, bytes.data() + at, sizeof(value));
            value -= placement->selection->rebase;
            std::memcpy(bytes.data() + at, &value, sizeof(value));
        }
        const ScopedRank bound(device, placement->rank);
        check_cuda(ninfer::copy_and_wait(bases[placement->rank] + placement->offset, bytes.data(),
                                         bytes.size(), cudaMemcpyHostToDevice),
                   "upload rebased index bytes");
        transcoded_bytes = checked_add(transcoded_bytes, bytes.size(), "rebased bytes");
    }
    totals.h2d_bytes     = transcoded_bytes;
    std::uint64_t copied = zeroed;
    const auto start     = std::chrono::steady_clock::now();
    bool uploaded_any    = false;
    // One staging pass per rank, each on its own device: a host-to-device copy and the event that
    // retires its slot both belong to the destination device, so they cannot be shared across
    // ranks. The ranks hold disjoint objects, so no payload is read twice. With one rank this is
    // the original single pass, on the original stream, with the original slot pool.
    for (std::size_t rank = 0; rank < rank_count; ++rank) {
        auto& ranges = ranges_by_rank[rank];
        if (ranges.empty()) { continue; }
        uploaded_any = true;
        const ScopedRank bound(device, rank);
        const cudaStream_t transfer_stream       = device.transfer_stream_for_rank(rank);
        const std::vector<RepeatedRange> repeats = take_repeated_ranges(ranges);
        const std::vector<ReadSpan> spans        = direct_spans(ranges);
        std::uint64_t aligned_bytes              = 0;
        for (const auto& span : spans) {
            aligned_bytes =
                checked_add(aligned_bytes,
                            align_up(span.end - span.begin, kPayloadAlignment, "direct range bytes"),
                            "direct bytes");
        }
        const auto slot_bytes =
            static_cast<std::size_t>(std::min<std::uint64_t>(kSlotBytes, aligned_bytes));
        const auto slot_count = static_cast<std::size_t>(
            std::min<std::uint64_t>(kMaximumSlotCount, 1 + (aligned_bytes - 1) / slot_bytes));
        std::vector<std::unique_ptr<Slot>> slots;
        // Passes run one after another and free their slots in between, so the process high-water
        // mark is the largest pass, not their sum.
        totals.peak_staging_bytes =
            std::max<std::uint64_t>(totals.peak_staging_bytes, slot_bytes * slot_count);
        StartupPhaseScope pin_phase(observer, StartupPhase::WeightsStagingPin,
                                    StartupProgressUnit::Bytes, slot_bytes * slot_count);
        for (std::size_t i = 0; i < slot_count; ++i) {
            slots.push_back(std::make_unique<Slot>(slot_bytes));
        }
        pin_phase.complete();
        TransferCompletion completion{transfer_stream};
        std::size_t next_slot  = 0;
        std::size_t next_range = 0;
        LineBatch lines;
        for (const auto& span : spans) {
            for (auto source = span.begin; source < span.end; source += slot_bytes) {
                auto& slot = *slots[next_slot++ % slot_count];
                slot.wait();
                lines.begin(slot, slot_bytes);
                const auto remaining = span.end - source;
                const auto request   = static_cast<std::size_t>(std::min<std::uint64_t>(
                    slot_bytes, align_up(remaining, kPayloadAlignment, "direct block bytes")));
                const auto received  = reader.read_direct(
                    span.file, source, {slot.data(), request});
                if (received < std::min<std::uint64_t>(request, remaining)) {
                    throw ArtifactError("direct read ended before the required payload");
                }
                totals.read_bytes    = checked_add(totals.read_bytes, received, "read bytes");
                const auto chunk_end = checked_add(source, received, "read block end");
                while (next_range < ranges.size() && ranges[next_range].file == span.file &&
                       ranges[next_range].begin < chunk_end) {
                    const auto& range = ranges[next_range];
                    const auto begin  = std::max(source, range.begin);
                    const auto end    = std::min(chunk_end, range.end);
                    if (begin < end) {
                        std::byte* const to     = range.destination + (begin - range.begin);
                        const std::byte* const from = slot.data() + (begin - source);
                        const auto bytes        = static_cast<std::size_t>(end - begin);
                        if (!lines.extend(to, from, bytes)) {
                            lines.flush(transfer_stream);
                            (void)lines.extend(to, from, bytes);
                        }
                        copied = checked_add(copied, end - begin, "copied bytes");
                    }
                    if (range.end > chunk_end) { break; }
                    ++next_range;
                }
                lines.flush(transfer_stream);
                check_cuda(cudaEventRecord(slot.event, transfer_stream),
                           "record weight staging completion");
                slot.pending = true;
                phase.progress(done + checked_add(copied, transcoded_bytes, "uploaded bytes"),
                               done + total);
            }
        }
        for (const auto& slot : slots) {
            slot->wait();
            if (slot->packed) {
                totals.peak_staging_bytes =
                    std::max<std::uint64_t>(totals.peak_staging_bytes, slot_bytes * 2 * slot_count);
            }
        }
        // Behind the uploads on the same stream, so each source already holds its bytes.
        for (const RepeatedRange& repeat : repeats) {
            check_cuda(cudaMemcpyAsync(repeat.destination, repeat.source,
                                       static_cast<std::size_t>(repeat.bytes),
                                       cudaMemcpyDeviceToDevice, transfer_stream),
                       "repeat stored weight bytes");
            copied = checked_add(copied, repeat.bytes, "copied bytes");
        }
        completion.finish();
        if (next_range != ranges.size()) { throw ArtifactError("incomplete device upload"); }
    }
    if (!uploaded_any) {
        phase.complete(done + transcoded_bytes, done + total);
        return totals;
    }
    if (checked_add(copied, transcoded_bytes, "uploaded bytes") != total) {
        throw ArtifactError("incomplete device upload");
    }
    totals.h2d_bytes = checked_add(copied - zeroed, transcoded_bytes, "uploaded bytes");
    totals.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    phase.complete(done + total, done + total);
    return totals;
}

// The geometry a device placement's bytes take: the selected piece's, the transcode target's, or
// the stored one.
WeightGeometry placement_geometry(const Reader& reader, const DevicePlacement& placement) {
    if (placement.selection) {
        return selected_geometry(reader.geometry(*placement.source), *placement.selection);
    }
    return placement.transcode ? weight_geometry(*placement.transcode, QuantLayout::RowSplit,
                                                 reader.directory().tensor(placement.object).shape)
                               : reader.geometry(placement.object);
}

// upload_device_bytes, then the panel permutation of the placements that ask for it. Loading and a
// resume after a model suspend upload through here alike.
UploadTotals upload_device_objects(const Reader& reader,
                                   std::span<const DevicePlacement> placements,
                                   std::span<std::byte* const> bases, DeviceContext& device,
                                   const StartupObserver& observer, StartupPhaseScope& phase,
                                   std::uint64_t done, std::uint64_t total) {
    const UploadTotals totals =
        upload_device_bytes(reader, placements, bases, device, observer, phase, done, total);
    std::vector<bool> permuted_rank(bases.size(), false);
    for (const auto& placement : placements) {
        if (!placement.panel) { continue; }
        const ScopedRank bound(device, placement.rank);
        permute_row_split_to_panel(placement_geometry(reader, placement),
                                   bases[placement.rank] + placement.offset, nullptr);
        permuted_rank[placement.rank] = true;
    }
    for (std::size_t rank = 0; rank < permuted_rank.size(); ++rank) {
        if (!permuted_rank[rank]) { continue; }
        const ScopedRank bound(device, rank);
        check_cuda(ninfer::wait_for_device(), "finish the panel permutation");
    }
    return totals;
}

} // namespace

const WeightParent& MaterializedArtifact::device_parent(ObjectHandle handle) const {
    if (!has_device(handle)) { throw ArtifactError("object has no device weight backing"); }
    return *objects_[handle.index].device;
}

const WeightParent& MaterializedArtifact::host_parent(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || !objects_[handle.index].host) {
        throw ArtifactError("object has no Host weight backing");
    }
    return *objects_[handle.index].host;
}

std::span<const std::byte> MaterializedArtifact::host_bytes(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].host_data.empty()) {
        throw ArtifactError("object has no retained Host bytes");
    }
    return objects_[handle.index].host_data;
}

bool MaterializedArtifact::has_device(ObjectHandle handle) const noexcept {
    return handle.index < objects_.size() && objects_[handle.index].device.has_value();
}

const WeightParent& MaterializedArtifact::pinned_parent(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || !objects_[handle.index].pinned) {
        throw ArtifactError("object has no pinned Host weight backing");
    }
    return *objects_[handle.index].pinned;
}

const WeightParent& MaterializedArtifact::mapped_parent(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || !objects_[handle.index].mapped) {
        throw ArtifactError("object has no mapped Host weight backing");
    }
    return *objects_[handle.index].mapped;
}

std::span<const std::byte> MaterializedArtifact::pinned_block() const noexcept {
    if (!pinned_) { return {}; }
    return {static_cast<const std::byte*>(pinned_->data()), pinned_->size()};
}

MaterializedArtifact materialize(const Reader& reader, MaterializationPlan&& plan,
                                 DeviceContext& device, const StartupObserver* startup_observer,
                                 std::unique_ptr<EvictableWeightPool> backing,
                                 const MaterializeOptions& options) {
    if (plan.source != &reader || plan.object_count != reader.directory().objects.size()) {
        throw ArtifactError("materialization plan belongs to another load session");
    }
    const StartupObserver no_observer;
    const auto& observer = startup_observer ? *startup_observer : no_observer;
    std::uint64_t total  = 0;
    for (const auto& placement : plan.device_objects) {
        total = checked_add(total, placement.bytes, "device payload bytes");
    }
    // The objects kept on the host (the pinned block's, and Host objects nobody has read yet) are
    // read first and count toward the same progress.
    std::uint64_t host_total = 0;
    for (const auto& placement : plan.pinned_objects) {
        host_total = checked_add(host_total, placement.bytes, "pinned payload bytes");
    }
    for (const auto& placement : plan.mapped_objects) {
        host_total = checked_add(host_total, placement.bytes, "mapped payload bytes");
    }
    for (const auto& placement : plan.host_objects) {
        if (placement.data.empty()) {
            host_total = checked_add(host_total,
                                     object_bytes(reader.directory().object(placement.object)),
                                     "Host payload bytes");
        }
    }
    StartupPhaseScope phase(observer, StartupPhase::WeightsMaterialize, StartupProgressUnit::Bytes,
                            checked_add(host_total, total, "weight payload bytes"));
    MaterializedArtifact out;
    out.objects_.resize(plan.object_count + plan.selected_count);
    out.stats_.file_bytes            = reader.file_bytes();
    out.stats_.read_bytes            = plan.prior_read_bytes;
    out.stats_.owned_value_bytes     = plan.owned_value_bytes;
    const std::size_t rank_count     = plan.device_rank_count();
    out.stats_.device_capacity_bytes = plan.device_capacity(0);
    out.stats_.device_object_count   = plan.device_objects.size();
    out.stats_.host_object_count     = plan.host_objects.size();
    if (backing && rank_count > 1) {
        // The pool's window is borrowed by a Vision encode that runs on the primary device, and
        // nothing on an offloaded rank is lendable. Combining them is a planning error, not a
        // configuration to silently half-apply.
        throw ArtifactError("an eviction pool backs only a single-device materialization");
    }
    out.arenas_.resize(rank_count);
    if (options.suspendable) { out.regions_.resize(rank_count); }
    for (std::size_t rank = 0; rank < rank_count; ++rank) {
        const auto capacity = plan.device_capacity(rank);
        if (capacity > std::numeric_limits<std::size_t>::max()) {
            throw ArtifactError("device backing exceeds size_t");
        }
        if (rank == 0 && backing) {
            if (backing->arena().bytes < capacity) {
                throw ArtifactError("eviction pool arena is smaller than the materialization plan");
            }
            out.pool_      = std::move(backing);
            out.arenas_[0] = std::make_unique<DeviceArena>(out.pool_->arena());
        } else if (capacity && options.suspendable) {
            // Fixed addresses whose memory a model suspend gives back; the weights are uploaded
            // again from the retained artifact when it resumes.
            const ScopedRank bound(device, rank);
            out.regions_[rank] = VmmRegion(
                static_cast<std::size_t>(capacity),
                VmmRegion::Options{.device = device.rank(rank).device, .peers = {}});
            out.arenas_[rank] = std::make_unique<DeviceArena>(out.regions_[rank].span());
        } else if (capacity) {
            // Each rank's arena must live in that rank's memory, so it is allocated while that
            // device is current.
            const ScopedRank bound(device, rank);
            out.arenas_[rank] = std::make_unique<DeviceArena>(static_cast<std::size_t>(capacity));
        }
        // Plan offsets are authoritative: an evictable tail starts at an aligned boundary that a
        // bump allocator would not reproduce. The arena accounts the whole planned extent.
        if (out.arenas_[rank] && capacity) {
            (void)out.arenas_[rank]->alloc_bytes(static_cast<std::size_t>(capacity), 1);
        }
        out.stats_.device_capacity_by_rank.push_back(capacity);
    }
    std::vector<HostReadTarget> host_reads;
    // The pinned and the mapped block are laid out alike: ascending, aligned, non-overlapping
    // placements of whole stored objects, read straight into the block.
    const auto host_block = [&](std::span<const PinnedPlacement> placements,
                                std::uint64_t capacity, const char* label) {
        if (capacity > std::numeric_limits<std::size_t>::max()) {
            throw ArtifactError(std::string(label) + " Host backing exceeds size_t");
        }
        auto pages = std::make_unique<HostPages>(static_cast<std::size_t>(capacity));
        std::uint64_t previous_end = 0;
        std::vector<bool> placed(out.objects_.size(), false);
        for (const auto& placement : placements) {
            reader.validate_object(placement.object);
            const auto& geometry = reader.geometry(placement.object);
            if (placed.at(placement.object.index) || geometry.bytes != placement.bytes ||
                placement.offset < previous_end || placement.alignment == 0 ||
                placement.offset % placement.alignment != 0 || placement.offset > capacity ||
                placement.bytes > capacity - placement.offset) {
                throw ArtifactError(std::string("invalid or duplicate ") + label +
                                    " Host placement");
            }
            placed[placement.object.index] = true;
            previous_end                   = placement.offset + placement.bytes;
            host_reads.push_back({reader.directory().tensor(placement.object).offset,
                                  placement.bytes, pages->data() + placement.offset});
        }
        return pages;
    };
    if (!plan.pinned_objects.empty()) {
        out.pinned_ = host_block(plan.pinned_objects, plan.pinned_capacity_bytes, "pinned");
    }
    if (!plan.mapped_objects.empty()) {
        out.mapped_ = host_block(plan.mapped_objects, plan.mapped_capacity_bytes, "mapped");
    }
    for (auto& placement : plan.host_objects) {
        reader.validate_object(placement.object);
        if (!placement.data.empty()) { continue; }
        const auto& object = reader.directory().object(placement.object);
        const auto bytes   = object_bytes(object);
        if (bytes > std::numeric_limits<std::size_t>::max()) {
            throw ArtifactError("Host object exceeds size_t");
        }
        placement.data.resize(static_cast<std::size_t>(bytes));
        host_reads.push_back({object_offset(object), bytes, placement.data.data()});
    }
    const auto host_read =
        read_host_bytes(reader, host_reads, phase,
                        checked_add(host_total, total, "weight payload bytes"));
    out.stats_.read_bytes = checked_add(out.stats_.read_bytes, host_read, "Host read bytes");
    if (out.mapped_) {
        out.mapped_->pin(true);
        // The address the primary device reads the block at. Under unified addressing it is
        // usually the host address, but only the runtime may say so.
        void* alias = nullptr;
        check_cuda(cudaHostGetDevicePointer(&alias, out.mapped_->data(), 0),
                   "resolve the mapped Host block's device address");
        for (const auto& placement : plan.mapped_objects) {
            const auto& geometry = reader.geometry(placement.object);
            const std::span<std::byte> destination(out.mapped_->data() + placement.offset,
                                                   static_cast<std::size_t>(placement.bytes));
            const auto divisor =
                read_divisor(reader, placement.object, geometry, destination, out.stats_);
            out.objects_.at(placement.object.index).mapped =
                WeightParent{geometry, static_cast<const std::byte*>(alias) + placement.offset,
                             divisor};
        }
        out.stats_.mapped_bytes        = plan.mapped_capacity_bytes;
        out.stats_.mapped_object_count = plan.mapped_objects.size();
        out.stats_.pinned_bytes =
            checked_add(out.stats_.pinned_bytes, plan.mapped_capacity_bytes, "mapped Host bytes");
    }
    if (out.pinned_) {
        out.pinned_->pin();
        for (const auto& placement : plan.pinned_objects) {
            const auto& geometry = reader.geometry(placement.object);
            const std::span<std::byte> destination(out.pinned_->data() + placement.offset,
                                                   static_cast<std::size_t>(placement.bytes));
            const auto divisor =
                read_divisor(reader, placement.object, geometry, destination, out.stats_);
            out.objects_.at(placement.object.index).pinned =
                WeightParent{geometry, destination.data(), divisor};
        }
        out.stats_.pinned_bytes = checked_add(out.stats_.pinned_bytes,
                                              plan.pinned_capacity_bytes, "pinned Host bytes");
        out.stats_.pinned_object_count = plan.pinned_objects.size();
    }
    for (auto& placement : plan.host_objects) {
        auto& storage = out.objects_.at(placement.object.index);
        if (!storage.host_data.empty()) { throw ArtifactError("duplicate Host placement"); }
        const auto& object = reader.directory().object(placement.object);
        if (placement.data.size() != object_bytes(object)) {
            throw ArtifactError("Host placement size differs from object");
        }
        storage.host_data = std::move(placement.data);
        if (placement.registered) {
            storage.registration = std::make_unique<HostMemoryRegistration>(
                storage.host_data.data(), storage.host_data.size());
            out.stats_.pinned_bytes = checked_add(out.stats_.pinned_bytes, storage.host_data.size(),
                                                "registered Host bytes");
            ++out.stats_.pinned_object_count;
        }
        out.stats_.retained_host_bytes =
            checked_add(out.stats_.retained_host_bytes, storage.host_data.size(), "retained bytes");
        if (std::holds_alternative<TensorObject>(object)) {
            const auto& geometry = reader.geometry(placement.object);
            const auto divisor =
                read_divisor(reader, placement.object, geometry, storage.host_data, out.stats_);
            storage.host = WeightParent{geometry, storage.host_data.data(), divisor};
        }
    }
    std::vector<std::uint64_t> previous_device_end(rank_count, 0);
    for (const auto& placement : plan.device_objects) {
        if (placement.selection.has_value() != placement.source.has_value() ||
            (placement.selection &&
             (placement.object.index < plan.object_count || placement.transcode))) {
            throw ArtifactError("a selected device placement is malformed");
        }
        const ObjectHandle stored   = placement.source.value_or(placement.object);
        const auto& stored_geometry = reader.geometry(stored);
        auto geometry               = placement_geometry(reader, placement);
        if (placement.panel) {
            if (!row_split_panel_supported(geometry)) {
                throw ArtifactError("a panel placement names an object without the layout");
            }
            geometry.layout = QuantLayout::RowSplitPanel;
        }
        auto& object = out.objects_.at(placement.object.index);
        if (placement.rank >= rank_count) {
            throw ArtifactError("device placement names a rank the plan has no arena for");
        }
        const auto& arena    = out.arenas_[placement.rank];
        const auto rank_capacity = plan.device_capacity(placement.rank);
        if (object.device || !arena || geometry.bytes != placement.bytes ||
            (placement.transcode && object.host)) {
            throw ArtifactError("invalid or duplicate device placement");
        }
        if (placement.offset < previous_device_end[placement.rank] || placement.alignment == 0 ||
            placement.offset % placement.alignment != 0 || placement.offset > rank_capacity ||
            placement.bytes > rank_capacity - placement.offset ||
            placement.tail > rank_capacity - placement.offset - placement.bytes) {
            throw ArtifactError("device offset differs from materialization plan");
        }
        previous_device_end[placement.rank] = placement.offset + placement.bytes + placement.tail;
        const DeviceSpan storage{static_cast<std::byte*>(arena->base()) + placement.offset,
                                 static_cast<std::size_t>(placement.bytes)};
        // A transcoded placement is NVFP4 no more; a selected piece starts at its first stored row.
        float divisor = object.host ? object.host->weight_scale_divisor : 0.0F;
        if (!object.host && geometry.format == QType::NVFP4) {
            divisor = read_divisor(reader, stored, stored_geometry, {}, out.stats_,
                                   placement.selection ? placement.selection->rows.front().first : 0);
        }
        object.device = WeightParent{geometry, static_cast<const std::byte*>(storage.data),
                                     divisor, stored_geometry.format};
    }
    const UploadTotals upload = upload_device_objects(reader, plan.device_objects,
                                                      out.arena_bases(), device, observer, phase,
                                                      host_total, total);
    out.stats_.read_bytes = checked_add(out.stats_.read_bytes, upload.read_bytes, "read bytes");
    out.stats_.h2d_bytes  = upload.h2d_bytes;
    out.stats_.peak_staging_bytes =
        std::max<std::uint64_t>(out.stats_.peak_staging_bytes, upload.peak_staging_bytes);
    out.stats_.upload_seconds = upload.seconds;
    if (options.suspendable) {
        if (!options.retained_reader || options.retained_reader.get() != &reader) {
            throw ArtifactError("a suspendable materialization must retain its own reader");
        }
        out.reader_            = options.retained_reader;
        out.device_placements_ = std::move(plan.device_objects);
        out.device_payload_    = total;
    }
    return out;
}

std::vector<std::byte*> MaterializedArtifact::arena_bases() const {
    std::vector<std::byte*> bases(arenas_.size(), nullptr);
    for (std::size_t rank = 0; rank < arenas_.size(); ++rank) {
        if (arenas_[rank]) { bases[rank] = static_cast<std::byte*>(arenas_[rank]->base()); }
    }
    return bases;
}

bool MaterializedArtifact::suspendable() const noexcept { return reader_ != nullptr; }

bool MaterializedArtifact::device_backed() const noexcept {
    if (pool_ && !pool_->backed()) { return false; }
    for (const VmmRegion& region : regions_) {
        if (region && !region.backed()) { return false; }
    }
    return true;
}

std::uint64_t MaterializedArtifact::device_backing_bytes() const noexcept {
    std::uint64_t bytes = pool_ ? pool_->arena().bytes : 0;
    for (const VmmRegion& region : regions_) {
        if (region) { bytes += region.reserved_bytes(); }
    }
    return bytes;
}

void MaterializedArtifact::release_device_backing() {
    if (!suspendable()) {
        throw std::logic_error("this materialization was not made suspendable");
    }
    if (pool_) { pool_->release_backing(); }
    for (VmmRegion& region : regions_) {
        if (region) { region.release_backing(); }
    }
}

std::vector<MaterializedArtifact::RankArena> MaterializedArtifact::device_arenas() const {
    std::vector<RankArena> out;
    for (std::size_t rank = 0; rank < arenas_.size(); ++rank) {
        if (arenas_[rank] && arenas_[rank]->capacity() != 0) {
            out.push_back({rank, DeviceSpan{arenas_[rank]->base(), arenas_[rank]->capacity()}});
        }
    }
    return out;
}

void MaterializedArtifact::restore_device_backing(DeviceContext& device) {
    if (!suspendable()) {
        throw std::logic_error("this materialization was not made suspendable");
    }
    try {
        if (pool_) { pool_->restore_backing(); }
        for (std::size_t rank = 0; rank < regions_.size(); ++rank) {
            if (!regions_[rank]) { continue; }
            const ScopedRank bound(device, rank);
            regions_[rank].restore_backing();
        }
    } catch (...) {
        // Leave nothing half mapped: a later retry starts from a fully released arena.
        try {
            release_device_backing();
        } catch (...) {}
        throw;
    }
}

MaterializationStats MaterializedArtifact::upload_device_objects_again(DeviceContext& device) {
    if (!suspendable()) {
        throw std::logic_error("this materialization was not made suspendable");
    }
    const StartupObserver no_observer;
    StartupPhaseScope phase(no_observer, StartupPhase::WeightsMaterialize,
                            StartupProgressUnit::Bytes, device_payload_);
    const UploadTotals upload = upload_device_objects(*reader_, device_placements_, arena_bases(),
                                                      device, no_observer, phase, 0,
                                                      device_payload_);
    MaterializationStats stats;
    stats.read_bytes         = upload.read_bytes;
    stats.h2d_bytes          = upload.h2d_bytes;
    stats.peak_staging_bytes = upload.peak_staging_bytes;
    stats.upload_seconds     = upload.seconds;
    return stats;
}

} // namespace ninfer::artifact

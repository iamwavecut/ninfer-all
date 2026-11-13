#include "models/qwen4_exp/expert_stream.h"

#include "core/device_buffer.h"
#include "core/device_wait.h"
#include "core/expert_direct.h"
#include "core/expert_mirror.h"
#include "core/file_read_error.h"
#include "core/pinned_memory.h"
#include "models/qwen4_exp/read_pool.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <cerrno>
#    include <fcntl.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

namespace ninfer::models::qwen4_exp {
namespace {

constexpr std::uint64_t kAlign = 256;
// Two halves of the page-locked ring: one fills from the files while the other's copies run.
constexpr std::size_t kStagingHalf = 128ULL << 20;
// Reads in flight at once: an NVMe drive reaches its bandwidth only with several requests queued.
constexpr std::size_t kReadThreads = 8;
// Zero bytes after every slot's down matrix. The expert matrix kernel reads whole 256-value K steps
// past a 640-value down row's end and decodes what follows as the down's blocks: another format's
// bytes there (the next slot's gate, or a larger down an earlier layer left) can hold a non-finite
// scale, which times the activation's zero padding is NaN.
constexpr std::uint64_t kTail = 256;

std::uint64_t aligned(std::uint64_t value) { return (value + kAlign - 1) / kAlign * kAlign; }

// A read-only file read at explicit offsets, through the page cache.
class File {
public:
    explicit File(const std::filesystem::path& path) {
#ifdef _WIN32
        handle_ = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
        if (handle_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("expert stream: cannot open " + path.string());
        }
#else
        descriptor_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor_ < 0) {
            throw std::runtime_error("expert stream: cannot open " + path.string() + ": " +
                                     std::strerror(errno));
        }
#    if defined(POSIX_FADV_RANDOM)
        (void)::posix_fadvise(descriptor_, 0, 0, POSIX_FADV_RANDOM);
#    endif
#endif
    }

    ~File() {
#ifdef _WIN32
        if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
#else
        if (descriptor_ >= 0) ::close(descriptor_);
#endif
    }

    File(const File&)            = delete;
    File& operator=(const File&) = delete;

    void read(std::uint64_t offset, std::byte* destination, std::size_t bytes) const {
        while (bytes > 0) {
#ifdef _WIN32
            OVERLAPPED position{};
            position.Offset     = static_cast<DWORD>(offset & 0xffffffffULL);
            position.OffsetHigh = static_cast<DWORD>(offset >> 32U);
            const DWORD request = static_cast<DWORD>(
                std::min<std::size_t>(bytes, std::numeric_limits<DWORD>::max() / 2));
            DWORD done = 0;
            if (!ReadFile(handle_, destination, request, &done, &position) || done == 0) {
                throw FileReadError("expert stream: read failed");
            }
#else
            const ssize_t done =
                ::pread(descriptor_, destination, bytes, static_cast<off_t>(offset));
            if (done < 0 && errno == EINTR) continue;
            if (done <= 0) {
                throw FileReadError(std::string("expert stream: read failed: ") +
                                    (done < 0 ? std::strerror(errno) : "end of file"));
            }
#endif
            destination += done;
            offset += static_cast<std::uint64_t>(done);
            bytes -= static_cast<std::size_t>(done);
        }
    }

private:
#ifdef _WIN32
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int descriptor_ = -1;
#endif
};

// One read of a staged batch.
struct Read {
    const File* file       = nullptr;
    std::uint64_t offset   = 0;
    std::byte* destination = nullptr;
    std::size_t bytes      = 0;
};

struct Slot {
    std::int32_t layer  = -1;
    std::int32_t expert = -1;
    std::uint64_t call  = 0; // the last prepare() that needed it
    bool referenced     = false;
    bool released       = false; // in the pool's released queue
    // Bytes from the start of the down region that a down has written: past a smaller down, the
    // tail must be zeroed again.
    std::uint64_t down_written = 0;
};

struct Pool {
    // The slots from the first, then grow's after them.
    DeviceBuffer storage, grown;
    std::size_t rank = 0;
    bool own         = false; // a layer's own slots, sized to its widest call
    std::uint64_t slot_bytes = 0;
    std::array<std::uint64_t, 3> offset{};
    std::vector<Slot> slots;
    std::size_t filled = 0; // slots that have held an expert, from the first
    std::size_t hand   = 0; // CLOCK
    // Taken before the hand's slots: a queued slot routed to again since is skipped.
    std::deque<std::uint32_t> released;
};

} // namespace

struct ExpertStream::Impl {
    struct Layer {
        ExpertStreamLayer spec;
        std::size_t pool = 0;
        std::vector<std::int32_t> slot_of;
        std::array<std::vector<const void*>, 3> mirror;
        std::unique_ptr<PinnedHostBuffer> table_staging;
    };

    DeviceContext& device;
    std::vector<std::unique_ptr<File>> files, mirror_files; // the artifact's, a drive mirror's
    std::vector<Layer> layers;
    std::vector<Pool> pools;
    std::unique_ptr<PinnedHostBuffer> staging;
    // Each half's event of every rank: the copies out of a half run on the stream of the rank
    // that filled it, and CUDA records an event only on a stream of the device it was made on.
    std::vector<std::array<cudaEvent_t, 2>> half_done;
    std::array<std::size_t, 2> half_rank{}; // the rank that last filled each half
    std::size_t half         = 0; // the half being filled
    std::size_t cursor       = 0; // bytes used in it
    cudaStream_t half_stream = nullptr;
    std::uint64_t call       = 0;
    bool grew                = false;
    ExpertStreamStats stats;
    ReadPool reads{kReadThreads - 1}; // the caller reads too

    // The batch being staged in the current half: its reads, then the copies out of the half (a
    // copy without a source zeroes its target).
    struct Copy {
        void* target            = nullptr;
        const std::byte* source = nullptr;
        std::size_t bytes       = 0;
    };

    std::vector<Read> batch_reads;
    std::vector<Copy> batch_copies;
    // Direct reads (core/expert_direct.h): a window per rank that serves layers, by rank, in place
    // of the host staging (its halves at the same offsets); the batch's reads, by window file (the
    // artifact's files, then a drive mirror's).
    std::vector<std::unique_ptr<DirectExpertWindow>> windows;
    std::vector<DirectRead> direct_batch;

    Impl(DeviceContext& d, const std::vector<std::filesystem::path>& paths,
         std::vector<ExpertStreamLayer> specs, std::span<const std::uint64_t> bytes_by_rank,
         bool grows, const ExpertDiskReads& disk)
        : device(d) {
        for (const auto& path : paths) { files.push_back(std::make_unique<File>(path)); }
        for (const auto& path : expert_mirror_files(paths, disk.mirror)) {
            mirror_files.push_back(std::make_unique<File>(path));
        }
        for (auto& spec : specs) {
            Layer layer;
            const std::size_t experts = spec.experts[0].size();
            layer.slot_of.assign(experts, -1);
            for (auto& mirror : layer.mirror) { mirror.assign(experts, nullptr); }
            layer.table_staging = std::make_unique<PinnedHostBuffer>(3 * experts * sizeof(void*));
            layer.spec          = std::move(spec);
            layers.push_back(std::move(layer));
        }
        std::map<std::size_t, std::vector<std::size_t>> on_rank;
        for (std::size_t i = 0; i < layers.size(); ++i) {
            on_rank[layers[i].spec.rank].push_back(i);
        }
        for (const auto& [rank, members] : on_rank) {
            const std::uint64_t budget = rank < bytes_by_rank.size() ? bytes_by_rank[rank] : 0;
            // The slots a pool starts with: all of the budget's, or with `grows` those of half the
            // budget (`first`), at least its widest call.
            const std::uint64_t first = grows ? budget / 2 : budget;
            const auto start = [](std::uint64_t count, std::uint64_t bytes, std::uint64_t slot,
                                  std::uint64_t widest) {
                return std::min(count, std::max(widest, bytes / slot));
            };
            // A layer with slots of its own takes as many as its widest call routes to, wider
            // than the shared slots; the shared slots must still hold the widest call of the
            // others. Otherwise every layer of the rank shares slots fitting the widest.
            std::vector<std::size_t> shared, own;
            for (const std::size_t i : members) {
                (layers[i].spec.own_pool ? own : shared).push_back(i);
            }
            std::uint64_t own_bytes = 0, widest_shared_call = 0;
            for (const std::size_t i : own) { own_bytes += slot_shape({i}).first * call_slots(i); }
            for (const std::size_t i : shared) {
                widest_shared_call = std::max<std::uint64_t>(widest_shared_call, call_slots(i));
            }
            const std::uint64_t shared_slot = shared.empty() ? 0 : slot_shape(shared).first;
            bool split = !own.empty() && !shared.empty() && own_bytes < budget &&
                         (budget - own_bytes) / shared_slot >= widest_shared_call;
            for (const std::size_t i : own) {
                split = split && slot_shape({i}).first > shared_slot;
            }
            if (!split) {
                const std::uint64_t slot = slot_shape(members).first;
                std::uint64_t widest     = 0;
                for (const std::size_t i : members) { widest = std::max(widest, call_slots(i)); }
                add_pool(rank, members, start(budget / slot, first, slot, widest));
                continue;
            }
            for (const std::size_t i : own) {
                add_pool(rank, {i}, call_slots(i));
                pools.back().own = true;
            }
            add_pool(rank, shared,
                     start((budget - own_bytes) / shared_slot,
                           first > own_bytes ? first - own_bytes : 0, shared_slot,
                           widest_shared_call));
        }
        // Every serving rank reads directly, or none does.
        if (disk.direct != ExpertDirect::Off && !on_rank.empty()) {
            std::vector<std::filesystem::path> all = paths;
            const auto mirror                      = expert_mirror_files(paths, disk.mirror);
            all.insert(all.end(), mirror.begin(), mirror.end());
            windows.resize(device.size());
            for (const auto& [rank, members] : on_rank) {
                RankBinding bind(device, rank);
                windows[rank] = open_direct_window(disk, device.rank(rank).device, all,
                                                   2 * kStagingHalf, "expert stream");
                if (!windows[rank]) {
                    windows.clear();
                    break;
                }
            }
        }
        if (windows.empty()) { staging = std::make_unique<PinnedHostBuffer>(2 * kStagingHalf); }
        half_done.resize(device.size());
        if (!on_rank.empty()) { half_rank.fill(on_rank.begin()->first); }
        for (const auto& [rank, members] : on_rank) {
            RankBinding bind(device, rank);
            for (auto& event : half_done.at(rank)) {
                CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            }
        }
    }

    // The most experts one call of layer `i` routes to.
    [[nodiscard]] std::uint64_t call_slots(std::size_t i) const {
        const std::size_t experts = layers[i].slot_of.size();
        const std::size_t widest  = layers[i].spec.widest_call;
        return widest == 0 ? experts : std::min(widest, experts);
    }

    // The slot bytes and the offsets of the up and down in a slot that fits every expert of
    // `members`.
    [[nodiscard]] std::pair<std::uint64_t, std::array<std::uint64_t, 3>>
    slot_shape(const std::vector<std::size_t>& members) const {
        std::array<std::uint64_t, 3> sizes{};
        for (const std::size_t i : members) {
            for (int k = 0; k < 3; ++k) {
                for (const auto& location : layers[i].spec.experts[k]) {
                    sizes[k] = std::max(sizes[k], location.bytes);
                }
            }
        }
        const std::array<std::uint64_t, 3> offset = {0, aligned(sizes[0]),
                                                     aligned(sizes[0]) + aligned(sizes[1])};
        return {offset[2] + aligned(sizes[2] + kTail), offset};
    }

    void add_pool(std::size_t rank, const std::vector<std::size_t>& members, std::uint64_t count) {
        const auto [slot_bytes, offset] = slot_shape(members);
        Pool pool;
        pool.rank       = rank;
        pool.slot_bytes = slot_bytes;
        pool.offset     = offset;
        pool.slots.resize(static_cast<std::size_t>(count));
        if (!pool.slots.empty()) {
            // Zeroed, so each slot's tail stays zero: copies write the matrices only.
            RankBinding bind(device, rank);
            pool.storage = DeviceBuffer(pool.slots.size() * pool.slot_bytes);
            CUDA_CHECK(ninfer::fill_and_wait(pool.storage.p, 0, pool.storage.bytes));
        }
        stats.slots += static_cast<std::uint32_t>(pool.slots.size());
        for (const std::size_t i : members) { layers[i].pool = pools.size(); }
        pools.push_back(std::move(pool));
    }

    std::vector<std::uint64_t> grow(std::span<const std::uint64_t> bytes_by_rank) {
        if (grew) { throw std::logic_error("expert stream: the pools have grown already"); }
        grew = true;
        std::vector<std::uint64_t> added(device.size(), 0);
        for (Pool& pool : pools) {
            if (pool.own || pool.slots.empty() || pool.rank >= bytes_by_rank.size()) { continue; }
            const std::size_t count =
                (bytes_by_rank[pool.rank] - added[pool.rank]) / pool.slot_bytes;
            if (count == 0) { continue; }
            RankBinding bind(device, pool.rank);
            pool.grown = DeviceBuffer(count * pool.slot_bytes);
            CUDA_CHECK(ninfer::fill_and_wait(pool.grown.p, 0, pool.grown.bytes));
            pool.slots.resize(pool.slots.size() + count);
            stats.slots += static_cast<std::uint32_t>(count);
            added[pool.rank] += pool.grown.bytes;
        }
        return added;
    }

    // Slot `s` of `pool`: its own block's, then grow's.
    static std::byte* slot_base(const Pool& pool, std::size_t s) {
        const std::size_t first = pool.storage.bytes / pool.slot_bytes;
        return s < first ? static_cast<std::byte*>(pool.storage.p) + s * pool.slot_bytes
                         : static_cast<std::byte*>(pool.grown.p) + (s - first) * pool.slot_bytes;
    }

    ~Impl() {
        for (std::size_t rank = 0; rank < half_done.size(); ++rank) {
            for (auto& event : half_done[rank]) {
                if (event == nullptr) { continue; }
                RankBinding bind(device, rank);
                cudaEventDestroy(event);
            }
        }
    }

    // Closes the half being filled behind the copies queued from it and starts the other one,
    // once the copies that last read it have finished (an event never recorded is complete). With
    // windows the half is a rank's own, and stage_direct() waits for that rank's copies.
    void switch_half() {
        if (half_stream != nullptr) {
            RankBinding bind(device, half_rank[half]);
            CUDA_CHECK(cudaEventRecord(half_done.at(half_rank[half])[half], half_stream));
        }
        half   = 1 - half;
        cursor = 0;
        if (windows.empty()) {
            CUDA_CHECK(ninfer::wait_for_event(half_done.at(half_rank[half])[half]));
        }
    }

    // Forgets every resident expert: after a failed read some slot holds an expert it was never
    // given, and the slots do not record which. Later calls read what they route to again. The
    // half being filled keeps its cursor, so no staged byte a queued copy still reads is reused.
    void forget_residents() {
        for (auto& pool : pools) {
            for (Slot& slot : pool.slots) {
                slot.layer      = -1;
                slot.expert     = -1;
                slot.call       = 0;
                slot.referenced = false;
                slot.released   = false;
            }
            pool.filled = 0;
            pool.released.clear();
        }
        for (Layer& layer : layers) { std::fill(layer.slot_of.begin(), layer.slot_of.end(), -1); }
        batch_reads.clear();
        direct_batch.clear();
        batch_copies.clear();
    }

    // Reads the staged batch in parallel and queues its copies on `stream`.
    void flush(cudaStream_t stream) {
        try {
            if (!windows.empty()) {
                if (!direct_batch.empty()) { windows.at(half_rank[half])->read(direct_batch); }
            } else {
                reads.run(batch_reads.size(), [&](std::size_t i) {
                    const Read& read = batch_reads[i];
                    read.file->read(read.offset, read.destination, read.bytes);
                });
            }
        } catch (const FileReadError&) {
            forget_residents();
            throw;
        }
        for (const Copy& copy : batch_copies) {
            if (copy.source == nullptr) {
                CUDA_CHECK(cudaMemsetAsync(copy.target, 0, copy.bytes, stream));
            } else {
                CUDA_CHECK(cudaMemcpyAsync(
                    copy.target, copy.source, copy.bytes,
                    windows.empty() ? cudaMemcpyHostToDevice : cudaMemcpyDefault, stream));
            }
        }
        batch_reads.clear();
        direct_batch.clear();
        batch_copies.clear();
    }

    // Direct reads: `cover` window bytes (whole blocks) in the half being filled of `rank`'s
    // window, after that half's last copies on the rank have finished; their window offset.
    std::uint64_t stage_direct(std::uint64_t cover, std::size_t rank, cudaStream_t stream) {
        if (cover > kStagingHalf) {
            throw std::logic_error("expert stream: an expert exceeds the staging half");
        }
        if (cursor + cover > kStagingHalf || (half_stream != nullptr && half_stream != stream)) {
            flush(half_stream != nullptr ? half_stream : stream);
            switch_half();
        }
        if (cursor == 0) { CUDA_CHECK(ninfer::wait_for_event(half_done.at(rank)[half])); }
        half_stream            = stream;
        half_rank[half]        = rank;
        const std::uint64_t at = half * kStagingHalf + cursor;
        cursor += static_cast<std::size_t>(cover);
        return at;
    }

    std::byte* stage(std::size_t bytes, std::size_t rank, cudaStream_t stream) {
        if (bytes > kStagingHalf) {
            throw std::logic_error("expert stream: an expert exceeds the staging half");
        }
        if (cursor + bytes > kStagingHalf || (half_stream != nullptr && half_stream != stream)) {
            // The half closes behind the batch staged in it.
            flush(half_stream != nullptr ? half_stream : stream);
            switch_half();
        }
        half_stream     = stream;
        half_rank[half] = rank;
        std::byte* out  = static_cast<std::byte*>(staging->data()) + half * kStagingHalf + cursor;
        cursor += aligned(bytes);
        return out;
    }

    std::size_t victim(Pool& pool) {
        if (pool.filled < pool.slots.size()) { return pool.filled++; }
        // Released slots next: none holds an expert this call needs, whose hit cleared the mark.
        while (!pool.released.empty()) {
            const std::size_t index = pool.released.front();
            pool.released.pop_front();
            if (pool.slots[index].released) {
                pool.slots[index].released = false;
                return index;
            }
        }
        // CLOCK: a slot used since the hand last passed gets another round, a slot this call
        // needs is never taken. The caller has checked that the call fits the pool.
        for (;;) {
            Slot& slot              = pool.slots[pool.hand];
            const std::size_t index = pool.hand;
            pool.hand               = (pool.hand + 1) % pool.slots.size();
            if (slot.call == call && slot.expert >= 0) { continue; }
            if (slot.referenced) {
                slot.referenced = false;
                continue;
            }
            return index;
        }
    }

    void prepare(std::size_t index, std::span<const std::int32_t> ids, bool release_others) {
        Layer& layer = layers.at(index);
        Pool& pool   = pools.at(layer.pool);
        ++call;
        std::vector<std::int32_t> needed;
        needed.reserve(ids.size());
        const auto experts = static_cast<std::int32_t>(layer.slot_of.size());
        for (const std::int32_t id : ids) {
            if (id >= 0 && id < experts) { needed.push_back(id); }
        }
        std::sort(needed.begin(), needed.end());
        needed.erase(std::unique(needed.begin(), needed.end()), needed.end());
        if (needed.size() > pool.slots.size()) {
            throw std::runtime_error("expert stream: one layer routes to " +
                                     std::to_string(needed.size()) +
                                     " experts but its device slots hold " +
                                     std::to_string(pool.slots.size()));
        }
        std::vector<std::int32_t> missing;
        for (const std::int32_t expert : needed) {
            ++stats.routes;
            const std::int32_t slot = layer.slot_of[expert];
            if (slot >= 0) {
                ++stats.hits;
                pool.slots[slot].call       = call;
                pool.slots[slot].referenced = true;
                pool.slots[slot].released   = false;
            } else {
                missing.push_back(expert);
            }
        }
        if (release_others) {
            for (const std::int32_t s : layer.slot_of) {
                if (s < 0) { continue; }
                Slot& slot = pool.slots[s];
                if (slot.call == call || slot.released) { continue; }
                slot.released   = true;
                slot.referenced = false;
                pool.released.push_back(static_cast<std::uint32_t>(s));
            }
        }
        if (missing.empty()) { return; }
        RankBinding bind(device, layer.spec.rank);
        const cudaStream_t stream = layer.spec.stream;
        const auto start          = std::chrono::steady_clock::now();
        for (const std::int32_t expert : missing) {
            const std::size_t s = victim(pool);
            Slot& slot          = pool.slots[s];
            if (slot.expert >= 0) { layers[slot.layer].slot_of[slot.expert] = -1; }
            auto* base = slot_base(pool, s);
            for (int k = 0; k < 3; ++k) {
                const ExpertLocation& location = layer.spec.experts[k][expert];
                std::byte* target              = base + pool.offset[k];
                if (!windows.empty()) {
                    // Each run's cover in the rank's window, and its copy out of it.
                    const DirectExpertWindow& window = *windows.at(layer.spec.rank);
                    std::uint64_t at                 = 0;
                    for (const auto& run : location.runs) {
                        if (run.bytes == 0) { continue; }
                        const std::uint64_t window_at = stage_direct(
                            direct_cover(run.offset, run.bytes), layer.spec.rank, stream);
                        const bool mirrored =
                            !mirror_files.empty() && expert_mirror_copy(run.offset) == 1;
                        direct_batch.push_back({run.file + (mirrored ? files.size() : 0),
                                                run.offset, run.bytes, window_at});
                        batch_copies.push_back({target + at,
                                                window.data() + window_at + direct_lead(run.offset),
                                                static_cast<std::size_t>(run.bytes)});
                        at += run.bytes;
                    }
                } else {
                    std::byte* buffer = stage(location.bytes, layer.spec.rank, stream);
                    std::uint64_t at  = 0;
                    for (const auto& run : location.runs) {
                        if (run.bytes == 0) { continue; }
                        const auto& copies =
                            mirror_files.empty() || expert_mirror_copy(run.offset) == 0
                                ? files
                                : mirror_files;
                        batch_reads.push_back({copies.at(run.file).get(), run.offset, buffer + at,
                                               static_cast<std::size_t>(run.bytes)});
                        at += run.bytes;
                    }
                    batch_copies.push_back(
                        {target, buffer, static_cast<std::size_t>(location.bytes)});
                }
                if (k == 2) {
                    if (slot.down_written > location.bytes) {
                        batch_copies.push_back({target + location.bytes, nullptr,
                                                static_cast<std::size_t>(std::min(
                                                    slot.down_written - location.bytes, kTail))});
                    }
                    slot.down_written = std::max(slot.down_written, location.bytes);
                }
                layer.mirror[k][expert] = target;
                stats.read_bytes += location.bytes;
            }
            slot.layer            = static_cast<std::int32_t>(index);
            slot.expert           = expert;
            slot.call             = call;
            slot.referenced       = true;
            layer.slot_of[expert] = static_cast<std::int32_t>(s);
        }
        flush(stream);
        auto* tables = static_cast<std::byte*>(layer.table_staging->data());
        for (int k = 0; k < 3; ++k) {
            const std::size_t bytes = layer.mirror[k].size() * sizeof(void*);
            std::memcpy(tables + k * bytes, layer.mirror[k].data(), bytes);
            CUDA_CHECK(cudaMemcpyAsync(layer.spec.tables[k], tables + k * bytes, bytes,
                                       cudaMemcpyHostToDevice, stream));
        }
        stats.read_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
};

ExpertStream::ExpertStream(DeviceContext& device, const std::vector<std::filesystem::path>& files,
                           std::vector<ExpertStreamLayer> layers,
                           std::span<const std::uint64_t> bytes_by_rank, bool grows,
                           const ExpertDiskReads& disk)
    : impl_(std::make_unique<Impl>(device, files, std::move(layers), bytes_by_rank, grows, disk)) {}

ExpertStream::~ExpertStream() = default;

void ExpertStream::prepare(std::size_t layer, std::span<const std::int32_t> ids,
                           bool release_others) {
    impl_->prepare(layer, ids, release_others);
}

ExpertStreamStats ExpertStream::stats() const noexcept { return impl_->stats; }

ExpertStream::Ranges ExpertStream::storage(std::size_t layer) const {
    const auto& pool  = impl_->pools.at(impl_->layers.at(layer).pool);
    const auto range = [](const DeviceBuffer& block) {
        const auto begin = reinterpret_cast<std::uintptr_t>(block.p);
        return std::pair<std::uintptr_t, std::uintptr_t>{begin, begin + block.bytes};
    };
    return {range(pool.storage), range(pool.grown)};
}

std::vector<std::uint64_t> ExpertStream::grow(std::span<const std::uint64_t> bytes_by_rank) {
    return impl_->grow(bytes_by_rank);
}

std::uint64_t ExpertStream::pool_bytes(std::size_t rank) const {
    std::uint64_t bytes = 0;
    for (const auto& pool : impl_->pools) {
        if (pool.rank == rank) { bytes += pool.storage.bytes + pool.grown.bytes; }
    }
    return bytes;
}

} // namespace ninfer::models::qwen4_exp

#include "core/expert_direct.h"

#include "core/device.h"
#include "core/device_wait.h"
#include "core/file_read_error.h"
#include "core/host_worker_pool.h"
#include "core/pinned_memory.h"
#include "core/vmm.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

#if !defined(_WIN32)
#    include <fcntl.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

#if defined(__linux__)
#    include "../../kernel/ninfer-direct/ninfer_direct_uapi.h"

#    include <linux/fiemap.h>
#    include <linux/fs.h>
#    include <sys/ioctl.h>
#    include <sys/mman.h>
#    include <sys/sysmacros.h>
#endif

namespace ninfer {

std::vector<FileExtent> direct_device_runs(std::span<const FileExtent> extents,
                                           std::uint64_t offset, std::uint64_t bytes) {
    std::vector<FileExtent> runs;
    if (bytes == 0) { return runs; }
    auto extent = std::upper_bound(
        extents.begin(), extents.end(), offset,
        [](std::uint64_t at, const FileExtent& candidate) { return at < candidate.logical; });
    if (extent != extents.begin()) { --extent; }
    const std::uint64_t end = offset + bytes;
    for (std::uint64_t at = offset; at < end; ++extent) {
        if (extent == extents.end() || at < extent->logical ||
            at - extent->logical >= extent->bytes) {
            throw std::runtime_error("direct reads: the file's extents do not hold byte " +
                                     std::to_string(at));
        }
        const std::uint64_t within   = at - extent->logical;
        const std::uint64_t count    = std::min(end - at, extent->bytes - within);
        const std::uint64_t physical = extent->physical + within;
        if (!runs.empty() && runs.back().physical + runs.back().bytes == physical) {
            runs.back().bytes += count;
        } else {
            runs.push_back({at, physical, count});
        }
        at += count;
    }
    return runs;
}

namespace {

using Refusal = DirectExpertWindow::Refusal;

// A window is whole 64 KiB pages: the unit the NVIDIA driver pins in BAR1.
constexpr std::uint64_t kWindowPage    = 64ULL << 10;
constexpr std::uint32_t kReaders       = 8;
constexpr std::uint64_t kSelfTestBytes = 1ULL << 20;

std::uint64_t round_up(std::uint64_t value, std::uint64_t unit) {
    return (value + unit - 1) / unit * unit;
}

// Opening a window stopped: the direct path is unavailable for `refusal`.
struct Unavailable {
    Refusal refusal;
    std::string why;
};

[[noreturn]] void refuse(std::string why) { throw Unavailable{Refusal::Failed, std::move(why)}; }

std::string hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%x", value);
    return text;
}

std::string read_text(const std::string& path) {
    std::ifstream file(path);
    std::stringstream text;
    text << file.rdbuf();
    return text.str();
}

std::vector<std::byte> read_normally(const std::filesystem::path& path, std::uint64_t offset,
                                     std::uint64_t bytes) {
    std::ifstream file(path, std::ios::binary);
    std::vector<std::byte> out(static_cast<std::size_t>(bytes));
    file.seekg(static_cast<std::streamoff>(offset));
    file.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    if (!file) { refuse("cannot read " + path.string()); }
    return out;
}

// One positioned read: `bytes` (whole blocks) into `destination`, of which the first `needed`
// must arrive (fewer than `bytes` only at the end of a file read through the file itself).
struct ReadTask {
#if defined(_WIN32)
    const std::filesystem::path* path = nullptr;
#else
    int descriptor = -1;
#endif
    std::uint64_t position = 0;
    std::byte* destination = nullptr;
    std::uint64_t bytes    = 0;
    std::uint64_t needed   = 0;
};

void run_task(const ReadTask& task) {
#if defined(_WIN32)
    std::ifstream file(*task.path, std::ios::binary);
    file.seekg(static_cast<std::streamoff>(task.position));
    file.read(reinterpret_cast<char*>(task.destination), static_cast<std::streamsize>(task.bytes));
    if (std::uint64_t(file.gcount()) < task.needed) {
        throw FileReadError("direct reads: cannot read " + task.path->string());
    }
#else
    std::uint64_t have = 0;
    while (have < task.needed) {
        const std::uint64_t request = std::min<std::uint64_t>(task.bytes - have, 1ULL << 30);
        const ssize_t done          = ::pread(task.descriptor, task.destination + have, request,
                                              static_cast<off_t>(task.position + have));
        if (done < 0 && errno == EINTR) { continue; }
        if (done <= 0) {
            throw FileReadError(std::string("direct reads: read failed: ") +
                                (done < 0 ? std::strerror(errno) : "end of file"));
        }
        have += static_cast<std::uint64_t>(done);
    }
#endif
}

// Runs every task on the pool; every task has ended when this returns or throws.
void run_tasks(HostWorkerPool& pool, std::span<const ReadTask> tasks) {
    if (tasks.size() == 1) {
        run_task(tasks.front());
        return;
    }
    std::vector<std::future<void>> pending;
    pending.reserve(tasks.size());
    std::exception_ptr failure;
    for (const ReadTask& task : tasks) {
        try {
            pending.push_back(pool.submit([&task] { run_task(task); }));
        } catch (...) {
            failure = std::current_exception();
            break;
        }
    }
    for (auto& read : pending) {
        try {
            read.get();
        } catch (...) {
            if (!failure) { failure = std::current_exception(); }
        }
    }
    if (failure) { std::rethrow_exception(failure); }
}

void check_reads(std::span<const DirectRead> reads, std::span<const std::uint64_t> sizes,
                 std::uint64_t window_bytes) {
    for (const DirectRead& read : reads) {
        if (read.file >= sizes.size() || read.window_offset % kDirectBlock != 0 ||
            read.bytes > sizes[read.file] || read.offset > sizes[read.file] - read.bytes ||
            read.window_offset > window_bytes ||
            direct_cover(read.offset, read.bytes) > window_bytes - read.window_offset) {
            throw std::invalid_argument("direct reads: a read outside its file or the window");
        }
    }
}

// Reads a range of each file over the window and compares it with a normal read; the window's
// bytes start as `fill` (filled by `fill_window`), so a read that writes nothing differs.
void self_test(DirectExpertWindow& window, int device, std::span<const std::filesystem::path> files,
               std::span<const std::uint64_t> sizes,
               const std::function<void(std::uint64_t)>& fill_window) {
    DeviceBinding bind(device);
    for (std::size_t f = 0; f < files.size(); ++f) {
        if (sizes[f] == 0) { continue; }
        // A start inside a block, so the reads cover a lead.
        const std::uint64_t offset = sizes[f] > 3 * kDirectBlock ? kDirectBlock + 1 : 0;
        const std::uint64_t bytes =
            std::min({sizes[f] - offset, kSelfTestBytes, window.bytes() - 2 * kDirectBlock});
        fill_window(direct_cover(offset, bytes));
        const DirectRead read{f, offset, bytes, 0};
        window.read({&read, 1});
        std::vector<std::byte> back(static_cast<std::size_t>(bytes));
        const cudaError_t copied = copy_and_wait(back.data(), window.data() + direct_lead(offset),
                                                 back.size(), cudaMemcpyDefault);
        if (copied != cudaSuccess) {
            refuse(std::string("self-test copy failed: ") + cudaGetErrorString(copied));
        }
        const auto expected = read_normally(files[f], offset, bytes);
        const auto differ   = std::mismatch(back.begin(), back.end(), expected.begin());
        if (differ.first != back.end()) {
            refuse("self-test read of " + files[f].string() +
                   " differs from a normal read at byte " +
                   std::to_string(offset + std::uint64_t(differ.first - back.begin())));
        }
    }
}

std::vector<std::uint64_t> file_sizes(std::span<const std::filesystem::path> files) {
    std::vector<std::uint64_t> sizes;
    for (const auto& path : files) {
        std::error_code error;
        const auto size = std::filesystem::file_size(path, error);
        if (error) { refuse("cannot stat " + path.string()); }
        sizes.push_back(size);
    }
    return sizes;
}

// Tests: pinned host memory as the window, filled by reads of the files themselves (O_DIRECT where
// the filesystem takes it), with the same covers, tasks and copies as the device window.
class HostWindow final : public DirectExpertWindow {
public:
    HostWindow(std::span<const std::filesystem::path> files, std::uint64_t bytes)
        : bytes_(round_up(std::max<std::uint64_t>(bytes, kWindowPage), kWindowPage)),
          buffer_(static_cast<std::size_t>(bytes_ + kDirectBlock)),
          base_(reinterpret_cast<std::byte*>(
              round_up(reinterpret_cast<std::uintptr_t>(buffer_.data()), kDirectBlock))),
          sizes_(file_sizes(files)) {
#if defined(_WIN32)
        paths_.assign(files.begin(), files.end());
#else
        for (const auto& path : files) {
            int flags = O_RDONLY | O_CLOEXEC;
#    if defined(__linux__)
            flags |= O_DIRECT;
#    endif
            int descriptor = ::open(path.c_str(), flags);
#    if defined(__linux__)
            if (descriptor < 0 && errno == EINVAL) { // a filesystem without O_DIRECT (tmpfs)
                descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
            }
#    endif
            if (descriptor < 0) {
                refuse("cannot open " + path.string() + ": " + std::strerror(errno));
            }
            descriptors_.push_back(descriptor);
        }
#endif
        pool_ = std::make_unique<HostWorkerPool>(kReaders, 4096);
    }

    ~HostWindow() override {
#if !defined(_WIN32)
        for (const int descriptor : descriptors_) { ::close(descriptor); }
#endif
    }

    const std::byte* data() const noexcept override { return base_; }

    std::uint64_t bytes() const noexcept override { return bytes_; }

    void read(std::span<const DirectRead> reads) override {
        check_reads(reads, sizes_, bytes_);
        tasks_.clear();
        for (const DirectRead& read : reads) {
            ReadTask task;
#if defined(_WIN32)
            task.path = &paths_[read.file];
#else
            task.descriptor = descriptors_[read.file];
#endif
            task.position    = read.offset - direct_lead(read.offset);
            task.destination = base_ + read.window_offset;
            task.bytes       = direct_cover(read.offset, read.bytes);
            task.needed      = direct_lead(read.offset) + read.bytes;
            tasks_.push_back(task);
        }
        run_tasks(*pool_, tasks_);
    }

    std::string describe() const override { return "the files themselves (host window, test)"; }

    void fill(std::uint64_t bytes) { std::memset(base_, 0xA5, static_cast<std::size_t>(bytes)); }

    const std::vector<std::uint64_t>& sizes() const noexcept { return sizes_; }

private:
    std::uint64_t bytes_ = 0;
    PinnedHostBuffer buffer_;
    std::byte* base_ = nullptr;
    std::vector<std::uint64_t> sizes_;
#if defined(_WIN32)
    std::vector<std::filesystem::path> paths_;
#else
    std::vector<int> descriptors_;
#endif
    std::vector<ReadTask> tasks_;
    std::unique_ptr<HostWorkerPool> pool_;
};

#if defined(__linux__)

// The NVIDIA driver's resource manager interface as the open GPU kernel modules declare it
// (nvos.h, nv_escape.h, cl0000.h, cl0080.h, cl2080.h, cl503c.h, ctrl503c.h, ctrl0000unix.h); the
// layouts and numbers are the same in drivers 595.99.02 and 620.70.06. A driver that changes them
// refuses the calls, and the window is unavailable.
namespace rm {

using Handle = std::uint32_t;

struct Alloc { // NVOS64_PARAMETERS
    Handle root = 0, parent = 0, object = 0;
    std::uint32_t class_id          = 0;
    alignas(8) std::uint64_t params = 0;
    alignas(8) std::uint64_t rights = 0;
    std::uint32_t params_size = 0, flags = 0, status = 0;
};

static_assert(sizeof(Alloc) == 48);

struct Control { // NVOS54_PARAMETERS
    Handle client = 0, object = 0;
    std::uint32_t command = 0, flags = 0;
    alignas(8) std::uint64_t params = 0;
    std::uint32_t params_size = 0, status = 0;
};

static_assert(sizeof(Control) == 32);

struct DeviceAlloc { // NV0080_ALLOC_PARAMETERS
    std::uint32_t device_id = 0;
    Handle client_share = 0, target_client = 0, target_device = 0;
    std::uint32_t flags                    = 0;
    alignas(8) std::uint64_t va_space_size = 0, va_start = 0, va_limit = 0;
    std::uint32_t va_mode = 0;
};

static_assert(sizeof(DeviceAlloc) == 56);

struct SubdeviceAlloc { // NV2080_ALLOC_PARAMETERS
    std::uint32_t subdevice_id = 0;
};

struct ThirdPartyAlloc { // NV503C_ALLOC_PARAMETERS
    std::uint32_t flags = 0;
};

struct ImportFromFd { // NV0000_CTRL_OS_UNIX_IMPORT_OBJECT_FROM_FD_PARAMS
    std::int32_t fd    = -1;
    std::uint32_t type = 0; // NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE
    Handle device = 0, parent = 0, object = 0;
};

static_assert(sizeof(ImportFromFd) == 20);

struct RegisterVaSpace { // NV503C_CTRL_REGISTER_VA_SPACE_PARAMS
    Handle va_space                = 0;
    alignas(8) std::uint64_t token = 0;
};

static_assert(sizeof(RegisterVaSpace) == 16);

struct RegisterVidmem { // NV503C_CTRL_REGISTER_VIDMEM_PARAMS
    Handle memory                    = 0;
    alignas(8) std::uint64_t address = 0, size = 0, offset = 0;
};

static_assert(sizeof(RegisterVidmem) == 32);

struct RegisterPid { // NV503C_CTRL_REGISTER_PID_PARAMS
    Handle client = 0;
};

constexpr std::uint32_t kRoot            = 0x0;    // NV01_ROOT
constexpr std::uint32_t kDevice          = 0x80;   // NV01_DEVICE_0
constexpr std::uint32_t kSubdevice       = 0x2080; // NV20_SUBDEVICE_0
constexpr std::uint32_t kThirdPartyP2p   = 0x503c; // NV50_THIRD_PARTY_P2P
constexpr std::uint32_t kThirdPartyBar1  = 1;      // NV503C_ALLOC_PARAMETERS_FLAGS_TYPE_BAR1
constexpr std::uint32_t kExportTypeRm    = 1;      // NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM
constexpr std::uint32_t kImportFromFd    = 0x3d06;
constexpr std::uint32_t kRegisterVaSpace = 0x503c0102;
constexpr std::uint32_t kRegisterVidmem  = 0x503c0104;
constexpr std::uint32_t kRegisterPid     = 0x503c0106;

// NV_IOCTL_MAGIC with NV_ESC_RM_ALLOC and NV_ESC_RM_CONTROL.
const unsigned long kAllocIoctl   = _IOWR('F', 0x2B, Alloc);
const unsigned long kControlIoctl = _IOWR('F', 0x2A, Control);

Handle alloc(int control, Handle root, Handle parent, std::uint32_t class_id, void* params,
             std::uint32_t size, const char* what) {
    Alloc request;
    request.root        = root;
    request.parent      = parent;
    request.class_id    = class_id;
    request.params      = reinterpret_cast<std::uintptr_t>(params);
    request.params_size = size;
    if (::ioctl(control, kAllocIoctl, &request) < 0) {
        refuse(std::string("NVIDIA driver: ") + what + ": " + std::strerror(errno));
    }
    if (request.status != 0) {
        refuse(std::string("NVIDIA driver: ") + what + " failed (status " + hex(request.status) +
               ")");
    }
    return request.object;
}

void control(int control, Handle client, Handle object, std::uint32_t command, void* params,
             std::uint32_t size, const char* what) {
    Control request;
    request.client      = client;
    request.object      = object;
    request.command     = command;
    request.params      = reinterpret_cast<std::uintptr_t>(params);
    request.params_size = size;
    if (::ioctl(control, kControlIoctl, &request) < 0) {
        refuse(std::string("NVIDIA driver: ") + what + ": " + std::strerror(errno));
    }
    if (request.status != 0) {
        refuse(std::string("NVIDIA driver: ") + what + " failed (status " + hex(request.status) +
               ")");
    }
}

} // namespace rm

int open_device(const char* path, int flags) {
    const int descriptor = ::open(path, flags | O_CLOEXEC);
    if (descriptor < 0) {
        refuse(
            std::string("cannot open ") + path + ": " + std::strerror(errno) +
            (errno == EACCES ? " (kernel/ninfer-direct/99-ninfer-direct.rules, group disk)" : ""));
    }
    return descriptor;
}

// The file's extents on its block device: plain, 4 KiB-aligned, written, and together covering
// the file, its last block whole, without holes.
std::vector<FileExtent> file_extents(int descriptor, const std::filesystem::path& path,
                                     std::uint64_t size) {
    constexpr std::uint32_t kRefused = FIEMAP_EXTENT_UNKNOWN | FIEMAP_EXTENT_DELALLOC |
                                       FIEMAP_EXTENT_ENCODED | FIEMAP_EXTENT_DATA_ENCRYPTED |
                                       FIEMAP_EXTENT_NOT_ALIGNED | FIEMAP_EXTENT_DATA_INLINE |
                                       FIEMAP_EXTENT_DATA_TAIL | FIEMAP_EXTENT_UNWRITTEN;
    constexpr std::uint32_t kBatch   = 512;
    std::vector<FileExtent> extents;
    std::vector<std::byte> buffer(sizeof(fiemap) + kBatch * sizeof(fiemap_extent));
    std::uint64_t covered = 0;
    for (bool last = false; !last;) {
        auto* map = reinterpret_cast<fiemap*>(buffer.data());
        std::memset(map, 0, buffer.size());
        map->fm_start        = covered;
        map->fm_length       = ~0ULL - covered;
        map->fm_flags        = FIEMAP_FLAG_SYNC;
        map->fm_extent_count = kBatch;
        if (::ioctl(descriptor, FS_IOC_FIEMAP, map) < 0) {
            refuse("the filesystem of " + path.string() +
                   " gives no extent map (FIEMAP): " + std::strerror(errno));
        }
        if (map->fm_mapped_extents == 0) { break; }
        for (std::uint32_t i = 0; i < map->fm_mapped_extents; ++i) {
            const fiemap_extent& extent = map->fm_extents[i];
            if ((extent.fe_flags & kRefused) != 0 ||
                ((extent.fe_logical | extent.fe_physical | extent.fe_length) % kDirectBlock) != 0) {
                refuse(path.string() + " has an extent the drive cannot read directly (flags " +
                       hex(extent.fe_flags) + ")");
            }
            if (extent.fe_logical != covered) {
                refuse(path.string() + " has a hole at byte " + std::to_string(covered));
            }
            extents.push_back({extent.fe_logical, extent.fe_physical, extent.fe_length});
            covered = extent.fe_logical + extent.fe_length;
            last    = last || (extent.fe_flags & FIEMAP_EXTENT_LAST) != 0;
        }
    }
    if (covered < round_up(size, kDirectBlock)) {
        refuse("the extents of " + path.string() + " cover less than the file");
    }
    return extents;
}

// The block device a file lives on, by its number: FIEMAP's physical offsets are relative to it
// (a partition's, when the file system is on a partition).
std::string block_device(dev_t number) {
    const std::string uevent = read_text("/sys/dev/block/" + std::to_string(major(number)) + ":" +
                                         std::to_string(minor(number)) + "/uevent");
    const auto at            = uevent.find("DEVNAME=");
    if (at == std::string::npos) {
        refuse("no block device for device number " + std::to_string(major(number)) + ":" +
               std::to_string(minor(number)));
    }
    const auto end = uevent.find('\n', at);
    return "/dev/" + uevent.substr(at + 8, end == std::string::npos ? end : end - at - 8);
}

// The window: an exportable CUDA allocation, imported into a resource manager client of this
// process and registered for third-party P2P access under a key, pinned in BAR1 by ninfer_pin,
// handed to the kernel's PCI P2PDMA layer and mapped by ninfer_p2p. Reads go from the raw block
// device under each file, at the file's extents, with O_DIRECT into that mapping.
class ModuleWindow final : public DirectExpertWindow {
public:
    ModuleWindow(int device, std::span<const std::filesystem::path> files, std::uint64_t bytes)
        : device_(device) {
        if (::access("/dev/ninfer_pin", F_OK) != 0 || ::access("/dev/ninfer_p2p", F_OK) != 0) {
            throw Unavailable{Refusal::NotSetUp,
                              "the kernel modules of kernel/ninfer-direct are not loaded"};
        }
        try {
            sizes_ = file_sizes(files);
            open_window(bytes);
            open_sources(files);
            pool_ = std::make_unique<HostWorkerPool>(kReaders, 4096);
        } catch (...) {
            release();
            throw;
        }
    }

    ~ModuleWindow() override { release(); }

    const std::byte* data() const noexcept override {
        return reinterpret_cast<const std::byte*>(static_cast<std::uintptr_t>(address_));
    }

    std::uint64_t bytes() const noexcept override { return bytes_; }

    void read(std::span<const DirectRead> reads) override {
        check_reads(reads, sizes_, bytes_);
        tasks_.clear();
        for (const DirectRead& read : reads) {
            const Source& source      = sources_[read.file];
            const std::uint64_t start = read.offset - direct_lead(read.offset);
            std::byte* const out      = mapping_ + read.window_offset;
            // A cover past the file's end stays inside its last block, which an extent holds.
            for (const FileExtent& run :
                 direct_device_runs(source.extents, start, direct_cover(read.offset, read.bytes))) {
                tasks_.push_back(ReadTask{source.raw, run.physical, out + (run.logical - start),
                                          run.bytes, run.bytes});
            }
        }
        run_tasks(*pool_, tasks_);
        if (flush_) {
            NINFER_CU_CHECK(cuCtxPushCurrent(context_));
            const CUresult flushed =
                cuFlushGPUDirectRDMAWrites(CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
                                           CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER);
            CUcontext popped = nullptr;
            (void)cuCtxPopCurrent(&popped);
            NINFER_CU_CHECK(flushed);
        }
    }

    std::string describe() const override { return devices_; }

    void fill(std::uint64_t bytes) {
        NINFER_CU_CHECK(cuCtxPushCurrent(context_));
        const CUresult filled = cuMemsetD8(address_, 0xA5, static_cast<std::size_t>(bytes));
        const CUresult synced = filled == CUDA_SUCCESS ? cuCtxSynchronize() : filled;
        CUcontext popped      = nullptr;
        (void)cuCtxPopCurrent(&popped);
        NINFER_CU_CHECK(synced);
    }

    const std::vector<std::uint64_t>& sizes() const noexcept { return sizes_; }

private:
    struct Source {
        std::vector<FileExtent> extents;
        int raw = -1; // the block device's descriptor, owned by raw_
    };

    void open_window(std::uint64_t requested) {
        pin_ = open_device("/dev/ninfer_pin", O_RDWR);
        p2p_ = open_device("/dev/ninfer_p2p", O_RDWR);
        vmm::ensure_driver_initialized();
        NINFER_CU_CHECK(cuDeviceGet(&cu_device_, device_));
        NINFER_CU_CHECK(cuDevicePrimaryCtxRetain(&context_, cu_device_));
        int exportable = 0;
        NINFER_CU_CHECK(cuDeviceGetAttribute(
            &exportable, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
            cu_device_));
        if (exportable == 0) { refuse("the GPU cannot export its memory as a file descriptor"); }
        int flush_options = 0;
        NINFER_CU_CHECK(cuDeviceGetAttribute(
            &flush_options, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS, cu_device_));
        flush_ = (flush_options & CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST) != 0;

        char bus_id[64] = {};
        NINFER_CU_CHECK(cuDeviceGetPCIBusId(bus_id, sizeof(bus_id), cu_device_));
        unsigned domain = 0, bus = 0, slot = 0, function = 0;
        if (std::sscanf(bus_id, "%x:%x:%x.%x", &domain, &bus, &slot, &function) != 4) {
            refuse(std::string("cannot parse the GPU's PCI address ") + bus_id);
        }
        char location[32];
        std::snprintf(location, sizeof(location), "%04x:%02x:%02x.%x", domain, bus, slot, function);
        const std::string information =
            read_text(std::string("/proc/driver/nvidia/gpus/") + location + "/information");
        const auto at = information.find("Device Minor:");
        if (at == std::string::npos) {
            refuse(std::string("no NVIDIA device node for ") + location);
        }
        const int minor_number = std::atoi(information.c_str() + at + 13);

        CUmemAllocationProp properties{};
        properties.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
        properties.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
        properties.location.id          = cu_device_;
        properties.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        std::size_t granularity         = 0;
        NINFER_CU_CHECK(cuMemGetAllocationGranularity(&granularity, &properties,
                                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        bytes_ = round_up(round_up(std::max<std::uint64_t>(requested, kWindowPage), kWindowPage),
                          std::max<std::uint64_t>(granularity, kWindowPage));
        NINFER_CU_CHECK(cuMemCreate(&allocation_, bytes_, &properties, 0));
        NINFER_CU_CHECK(cuMemAddressReserve(&address_, bytes_, 0, 0, 0));
        NINFER_CU_CHECK(cuMemMap(address_, bytes_, 0, allocation_, 0));
        mapped_ = true;
        CUmemAccessDesc access{};
        access.location = properties.location;
        access.flags    = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        NINFER_CU_CHECK(cuMemSetAccess(address_, bytes_, &access, 1));
        int exported = -1;
        NINFER_CU_CHECK(cuMemExportToShareableHandle(&exported, allocation_,
                                                     CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));

        // A resource manager client of this process imports the allocation and registers it for
        // third-party P2P access under `key`, by which ninfer_pin's call finds it.
        rm::ImportFromFd import;
        try {
            control_ = open_device("/dev/nvidiactl", O_RDWR);
            gpu_     = open_device(("/dev/nvidia" + std::to_string(minor_number)).c_str(), O_RDWR);
            const rm::Handle client = rm::alloc(control_, 0, 0, rm::kRoot, nullptr, 0, "client");
            rm::DeviceAlloc device_params;
            device_params.device_id = std::uint32_t(minor_number);
            const rm::Handle device_handle =
                rm::alloc(control_, client, client, rm::kDevice, &device_params,
                          sizeof(device_params), "device");
            rm::SubdeviceAlloc subdevice_params;
            const rm::Handle subdevice =
                rm::alloc(control_, client, device_handle, rm::kSubdevice, &subdevice_params,
                          sizeof(subdevice_params), "subdevice");
            import.fd     = exported;
            import.type   = rm::kExportTypeRm;
            import.device = device_handle;
            import.parent = device_handle;
            rm::control(control_, client, client, rm::kImportFromFd, &import, sizeof(import),
                        "import of the CUDA allocation");
            rm::ThirdPartyAlloc third_party;
            third_party.flags    = rm::kThirdPartyBar1;
            const rm::Handle p2p = rm::alloc(control_, client, subdevice, rm::kThirdPartyP2p,
                                             &third_party, sizeof(third_party), "third-party P2P");
            // A key per process and window, inside the user address range: every window of
            // this process gets its own 64 GiB of keys.
            static std::atomic<std::uint64_t> windows{0};
            key_ = (1ULL << 46) + ((std::uint64_t(::getpid()) % 64) << 40) +
                   ((windows.fetch_add(1) % 16) << 36);
            if (bytes_ > (1ULL << 36)) { refuse("a window holds at most 64 GiB"); }
            rm::RegisterVaSpace va_space;
            rm::control(control_, client, p2p, rm::kRegisterVaSpace, &va_space, sizeof(va_space),
                        "third-party P2P address space");
            rm::RegisterVidmem vidmem;
            vidmem.memory  = import.object;
            vidmem.address = key_;
            vidmem.size    = bytes_;
            rm::control(control_, client, p2p, rm::kRegisterVidmem, &vidmem, sizeof(vidmem),
                        "third-party P2P memory");
            rm::RegisterPid pid;
            pid.client = client;
            rm::control(control_, client, p2p, rm::kRegisterPid, &pid, sizeof(pid),
                        "third-party P2P process");
        } catch (...) {
            ::close(exported);
            throw;
        }
        ::close(exported);

        ninfer_pin_hold hold{};
        hold.abi   = NINFER_DIRECT_ABI;
        hold.key   = key_;
        hold.bytes = bytes_;
        if (::ioctl(pin_, NINFER_PIN_HOLD, &hold) < 0) {
            refuse(std::string("ninfer_pin: ") + std::strerror(errno));
        }
        if (hold.status != 0) {
            refuse("ninfer_pin: the NVIDIA driver did not pin the window in BAR1 (status " +
                   std::to_string(hold.status) +
                   "; is Resizable BAR on? BAR1 must hold the window)");
        }
        if (hold.contiguous == 0) {
            refuse("ninfer_pin: the window's BAR1 pages are not contiguous");
        }
        ninfer_p2p_map map{};
        map.abi        = NINFER_DIRECT_ABI;
        map.domain     = domain;
        map.bus_number = bus;
        map.devfn      = (slot << 3) | function;
        map.bus        = hold.bus;
        map.bytes      = bytes_;
        if (::ioctl(p2p_, NINFER_P2P_MAP, &map) < 0) {
            refuse(std::string("ninfer_p2p: ") + std::strerror(errno) +
                   (errno == EOPNOTSUPP ? " (a kernel without CONFIG_PCI_P2PDMA)" : ""));
        }
        void* mapping = ::mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, p2p_, 0);
        if (mapping == MAP_FAILED) {
            refuse(std::string("ninfer_p2p mmap: ") + std::strerror(errno));
        }
        mapping_ = static_cast<std::byte*>(mapping);
    }

    void open_sources(std::span<const std::filesystem::path> files) {
        for (std::size_t f = 0; f < files.size(); ++f) {
            const auto& path     = files[f];
            const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
            if (descriptor < 0) {
                refuse("cannot open " + path.string() + ": " + std::strerror(errno));
            }
            struct stat status{};
            Source source;
            try {
                if (::fstat(descriptor, &status) != 0) { refuse("cannot stat " + path.string()); }
                source.extents = file_extents(descriptor, path, sizes_[f]);
            } catch (...) {
                ::close(descriptor);
                throw;
            }
            ::close(descriptor);
            auto raw = raw_.find(status.st_dev);
            if (raw == raw_.end()) {
                const std::string device_path = block_device(status.st_dev);
                const int descriptor_raw =
                    ::open(device_path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
                if (descriptor_raw < 0) {
                    refuse("cannot open " + device_path + " to read " + path.string() + ": " +
                           std::strerror(errno) + (errno == EACCES ? " (join group disk)" : ""));
                }
                raw = raw_.emplace(status.st_dev, descriptor_raw).first;
                devices_ += (devices_.empty() ? "" : ", ") + device_path;
            }
            source.raw = raw->second;
            sources_.push_back(std::move(source));
        }
    }

    // In reverse order of opening: the mapping, the P2PDMA pages, the BAR1 pin, the resource
    // manager client and its registrations, then the CUDA allocation.
    void release() noexcept {
        if (mapping_ != nullptr) { ::munmap(mapping_, bytes_); }
        mapping_ = nullptr;
        for (int* descriptor : {&p2p_, &pin_, &control_, &gpu_}) {
            if (*descriptor >= 0) { ::close(*descriptor); }
            *descriptor = -1;
        }
        for (const auto& [number, descriptor] : raw_) { ::close(descriptor); }
        raw_.clear();
        if (mapped_) { (void)cuMemUnmap(address_, bytes_); }
        mapped_ = false;
        if (address_ != 0) { (void)cuMemAddressFree(address_, bytes_); }
        address_ = 0;
        if (allocation_ != 0) { (void)cuMemRelease(allocation_); }
        allocation_ = 0;
        if (context_ != nullptr) { (void)cuDevicePrimaryCtxRelease(cu_device_); }
        context_ = nullptr;
    }

    int device_                              = 0;
    CUdevice cu_device_                      = 0;
    CUcontext context_                       = nullptr;
    std::uint64_t bytes_                     = 0;
    std::uint64_t key_                       = 0;
    CUmemGenericAllocationHandle allocation_ = 0;
    CUdeviceptr address_                     = 0;
    bool mapped_                             = false;
    bool flush_                              = false;
    int pin_ = -1, p2p_ = -1, control_ = -1, gpu_ = -1;
    // The window's CPU mapping: only the drive's DMA writes it, the CPU never touches it.
    std::byte* mapping_ = nullptr;
    std::vector<std::uint64_t> sizes_;
    std::vector<Source> sources_;
    std::map<dev_t, int> raw_;
    std::string devices_;
    std::vector<ReadTask> tasks_;
    std::unique_ptr<HostWorkerPool> pool_;
};

#endif // __linux__

} // namespace

std::unique_ptr<DirectExpertWindow>
DirectExpertWindow::open(int device, std::span<const std::filesystem::path> files,
                         std::uint64_t bytes, bool host_window, Refusal& refusal,
                         std::string& why) {
    try {
        if (host_window) {
            auto window = std::make_unique<HostWindow>(files, bytes);
            self_test(*window, device, files, window->sizes(),
                      [&](std::uint64_t filled) { window->fill(filled); });
            return window;
        }
#if defined(__linux__)
        auto window = std::make_unique<ModuleWindow>(device, files, bytes);
        self_test(*window, device, files, window->sizes(),
                  [&](std::uint64_t filled) { window->fill(filled); });
        return window;
#else
        (void)device;
        (void)files;
        (void)bytes;
        throw Unavailable{Refusal::NotSetUp, "direct expert reads need Linux"};
#endif
    } catch (const Unavailable& unavailable) {
        refusal = unavailable.refusal;
        why     = unavailable.why;
    } catch (const std::exception& error) {
        refusal = Refusal::Failed;
        why     = error.what();
    }
    return nullptr;
}

std::unique_ptr<DirectExpertWindow> open_direct_window(const ExpertDiskReads& disk, int device,
                                                       std::span<const std::filesystem::path> files,
                                                       std::uint64_t bytes, const char* owner) {
    if (disk.direct == ExpertDirect::Off) { return nullptr; }
    auto refusal = DirectExpertWindow::Refusal::NotSetUp;
    std::string why;
    auto window = DirectExpertWindow::open(device, files, bytes, disk.host_window, refusal, why);
    if (window) {
        std::fprintf(stderr,
                     "ninfer: %s: direct expert reads into a %llu MiB window on GPU %d from %s\n",
                     owner, static_cast<unsigned long long>(window->bytes() >> 20), device,
                     window->describe().c_str());
        return window;
    }
    if (disk.direct == ExpertDirect::On) {
        throw std::runtime_error(std::string("--expert-direct on: ") + owner + ": " + why);
    }
    if (refusal == DirectExpertWindow::Refusal::Failed) {
        std::fprintf(
            stderr,
            "ninfer: %s: direct expert reads unavailable (%s); reading through host memory\n",
            owner, why.c_str());
    }
    return nullptr;
}

} // namespace ninfer

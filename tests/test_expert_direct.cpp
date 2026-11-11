// Direct expert reads (core/expert_direct.h): the cover arithmetic and the extent walk against
// worked cases, and the window's reads, copies and self-test through its host window, against the
// file's bytes. The kernel modules and the drive's DMA need hardware this test does not assume.
// Also the drive mirror's startup checks (core/expert_mirror.h).
#include "core/expert_direct.h"
#include "core/expert_mirror.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) { throw std::runtime_error(message); }
}

template <class Error, class Call>
void refused(Call&& call, const std::string& what) {
    try {
        call();
    } catch (const Error&) { return; }
    throw std::runtime_error("accepted " + what);
}

void covers() {
    using ninfer::direct_cover;
    using ninfer::direct_lead;
    require(direct_lead(0) == 0 && direct_lead(4095) == 4095 && direct_lead(4096) == 0 &&
                direct_lead(8193) == 1,
            "direct_lead");
    require(direct_cover(0, 0) == 0, "an empty range at a block start takes no block");
    require(direct_cover(0, 1) == 4096 && direct_cover(0, 4096) == 4096 &&
                direct_cover(0, 4097) == 8192,
            "covers from a block start");
    require(direct_cover(4095, 1) == 4096 && direct_cover(4095, 2) == 8192 &&
                direct_cover(1, 8191) == 8192 && direct_cover(1, 8192) == 12288,
            "covers from inside a block");
}

void device_runs() {
    using ninfer::FileExtent;
    // Three extents; the second and third lie next to each other on the device.
    const std::vector<FileExtent> extents{{0, 1 << 20, 64 << 10},
                                          {64 << 10, 5 << 20, 128 << 10},
                                          {192 << 10, (5 << 20) + (128 << 10), 64 << 10}};
    auto runs = ninfer::direct_device_runs(extents, 4096, 8192);
    require(runs.size() == 1 && runs[0].logical == 4096 && runs[0].physical == (1 << 20) + 4096 &&
                runs[0].bytes == 8192,
            "a range inside one extent");
    runs = ninfer::direct_device_runs(extents, (64 << 10) - 4096, 8192);
    require(runs.size() == 2 && runs[0].physical == (1 << 20) + (60 << 10) &&
                runs[0].bytes == 4096 && runs[1].logical == 64 << 10 &&
                runs[1].physical == 5 << 20 && runs[1].bytes == 4096,
            "a range across two extents apart on the device");
    runs = ninfer::direct_device_runs(extents, (128 << 10), 128 << 10);
    require(runs.size() == 1 && runs[0].physical == (5 << 20) + (64 << 10) &&
                runs[0].bytes == 128 << 10,
            "a range across two adjacent extents is one run");
    require(ninfer::direct_device_runs(extents, 0, 0).empty(), "an empty range has no run");
    refused<std::runtime_error>([&] { (void)ninfer::direct_device_runs(extents, 250 << 10, 8192); },
                                "a range past the extents");
    const std::vector<FileExtent> holed{{0, 0, 4096}, {8192, 4096, 4096}};
    refused<std::runtime_error>([&] { (void)ninfer::direct_device_runs(holed, 0, 12288); },
                                "a range across a hole");
}

struct Fixture {
    std::vector<std::filesystem::path> paths;
    std::vector<std::vector<unsigned char>> bytes;

    Fixture() {
        std::mt19937 random(4241);
        const auto prefix = std::filesystem::current_path() /
                            ("ninfer_direct_" + std::to_string(std::random_device{}()));
        // A file past several blocks with a short last block, and a file under one block.
        for (const std::size_t size : {std::size_t(3 << 20) + 123, std::size_t(1000)}) {
            paths.push_back(prefix.string() + "." + std::to_string(paths.size()));
            bytes.emplace_back(size);
            for (auto& byte : bytes.back()) { byte = static_cast<unsigned char>(random()); }
            std::ofstream file(paths.back(), std::ios::binary);
            file.write(reinterpret_cast<const char*>(bytes.back().data()),
                       std::streamsize(bytes.back().size()));
            require(bool(file), "cannot write the fixture");
        }
    }

    ~Fixture() {
        for (const auto& path : paths) {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    }
};

void host_window() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP host window: no CUDA device\n";
        return;
    }
    Fixture fixture;
    ninfer::ExpertDiskReads disk;
    disk.direct      = ninfer::ExpertDirect::On;
    disk.host_window = true;
    auto window      = ninfer::open_direct_window(disk, 0, fixture.paths, (1 << 20) + 5, "test");
    require(window != nullptr, "no host window");
    require(window->bytes() >= (1 << 20) + 5 && window->bytes() % (64 << 10) == 0,
            "the window is not whole 64 KiB pages");

    // Starts at, inside and at the end of blocks, the file's last bytes, and a whole small file.
    struct Range {
        std::size_t file;
        std::uint64_t offset, bytes;
    };

    const auto size0 = fixture.bytes[0].size();
    const std::vector<Range> ranges{{0, 0, 4096},
                                    {0, 1, 4095},
                                    {0, 4095, 2},
                                    {0, 4096, 100000},
                                    {0, 777777, 333333},
                                    {0, size0 - 5, 5},
                                    {0, size0 - 70000, 70000},
                                    {1, 0, 1000},
                                    {1, 999, 1}};
    std::vector<ninfer::DirectRead> reads;
    std::uint64_t at = 0;
    for (const Range& range : ranges) {
        reads.push_back({range.file, range.offset, range.bytes, at});
        at += ninfer::direct_cover(range.offset, range.bytes);
    }
    require(at <= window->bytes(), "the test's covers exceed the window");
    for (int repeat = 0; repeat < 2; ++repeat) {
        window->read(reads);
        for (const auto& read : reads) {
            std::vector<unsigned char> back(read.bytes);
            require(
                cudaMemcpy(back.data(),
                           window->data() + read.window_offset + ninfer::direct_lead(read.offset),
                           back.size(), cudaMemcpyDefault) == cudaSuccess,
                "copy from the window");
            const auto& file = fixture.bytes[read.file];
            require(
                std::equal(back.begin(), back.end(), file.begin() + std::ptrdiff_t(read.offset)),
                "window bytes differ from file " + std::to_string(read.file) + " at " +
                    std::to_string(read.offset));
        }
    }
    refused<std::invalid_argument>(
        [&] {
            const ninfer::DirectRead outside{0, 0, 4096, window->bytes()};
            window->read({&outside, 1});
        },
        "a read past the window");
    refused<std::invalid_argument>(
        [&] {
            const ninfer::DirectRead unaligned{0, 0, 4096, 512};
            window->read({&unaligned, 1});
        },
        "a window offset inside a block");
    refused<std::invalid_argument>(
        [&] {
            const ninfer::DirectRead past{1, 900, 101, 0};
            window->read({&past, 1});
        },
        "a read past its file");
    std::cout << "host window: " << reads.size() << " reads PASS\n";
}

void modes() {
    Fixture fixture;
    ninfer::ExpertDiskReads disk;
    disk.direct = ninfer::ExpertDirect::Off;
    require(ninfer::open_direct_window(disk, 0, fixture.paths, 1 << 20, "test") == nullptr,
            "off opened a window");
    ninfer::DirectExpertWindow::Refusal refusal{};
    std::string why;
    auto window = ninfer::DirectExpertWindow::open(0, fixture.paths, 1 << 20, false, refusal, why);
    if (window) {
        std::cout << "SKIP modes: this machine has the direct path (" << window->describe()
                  << ")\n";
        return;
    }
    require(!why.empty(), "a refusal without a reason");
    disk.direct = ninfer::ExpertDirect::Auto;
    require(ninfer::open_direct_window(disk, 0, fixture.paths, 1 << 20, "test") == nullptr,
            "auto opened a window that open() refused");
    disk.direct = ninfer::ExpertDirect::On;
    try {
        (void)ninfer::open_direct_window(disk, 0, fixture.paths, 1 << 20, "test");
        throw std::logic_error("on without a direct path did not fail");
    } catch (const std::runtime_error& error) {
        require(std::string(error.what()).find(why) != std::string::npos,
                "on's error does not give the reason");
    }
    std::cout << "modes: " << why << " PASS\n";
}

// The mirror's startup checks: copies of every file pass; a missing copy, a copy of another size,
// a copy whose first or last byte differs, and a directory that is a file are refused.
void mirror_checks() {
    Fixture fixture;
    const std::filesystem::path directory = fixture.paths[0].string() + ".mirror";
    std::filesystem::create_directories(directory);

    struct Cleanup {
        std::filesystem::path path;

        ~Cleanup() {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }
    } cleanup{directory};

    const auto copy_of = [&](std::size_t i) { return directory / fixture.paths[i].filename(); };
    const auto write   = [&](std::size_t i, const std::vector<unsigned char>& bytes) {
        std::ofstream file(copy_of(i), std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
        require(bool(file), "cannot write a mirror copy");
    };
    const auto mirror = [&](const std::filesystem::path& at) {
        return ninfer::expert_mirror_files(fixture.paths, at);
    };

    require(mirror({}).empty(), "no mirror directory gave copies");
    refused<std::invalid_argument>([&] { (void)mirror(fixture.paths[1]); },
                                   "a mirror directory that is a file");
    write(0, fixture.bytes[0]);
    refused<std::invalid_argument>([&] { (void)mirror(directory); }, "a missing copy");
    write(1, fixture.bytes[1]);
    const auto copies = mirror(directory);
    require(copies.size() == 2 && copies[0] == copy_of(0) && copies[1] == copy_of(1),
            "the mirror's copies");

    auto changed = fixture.bytes[0];
    changed.pop_back();
    write(0, changed);
    refused<std::invalid_argument>([&] { (void)mirror(directory); }, "a copy of another size");
    // The large file's first and last byte; the small file is compared whole.
    const std::pair<std::size_t, std::size_t> differences[] = {
        {0, 0}, {0, fixture.bytes[0].size() - 1}, {1, fixture.bytes[1].size() / 2}};
    for (const auto& [file, at] : differences) {
        write(0, fixture.bytes[0]);
        write(1, fixture.bytes[1]);
        changed = fixture.bytes[file];
        changed[at] ^= 1;
        write(file, changed);
        refused<std::invalid_argument>([&] { (void)mirror(directory); },
                                       "a copy that differs at byte " + std::to_string(at));
    }
    std::cout << "mirror checks PASS\n";
}

} // namespace

int main() {
    try {
        covers();
        device_runs();
        host_window();
        modes();
        mirror_checks();
        std::cout << "PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

// DeviceBuffer's host-to-device copy and fill must be visible to a non-blocking stream.
//
// Both are implemented with the synchronous CUDA runtime calls -- `cudaMemcpy` and `cudaMemset` --
// whose names suggest the work is finished when they return. Neither is. CUDA documents that a
// host-to-device copy out of *pageable* memory returns once the source has been staged for DMA
// "but the DMA to final destination may not have completed", and that a device memset is
// asynchronous with respect to the host. Both complete on the legacy stream, and every stream
// `DeviceContext` owns is created with `cudaStreamNonBlocking`, which is exempt from the legacy
// stream's implicit ordering.
//
// So without an explicit completion the sequence "stage inputs, then launch on the engine stream"
// is a race, and it is the sequence every Op test's setup uses. It cost real time: an
// `attn_input_proj` case at `T=112` failed in roughly a third of full-suite runs, always in the
// graph-replay phase that re-uploads its activation, always on every output at once -- the
// signature of one shared input being read early -- and never in isolation, because an idle box
// lands the DMA in time.
//
// This test reproduces that sequence directly and asserts the outcome, rather than asserting the
// implementation. Against an unsynchronized DeviceBuffer it reads stale data in roughly 90 of
// every 100 iterations on an idle RTX 3090.
//
// The window is narrow and easy to close by accident: an earlier draft cleared the counter with
// DeviceBuffer::fill between the copy and the stream work, and that single synchronous runtime
// call in the gap was enough to make the race vanish entirely. Keep the gap empty.
//
// The `fill` half below asserts the same contract for cudaMemset, which is documented as
// asynchronous with respect to the host in the same way. It has never actually caught a stale
// read on this driver -- the memset appears to complete before the call returns -- so treat it as
// pinning the contract rather than as a reproduction.
//
// The last case makes the window wide on purpose: a kernel holds the legacy stream for 50 ms, so a
// fill_and_wait (cudaMemset) or a device-to-device copy_and_wait (cudaMemcpy) queued behind it
// lands after the caller's next work on a non-blocking stream unless the call waits for the legacy
// stream. The first is the weight loader's sequence: zero a placement, then copy the loaded bytes
// into it on a transfer stream.

#include "core/device_buffer.h"
#include "core/device_launch.h"
#include "core/device_wait.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace ninfer;

namespace {

// Counts elements that still hold the value written before the copy under test.
__global__ void count_stale(const std::uint16_t* data, int count, std::uint16_t stale_value,
                            int* out) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    if (data[index] == stale_value) atomicAdd(out, 1);
}

// Occupies the stream it runs on for `nanoseconds`.
__global__ void hold(unsigned long long nanoseconds) {
    unsigned long long start = 0, now = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    do {
        __nanosleep(10000);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    } while (now - start < nanoseconds);
}

// Sized as the case that exposed this: 112 tokens of a 5,120-wide BF16 activation, 1,146,880
// bytes. Copies at or below ~256 KiB were never observed to race on an RTX 3090; every size above
// it did, so a small buffer here would pass whether the bug is present or not.
constexpr int kElements = 112 * 5120;
constexpr std::uint16_t kStale = 0xAAAA;
constexpr std::uint16_t kFresh = 0xBBBB;

// One iteration wins the race far more often than it loses even when the bug is present. Ninety
// reproduced it in every trial while developing this; the observed rate was above 70%.
constexpr int kIterations = 90;

int check(cudaError_t status, const char* what) {
    if (status == cudaSuccess) return 0;
    std::printf("FAIL %s: %s\n", what, cudaGetErrorString(status));
    return 1;
}

// fill_and_wait and a device-to-device copy_and_wait behind a held legacy stream: on return their
// bytes must be in place for work on `stream`. Returns the number of failed checks.
int held_legacy_stream(cudaStream_t stream) {
    constexpr unsigned long long kHold = 50'000'000;
    const std::size_t bytes = sizeof(std::uint16_t) * kElements;
    DeviceBuffer payload(bytes), fresh(bytes), counter(sizeof(int));
    std::vector<std::uint16_t> got(kElements);
    int failures = check(cudaMemset(fresh.p, 0xBB, bytes), "fresh") +
                   check(cudaMemset(payload.p, 0xAA, bytes), "stale") +
                   check(cudaDeviceSynchronize(), "settle held");
    if (failures != 0) return failures;

    // Zero the payload, then copy the fresh bytes over it on the stream: the zeros must not land
    // last.
    hold<<<1, 1>>>(kHold);
    failures += check(cudaGetLastError(), "hold before fill");
    failures += check(ninfer::fill_and_wait(payload.p, 0, bytes), "fill_and_wait");
    failures += check(cudaMemcpyAsync(payload.p, fresh.p, bytes, cudaMemcpyDeviceToDevice, stream),
                      "stream copy after the fill");
    failures += check(cudaStreamSynchronize(stream), "stream sync after the fill");
    failures += check(cudaDeviceSynchronize(), "settle after the fill");
    payload.copy_to_host(got.data(), bytes);
    int lost = 0;
    for (const std::uint16_t value : got) { lost += value != kFresh ? 1 : 0; }
    if (lost != 0) {
        std::printf("FAIL fill_and_wait behind a held legacy stream: %d of %d elements lost the "
                    "stream's later copy\n", lost, kElements);
        ++failures;
    }

    // Copy the fresh bytes over stale ones, then read them on the stream.
    failures += check(cudaMemset(payload.p, 0xAA, bytes), "stale again") +
                check(cudaMemset(counter.p, 0, sizeof(int)), "counter") +
                check(cudaDeviceSynchronize(), "settle before the copy");
    hold<<<1, 1>>>(kHold);
    failures += check(cudaGetLastError(), "hold before copy");
    failures += check(ninfer::copy_and_wait(payload.p, fresh.p, bytes, cudaMemcpyDeviceToDevice),
                      "copy_and_wait");
    count_stale<<<(kElements + 255) / 256, 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(payload.p), kElements, kStale,
        static_cast<int*>(counter.p));
    failures += check(cudaGetLastError(), "count after the copy");
    failures += check(cudaStreamSynchronize(stream), "stream sync after the copy");
    failures += check(cudaDeviceSynchronize(), "settle after the copy");
    int stale = 0;
    counter.copy_to_host(&stale, sizeof(stale));
    if (stale != 0) {
        std::printf("FAIL copy_and_wait behind a held legacy stream: the stream read %d of %d "
                    "elements stale\n", stale, kElements);
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }

    // A non-blocking stream, exactly as DeviceContext creates.
    cudaStream_t stream = nullptr;
    if (check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create"))
        return 1;

    DeviceBuffer payload(sizeof(std::uint16_t) * kElements);
    DeviceBuffer counter(sizeof(int));
    // Poison targets standing in for the outputs an Op test clears before launching.
    DeviceBuffer poison_a(sizeof(std::uint16_t) * 4096 * 112);
    DeviceBuffer poison_b(sizeof(std::uint16_t) * 1024 * 112);

    const std::vector<std::uint16_t> stale_host(kElements, kStale);
    const std::vector<std::uint16_t> fresh_host(kElements, kFresh);

    int copy_races = 0, fill_races = 0, worst = 0;
    for (int iteration = 0; iteration < kIterations; ++iteration) {
        payload.copy_from_host(stale_host.data(), payload.bytes);
        counter.fill(0);
        if (check(cudaDeviceSynchronize(), "settle")) return 1;

        // The copy under test, followed immediately by work on the non-blocking stream. Nothing
        // may sit between the two: a synchronous runtime call in the gap closes the window the
        // test exists to open.
        payload.copy_from_host(fresh_host.data(), payload.bytes);
        if (check(cudaMemsetAsync(poison_a.p, 0xff, poison_a.bytes, stream), "poison a")) return 1;
        if (check(cudaMemsetAsync(poison_b.p, 0xff, poison_b.bytes, stream), "poison b")) return 1;
        count_stale<<<(kElements + 255) / 256, 256, 0, stream>>>(
            static_cast<const std::uint16_t*>(payload.p), kElements, kStale,
            static_cast<int*>(counter.p));
        if (check(cudaGetLastError(), "launch")) return 1;
        if (check(cudaStreamSynchronize(stream), "stream sync")) return 1;

        int stale = 0;
        counter.copy_to_host(&stale, sizeof(stale));
        if (stale > 0) {
            ++copy_races;
            if (stale > worst) worst = stale;
        }

        // The same question for fill(): clear the payload, then read it on the stream. Every
        // element must now be zero, so any element still holding kFresh is a lost memset. The
        // counter is cleared first, for the same reason as above.
        counter.fill(0);
        if (check(cudaDeviceSynchronize(), "settle fill")) return 1;
        payload.fill(0);
        count_stale<<<(kElements + 255) / 256, 256, 0, stream>>>(
            static_cast<const std::uint16_t*>(payload.p), kElements, kFresh,
            static_cast<int*>(counter.p));
        if (check(cudaGetLastError(), "fill launch")) return 1;
        if (check(cudaStreamSynchronize(stream), "fill stream sync")) return 1;
        counter.copy_to_host(&stale, sizeof(stale));
        if (stale > 0) ++fill_races;
    }

    const int held_failures = held_legacy_stream(stream);
    (void)cudaStreamDestroy(stream);
    if (held_failures != 0) return 1;

    if (copy_races != 0 || fill_races != 0) {
        std::printf("FAIL device_buffer_visibility: over %d iterations, copy_from_host was read "
                    "stale %d times (worst %d of %d elements) and fill %d times\n",
                    kIterations, copy_races, worst, kElements, fill_races);
        return 1;
    }
    std::printf("OK device_buffer_visibility (%d iterations at %d bytes)\n", kIterations,
                int(sizeof(std::uint16_t) * kElements));
    return 0;
}

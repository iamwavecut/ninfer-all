#pragma once

// CPU workers for routed experts computed on the host while the GPU runs (every MoE family's
// host and disk experts). Workers spin between spin(true) and spin(false), so a decode call's
// products start without a wake-up, and sleep otherwise. run(count, job) executes job(i) for every
// i < count on the workers and the calling thread and returns when all are done, rethrowing the
// first failure; run(count, job, side) first runs side() on the calling thread while the workers
// start the jobs (a call issues its copies there). A claim is one fetch_add on a ticket holding the
// run's generation and next index; an index past the generation's count, or a generation the run
// size no longer names, ends the claiming thread's part, so a worker still leaving an earlier run
// either runs a later run's job in full or none. Each thread adds the jobs it ran to the done count
// once, when its part ends.
//
// The waiting workers, the ticket, the done count, the completion and the run's job each have a
// cache line of their own, so neither a claim nor a count invalidates the line the waiting workers
// read. On two 32-core AMD EPYC sockets, spinning pools of 2, 8, 16, 32 and 61 threads took 0.86,
// 1.43, 2.42, 4.82 and 12.77 us for a run of 1 to 50 empty jobs; with every claim counting on one
// shared line, 1.09, 1.91, 3.38, 7.67 and 14.63 us.
//
// row_tasks() cuts a phase's rows into about three tasks per thread of about equal work, as
// Strata's pool does: with fixed 64-row tasks one Flash-Next expert's 640 gate rows were ten
// tasks, so a call of one or two CPU experts left most of a 15-thread pool idle.
//
// The expert products read stored weights at memory speed, which one thread per physical core
// saturates: an SMT sibling shares its core's loads. The workers are pinned one to a physical core
// of the process's CPUs (its affinity mask), the calling thread pins itself to the first of them
// when it first runs (unless run() is told not to: a thread that drives the GPU and computes a
// call's share itself stays where it runs), and pool_cpu_count() bounds the default by half the
// CPUs the process may use, the rest left to the threads that drive the GPU and serve requests:
// on two EPYC 7742 sockets with a 61-CPU quota, a pool spinning on every CPU of the quota
// throttled the process in 208 of 211 scheduling periods while it decoded, and 59 threads decoded
// Flash-Next 1.7-4.2% slower than 8, 16, 24 or 32. A thread that drives the GPU pinned to the
// first core shares it with the spinning service thread: Flash-Next's wide calls pinned the
// executor's thread that way, and eight requests on host experts decoded 8 instead of 95 tokens
// per second on an RTX A6000.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <span>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace ninfer {

// Logical CPUs this process may keep busy: std::thread's count, bounded on Linux by the affinity
// mask and by a container's CPU quota (cgroup v2 cpu.max, v1 cpu.cfs_quota_us).
[[nodiscard]] unsigned usable_cpus();

// One logical CPU of each physical core this process may run on, in CPU order (Linux sysfs
// topology, Windows processor information; every allowed CPU where neither is readable).
[[nodiscard]] std::vector<int> physical_core_cpus();

// The default thread count of a CPU expert pool, the calling thread included: one per physical
// core, at most half of usable_cpus(), at least one.
[[nodiscard]] std::uint32_t pool_cpu_count();

// One turn of a spin wait: the processor's pause hint where it has one, a yield elsewhere.
void spin_hint() noexcept;

// A phase's rows of one part (an expert's projection) and one row's relative work.
struct RowPart {
    int rows    = 0;
    double cost = 1.0;
};

// Rows [begin, end) of part `part`.
struct RowTask {
    std::uint32_t part = 0;
    int begin = 0, end = 0;
};

// The parts' rows as about three tasks per thread of about equal work; a task stays in its part
// and starts at a multiple of `quantum` rows.
[[nodiscard]] std::vector<RowTask> row_tasks(std::span<const RowPart> parts, std::uint32_t threads,
                                             int quantum);

class CpuSpinPool {
public:
    // `threads` counts the calling thread: threads - 1 workers. `pin`: one physical core each.
    CpuSpinPool(std::uint32_t threads, bool pin);
    ~CpuSpinPool();
    CpuSpinPool(const CpuSpinPool&)            = delete;
    CpuSpinPool& operator=(const CpuSpinPool&) = delete;

    [[nodiscard]] std::uint32_t threads() const noexcept { return std::uint32_t(workers_.size()) + 1; }
    [[nodiscard]] bool pinned() const noexcept { return pinned_; }

    // When the latest run()'s last job ended (steady clock), on whichever thread ran it; stamped
    // before run() returns.
    [[nodiscard]] std::chrono::steady_clock::time_point finished() const noexcept {
        return std::chrono::steady_clock::time_point(
            std::chrono::steady_clock::duration(complete_.finished.load(std::memory_order_acquire)));
    }

    void spin(bool on);

    template <class Job>
    void run(std::size_t count, Job&& job) {
        run(count, std::forward<Job>(job), [] {});
    }

    // `pin`: the calling thread takes the pool's first core (the thread that serves the pool's
    // requests); false leaves it where it runs, for a thread that also drives the GPU.

    template <class Job, class Side>
    void run(std::size_t count, Job&& job, Side&& side, bool pin = true) {
        if (count == 0) {
            side();
            complete_.finished = std::chrono::steady_clock::now().time_since_epoch().count();
            return;
        }
        if (pin) { pin_caller(); }
        job_.context = &job;
        job_.call    = [](void* context, std::size_t i) {
            (*static_cast<std::remove_reference_t<Job>*>(context))(i);
        };
        const std::uint64_t generation = wake_line_.posted.load(std::memory_order_relaxed) + 1;
        done_.store(0, std::memory_order_relaxed);
        // A claim runs a job only when the run size names the ticket's generation.
        job_.size.store(generation << 32 | count, std::memory_order_release);
        ticket_.store(generation << 32, std::memory_order_release);
        {
            std::lock_guard lock(mutex_);
            wake_line_.posted.store(generation, std::memory_order_release);
        }
        wake_.notify_all();
        // The jobs reference the caller's frame: a failing side still waits for them.
        std::exception_ptr side_failure;
        try {
            side();
        } catch (...) { side_failure = std::current_exception(); }
        drain();
        // The thread whose part ends the run stamps its end and then marks the generation
        // complete.
        while (complete_.generation.load(std::memory_order_acquire) != generation) { spin_hint(); }
        auto failure = std::exchange(failure_, nullptr);
        if (side_failure) { std::rethrow_exception(side_failure); }
        if (failure) { std::rethrow_exception(failure); }
    }

private:
    void drain();
    // Adds a part's `ran` jobs of the run `size` names; the part that ends the run stamps it.
    void finish(std::uint64_t size, std::size_t ran);
    void work(int cpu);
    void pin_caller();

    std::vector<std::thread> workers_;
    int caller_cpu_     = -1;
    bool pinned_        = false;
    std::thread::id caller_;
    // What the waiting workers read: written once per run and by spin() and the destructor.
    struct alignas(64) WakeLine {
        std::atomic<std::uint64_t> posted{0}; // the latest run's generation
        std::atomic<bool> spinning{false}, stop{false};
    } wake_line_;
    // The run's job, read by every claim.
    struct alignas(64) JobLine {
        void (*call)(void*, std::size_t) = nullptr;
        void* context                    = nullptr;
        std::atomic<std::uint64_t> size{0}; // generation << 32 | count
    } job_;
    alignas(64) std::atomic<std::uint64_t> ticket_{0}; // generation << 32 | next index
    alignas(64) std::atomic<std::size_t> done_{0};
    struct alignas(64) CompleteLine {
        std::atomic<std::uint64_t> generation{0}; // the generation whose last job has ended
        std::atomic<std::chrono::steady_clock::rep> finished{0};
    } complete_;
    alignas(64) std::mutex mutex_;
    std::mutex failure_mutex_;
    std::condition_variable wake_;
    std::exception_ptr failure_;
};

} // namespace ninfer

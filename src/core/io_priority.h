#pragma once

// Windows I/O priority. A process started by a low-priority parent (a Task Scheduler task at its
// default priority, a service, some IDE and agent terminals) inherits I/O priority Low although its
// CPU class is Normal, and Windows throttles its reads: the artifact load, the n-gram and Engram
// rows, disk experts and the disk KV tier.

#include <optional>

namespace ninfer {

// Windows IO_PRIORITY_HINT: very low 0, low 1, normal 2.
inline constexpr int kIoPriorityNormal = 2;

struct IoPriorityRaise {
    int before = kIoPriorityNormal; // the inherited priority
    int after  = kIoPriorityNormal; // the priority now; `before` when Windows refused the raise
};

// Raises the calling process's I/O priority to normal when it is lower, once per process: every
// call returns the first call's outcome. nullopt when the priority was normal or higher, could not
// be read, or the platform has no such priority (every build but Windows).
[[nodiscard]] std::optional<IoPriorityRaise> raise_low_io_priority() noexcept;

} // namespace ninfer

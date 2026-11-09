#include "core/tensor_split.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <stdexcept>

namespace ninfer {
namespace {

void require_positive(std::span<const double> values, const char* what) {
    for (const double value : values) {
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::invalid_argument(std::string(what) + " must be positive and finite");
        }
    }
}

} // namespace

std::vector<std::uint64_t> split_shares(std::uint64_t total, std::span<const std::uint64_t> allowed,
                                        std::span<const double> weights) {
    const std::size_t ranks = weights.size();
    if (ranks == 0) { throw std::invalid_argument("a split needs at least one rank"); }
    require_positive(weights, "tensor-split weights");
    for (std::size_t i = 0; i < allowed.size(); ++i) {
        if (allowed[i] == 0 || allowed[i] >= total || (i > 0 && allowed[i] <= allowed[i - 1])) {
            throw std::invalid_argument("share ends must ascend strictly inside the dimension");
        }
    }
    if (allowed.size() + 1 < ranks) {
        throw std::invalid_argument("a dimension of " + std::to_string(total) + " divides into " +
                                    std::to_string(allowed.size() + 1) +
                                    " shares at most, fewer than the " + std::to_string(ranks) +
                                    " ranks");
    }
    const double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    std::vector<std::uint64_t> out{0};
    out.reserve(ranks + 1);
    double cumulative  = 0.0;
    std::size_t lowest = 0; // the first position the next end may take
    for (std::size_t rank = 0; rank + 1 < ranks; ++rank) {
        cumulative += weights[rank];
        const double target = static_cast<double>(total) * (cumulative / sum);
        // Leave one position for each later inner end.
        const std::size_t highest = allowed.size() - (ranks - 1 - rank);
        const auto first          = allowed.begin() + static_cast<std::ptrdiff_t>(lowest);
        const auto last           = allowed.begin() + static_cast<std::ptrdiff_t>(highest) + 1;
        auto above                = std::lower_bound(first, last, target, [](std::uint64_t a, double t) {
            return static_cast<double>(a) < t;
        });
        auto pick = above;
        if (above == last) {
            pick = above - 1;
        } else if (above != first) {
            const auto below = above - 1;
            if (target - static_cast<double>(*below) <= static_cast<double>(*above) - target) {
                pick = below;
            }
        }
        out.push_back(*pick);
        lowest = static_cast<std::size_t>(pick - allowed.begin()) + 1;
    }
    out.push_back(total);
    return out;
}

std::vector<std::uint64_t> unit_positions(std::uint64_t total, std::uint64_t unit) {
    if (unit == 0) { throw std::invalid_argument("a share unit must be positive"); }
    std::vector<std::uint64_t> out;
    for (std::uint64_t position = unit; position < total; position += unit) {
        out.push_back(position);
    }
    return out;
}

std::vector<double> shares_beside_first(std::span<const double> speed, double first) {
    if (speed.empty()) { throw std::invalid_argument("shares need one speed per rank"); }
    require_positive(speed, "rank speeds");
    if (!std::isfinite(first) || first < 0.0) {
        throw std::invalid_argument("rank 0's own work must be finite and not negative");
    }
    // Rank r runs out[r] (+ first on rank 0) in out[r] / speed[r]: one level for every rank.
    const double sum = std::accumulate(speed.begin(), speed.end(), 0.0);
    std::vector<double> out(speed.size());
    const double level = (1.0 + first) / sum;
    out[0]             = level * speed[0] - first;
    if (out[0] > 0.0 || speed.size() == 1) {
        for (std::size_t rank = 1; rank < speed.size(); ++rank) { out[rank] = level * speed[rank]; }
    } else {
        const double rest = sum - speed[0];
        for (std::size_t rank = 1; rank < speed.size(); ++rank) { out[rank] = speed[rank] / rest; }
    }
    out[0] = std::max(out[0], 1e-6);
    return out;
}

std::vector<double> fill_shares(std::span<const double> speed, std::span<const double> capacity,
                                bool shrinks) {
    const std::size_t ranks = speed.size();
    if (ranks == 0 || capacity.size() != ranks) {
        throw std::invalid_argument("shares need one speed and one capacity per rank");
    }
    require_positive(speed, "rank speeds");
    for (const double room : capacity) {
        if (!std::isfinite(room) || room < 0.0) {
            throw std::invalid_argument("rank capacities must be finite and not negative");
        }
    }
    const double speed_sum = std::accumulate(speed.begin(), speed.end(), 0.0);
    const double room      = std::accumulate(capacity.begin(), capacity.end(), 0.0);
    std::vector<double> out(ranks);
    if (room < 1.0) {
        const bool every_rank_has_room =
            std::all_of(capacity.begin(), capacity.end(), [](double c) { return c > 0.0; });
        for (std::size_t rank = 0; rank < ranks; ++rank) {
            out[rank] =
                shrinks && every_rank_has_room ? capacity[rank] / room : speed[rank] / speed_sum;
        }
        return out;
    }
    std::vector<bool> full(ranks, false);
    for (;;) {
        double remaining  = 1.0;
        double open_speed = 0.0;
        for (std::size_t rank = 0; rank < ranks; ++rank) {
            if (full[rank]) {
                remaining -= out[rank];
            } else {
                open_speed += speed[rank];
            }
        }
        bool capped = false;
        for (std::size_t rank = 0; rank < ranks; ++rank) {
            if (full[rank]) { continue; }
            out[rank] = remaining * speed[rank] / open_speed;
            if (out[rank] > capacity[rank]) {
                out[rank]  = capacity[rank];
                full[rank] = true;
                capped     = true;
            }
        }
        if (!capped) { return out; }
    }
}

std::vector<double> delivery_shares(std::span<const double> room, std::span<const double> bus,
                                    double bytes, std::span<const double> speed) {
    const std::size_t ranks = room.size();
    if (ranks == 0 || bus.size() != ranks || speed.size() != ranks) {
        throw std::invalid_argument("delivery shares need one room, bus and speed per rank");
    }
    if (!std::isfinite(bytes) || bytes <= 0.0) {
        throw std::invalid_argument("delivery shares need the experts' bytes");
    }
    require_positive(bus, "rank bus speeds");
    for (const double held : room) {
        if (!std::isfinite(held) || held < 0.0) {
            throw std::invalid_argument("rank cache rooms must be finite and not negative");
        }
    }
    const double held = std::accumulate(room.begin(), room.end(), 0.0);
    if (held >= bytes) {
        std::vector<double> capacity(ranks);
        for (std::size_t rank = 0; rank < ranks; ++rank) { capacity[rank] = room[rank] / bytes; }
        return fill_shares(speed, capacity);
    }
    // Rank r copies share[r] * bytes - room[r]; equal times over the buses leave each rank its room
    // plus `time` seconds of its bus, and the shares add up to every expert.
    const double time = (bytes - held) / std::accumulate(bus.begin(), bus.end(), 0.0);
    std::vector<double> out(ranks);
    for (std::size_t rank = 0; rank < ranks; ++rank) {
        out[rank] = (room[rank] + time * bus[rank]) / bytes;
    }
    return out;
}

} // namespace ninfer

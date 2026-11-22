#include "serve/serve_metrics.h"

#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace ninfer::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

std::map<std::string, double> samples(const std::string& body) {
    std::map<std::string, double> out;
    std::istringstream lines(body);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.empty() || line[0] == '#') { continue; }
        const auto space = line.find(' ');
        out[line.substr(0, space)] = std::stod(line.substr(space + 1));
    }
    return out;
}

} // namespace

int main() {
    int failures = 0;
    LoadCapacity capacity;
    capacity.max_concurrency    = 4;
    capacity.kv_capacity_tokens = 131072;
    capacity.kv_capacity_pages  = 2048;

    LoadSample sample;
    sample.uptime_seconds                      = 30.0;
    sample.admitted_requests                   = 5;
    sample.stats.computed_prefill_tokens       = 9000;
    sample.stats.committed_decode_tokens       = 80;
    sample.stats.prefill_seconds_total         = 3.0;
    sample.stats.decode_seconds_total          = 4.0;
    sample.stats.decode_rounds                 = 10;
    sample.stats.decode_row_rounds             = 25;
    sample.stats.device_main_kv_occupied_pages = 512;
    sample.stats.running_requests              = 3;
    sample.stats.waiting_requests              = 2;
    sample.stats.reused_prompt_tokens          = 700;
    sample.stats.engine_recoveries             = 1;
    sample.stats.root_selections               = 3;
    sample.stats.private_endpoint_selections   = 9;
    sample.stats.pressure_checkpoints_dropped  = 4;
    sample.stats.main_kv_h2d_bytes             = 1 << 20;
    sample.stats.host_state_occupied_slots     = 6;
    sample.stats.waiting_expired_requests      = 2;
    sample.stats.cancelled_prefills_salvaged   = 1;
    sample.stats.paced_prefill_steps           = 3;
    sample.stats.paced_prefill_deferred_tokens = 4608;

    ServeMetrics metrics;
    GenerationOutcome outcome;
    outcome.completion_tokens                   = 40;
    outcome.metrics.prefill_seconds             = 100.0; // per-request wall time is not a rate input
    outcome.metrics.decode_seconds              = 100.0;
    outcome.metrics.prefix_cache_hit_tokens     = 300;
    outcome.metrics.speculative_draft_tokens    = 30;
    outcome.metrics.speculative_accepted_tokens = 21;
    metrics.record_done(outcome);
    metrics.record_done(outcome);
    metrics.record_failure();
    metrics.record_rejection();
    metrics.record_rejection();

    const std::string body = metrics.render(capacity, sample);
    const auto values      = samples(body);
    failures += check(body.find("# TYPE llamacpp:prompt_tokens_total counter") != std::string::npos &&
                          body.find("# TYPE llamacpp:requests_processing gauge") !=
                              std::string::npos,
                      "series carry their Prometheus type");
    failures += check(values.at("llamacpp:prompt_tokens_total") == 9000.0 &&
                          values.at("llamacpp:prompt_seconds_total") == 3.0 &&
                          values.at("llamacpp:tokens_predicted_total") == 80.0 &&
                          values.at("llamacpp:tokens_predicted_seconds_total") == 4.0,
                      "token and seconds counters come from the Engine's live per-unit totals");
    failures += check(values.at("llamacpp:predicted_tokens_seconds") == 20.0 &&
                          values.at("llamacpp:prompt_tokens_seconds") == 3000.0,
                      "average throughputs divide the matching live counters");
    failures += check(values.at("llamacpp:n_decode_total") == 10.0 &&
                          values.at("llamacpp:n_busy_slots_per_decode") == 2.5,
                      "decode rounds and their average batch are reported");
    failures += check(values.at("llamacpp:kv_cache_usage_ratio") == 0.25 &&
                          values.at("llamacpp:kv_cache_tokens") == 32768.0,
                      "KV occupancy is reported in pages and tokens");
    failures += check(values.at("llamacpp:requests_processing") == 3.0 &&
                          values.at("llamacpp:requests_deferred") == 2.0 &&
                          values.at("ninfer:requests_admitted") == 5.0,
                      "request gauges follow the Engine snapshot and ingress");
    failures += check(values.at("ninfer:requests_total") == 2.0 &&
                          values.at("ninfer:requests_failed_total") == 1.0 &&
                          values.at("ninfer:requests_rejected_total") == 2.0,
                      "completed, failed and rejected requests are counted separately");
    failures += check(values.at("ninfer:prefix_cache_hit_tokens_total") == 600.0 &&
                          values.at("ninfer:draft_tokens_total") == 60.0 &&
                          values.at("ninfer:draft_accepted_tokens_total") == 42.0 &&
                          values.at("ninfer:reused_prompt_tokens_total") == 700.0 &&
                          values.at("ninfer:engine_recoveries_total") == 1.0,
                      "ninfer series accumulate reuse, speculation and recoveries");
    const auto at = [&](const char* series) { return values.at(series); };
    failures += check(at("ninfer:context_selections_total{source=\"root\"}") == 3.0 &&
                          at("ninfer:context_selections_total"
                             "{source=\"private_endpoint\"}") == 9.0 &&
                          at("ninfer:context_pressure_events_total"
                             "{event=\"checkpoint_dropped\"}") == 4.0 &&
                          at("ninfer:context_transfer_bytes_total"
                             "{object=\"main_kv\",direction=\"h2d\"}") == 1048576.0 &&
                          at("ninfer:context_occupancy{pool=\"host_state_slots\"}") == 6.0 &&
                          body.find("# TYPE ninfer:context_selections_total counter") !=
                              std::string::npos &&
                          body.find("# TYPE ninfer:context_occupancy gauge") != std::string::npos,
                      "context-cache selections, pressure, transfers and occupancy are exported");
    failures += check(at("ninfer:waiting_expired_requests_total") == 2.0 &&
                          at("ninfer:waiting_cancelled_requests_total") == 0.0 &&
                          at("ninfer:cancelled_prefills_salvaged_total") == 1.0,
                      "abandoned and cancelled requests are counted");
    failures += check(at("ninfer:paced_prefill_steps_total") == 3.0 &&
                          at("ninfer:paced_prefill_deferred_tokens_total") == 4608.0,
                      "paced prefill steps and their deferred tokens are exported");

    failures += check(body.find("ninfer:gpu_") == std::string::npos,
                      "no GPU reading gives no GPU series");

    std::vector<ninfer::GpuReading> gpus(2);
    gpus[0].device                     = 0;
    gpus[0].utilization_percent        = 97;
    gpus[0].memory_utilization_percent = 61;
    gpus[0].memory_used_bytes          = 23ull << 30;
    gpus[0].memory_total_bytes         = 24ull << 30;
    gpus[0].temperature_celsius        = 71;
    gpus[0].power_watts                = 341.5;
    gpus[0].power_limit_watts          = 350.0;
    gpus[0].sm_clock_mhz               = 1755;
    gpus[0].memory_clock_mhz           = 9751;
    gpus[1].device                     = 3;
    gpus[1].utilization_percent        = 4;
    const std::string gpu_body = metrics.render(capacity, sample, gpus);
    const auto gpu_values      = samples(gpu_body);
    const auto gpu             = [&](const char* series) { return gpu_values.at(series); };
    failures += check(gpu("ninfer:gpu_utilization_percent{device=\"0\"}") == 97.0 &&
                          gpu("ninfer:gpu_utilization_percent{device=\"3\"}") == 4.0 &&
                          gpu("ninfer:gpu_memory_used_bytes{device=\"0\"}") == 24696061952.0 &&
                          gpu("ninfer:gpu_power_watts{device=\"0\"}") == 341.5 &&
                          gpu("ninfer:gpu_power_limit_watts{device=\"0\"}") == 350.0 &&
                          gpu("ninfer:gpu_sm_clock_mhz{device=\"0\"}") == 1755.0 &&
                          gpu("ninfer:gpu_memory_clock_mhz{device=\"0\"}") == 9751.0 &&
                          gpu("ninfer:gpu_temperature_celsius{device=\"0\"}") == 71.0,
                      "GPU readings are exported per CUDA device");
    failures += check(gpu_values.count("ninfer:gpu_power_watts{device=\"3\"}") == 0 &&
                          gpu_body.find("# TYPE ninfer:gpu_power_watts gauge") != std::string::npos,
                      "a field NVML did not read gives no sample, not a zero");

    const auto empty = samples(ServeMetrics{}.render(LoadCapacity{}, LoadSample{}));
    failures += check(empty.at("llamacpp:predicted_tokens_seconds") == 0.0 &&
                          empty.at("llamacpp:prompt_tokens_seconds") == 0.0 &&
                          empty.at("llamacpp:kv_cache_usage_ratio") == 0.0 &&
                          empty.at("ninfer:requests_total") == 0.0,
                      "an idle server reports zero rather than a division by zero");
    if (failures != 0) { return 1; }
    std::cout << "serve metrics render llama.cpp-compatible Prometheus series\n";
    return 0;
}

// ninfer-calibrate: measures this GPU's device route profile and stores it where ninfer-serve and
// ninfer look for it (EngineOptions::device_profile). The engine calibrates an unknown device on
// its own at first start; this tool refreshes a profile, writes it elsewhere, or prints it for the
// compiled table.

#include "calibration/device_calibration.h"
#include "calibration/route_catalog.h"
#include "core/device_launch.h"
#include "product/launch_config/launch_config.h"
#include "product/option_layers.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/device_profile.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace {

[[noreturn]] void usage(const char* program, int code) {
    std::fprintf(
        code == 0 ? stdout : stderr,
        "usage: %s [--config FILE] [--device N] [--out PATH] [--print] [--repeat N]\n"
        "          [--margin F] [--no-ternary] [--no-groupwise] [--no-nvfp4] [--no-attention]\n"
        "          [--no-linear-attention] [--only PREFIX] [--detail] [--quiet]\n"
        "  Times every route a device profile can steer on this GPU and stores the profile\n"
        "  at PATH (default: $NINFER_DEVICE_PROFILES, else the user cache), stamped with\n"
        "  this build's route catalog. --only limits it to the route keys starting with\n"
        "  PREFIX (the stored entry keeps the other keys it measured before); --detail logs\n"
        "  every candidate; --print writes the stored entry to stdout. --config FILE reads\n"
        "  options from a launch configuration (docs/cli.md#launch-configuration-files);\n"
        "  every switch also takes its --no- form.\n",
        program);
    std::exit(code);
}

struct Arguments {
    int device = 0;
    std::filesystem::path out;
    bool print = false;
    bool quiet = false;
    ninfer::calibration::CalibrationOptions options;
};

// The arguments as one list. With `spans` it only splits them into options
// (ninfer::product::parse_layered).
Arguments parse_arguments(const std::vector<std::string>& arguments,
                          std::vector<ninfer::product::OptionSpan>* spans) {
    Arguments out;
    ninfer::product::OptionSpanRecorder recorder(spans);
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        recorder.begin(index);
        const std::string_view arg(arguments[index]);
        const auto value = [&]() -> const char* {
            if (index + 1 >= arguments.size()) {
                throw std::invalid_argument(std::string(arg) + " needs a value");
            }
            return arguments[++index].c_str();
        };
        const auto flag = [&](std::string_view name, bool& field) {
            return ninfer::product::switch_option(arg, name, [&](bool on) { field = on; });
        };
        if (arg == "--device") {
            out.device = std::atoi(value());
        } else if (arg == "--out") {
            out.out = value();
        } else if (flag("print", out.print) || flag("ternary", out.options.ternary) ||
                   flag("groupwise", out.options.groupwise) || flag("nvfp4", out.options.nvfp4) ||
                   flag("attention", out.options.attention) ||
                   flag("linear-attention", out.options.linear_attention) ||
                   flag("detail", out.options.detail) || flag("quiet", out.quiet)) {
        } else if (arg == "--repeat") {
            out.options.repeat = std::max(3, std::atoi(value()));
        } else if (arg == "--margin") {
            out.options.margin = std::atof(value());
        } else if (arg == "--only") {
            out.options.only = value();
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    recorder.finish(arguments.size());
    return out;
}

} // namespace

int main(int argc, char** argv) {
    Arguments arguments;
    try {
        const ninfer::product::CommandLine command_line = ninfer::product::read_command_line(
            argc, argv, ninfer::product::PositionalArtifact::None);
        if (command_line.help) { usage(argv[0], 0); }
        arguments = ninfer::product::parse_layered(command_line.layer_arguments(), {},
                                                   parse_arguments);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer-calibrate: %s\n", error.what());
        usage(argv[0], 2);
    }
    const int device                                = arguments.device;
    const std::filesystem::path& out                = arguments.out;
    const bool print                                = arguments.print;
    ninfer::calibration::CalibrationOptions& options = arguments.options;
    try {
        CUDA_CHECK(cudaSetDevice(device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        if (!arguments.quiet) {
            options.log = [](const std::string& line) { std::fprintf(stderr, "%s\n", line.c_str()); };
        }
        ninfer::runtime::DeviceProfileEntry entry =
            ninfer::runtime::calibrated_device_profile_entry(
                ninfer::calibration::calibrate_device_routes(options),
                ninfer::calibration::calibration_route_catalog_digest(
                    ninfer::ops::device_tier(device)));
        entry.profile.hardware_class = ninfer::runtime::context_cost_hardware_class(
            properties.name, properties.major, properties.minor);
        entry.profile.origin = std::string("ninfer-calibrate on ") + properties.name;
        const std::filesystem::path path =
            out.empty() ? ninfer::runtime::default_device_profile_path() : out;
        const ninfer::runtime::DeviceProfileEntry stored =
            ninfer::runtime::upsert_device_route_profile_atomic(path, entry);
        std::fprintf(stderr, "stored %zu measured keys for %s (%d SMs, route catalog %s) in %s\n",
                     stored.profile.routes.size(), stored.profile.hardware_class.c_str(),
                     stored.profile.multiprocessors, stored.calibration->route_catalog.c_str(),
                     path.string().c_str());
        if (print) {
            std::fputs(ninfer::runtime::serialize_device_route_profiles({stored}).c_str(), stdout);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer-calibrate: %s\n", error.what());
        return 1;
    }
    return 0;
}

# Build NInfer on Windows

Native Windows builds run the same engine as Linux: one model per server with concurrent
requests, the context cache, speculative decoding, Vision, and the OpenAI, Anthropic and llama.cpp
APIs, and pipeline stages over several GPUs (`--devices`), which exchange their boundary rows
through pinned host memory under WDDM ([pipeline stages](maintainer/pipeline-stages.md)). The
README inside a release archive is [`release-archive-windows.md`](release-archive-windows.md);
this guide is for a checkout.

**Status.** The last full native build and test run was in October 2026: every target of a `75`
build and of an `86` build with Visual Studio 2022 (MSVC 14.44) and CUDA 12.8, the test suite on a
TITAN RTX (TCC, and WDDM for the virtual-memory tests) and an RTX 3060 (WDDM), run by a contributor.
This repository has no Windows test host. CI compiles the host code of every translation unit with
MSVC 14.44 for `70;75;80;86;89` (CUDA 12.9) and `75;80;86;89;120a` (CUDA 13.4) on each change
to the build inputs (`scripts/check-host-compile.py`), but that host check builds no device code.
The separate `Native release` workflow compiles
and links the release products for two CUDA versions, checks their device images and tests `--help`
without the Toolkit on `PATH`. It has no GPU and does not qualify inference or performance.
The October 22 [dual-CUDA release run](https://github.com/iamwavecut/ninfer-all/actions/runs/39741764110)
completed native linking, device-image checks, startup checks, archive verification and upload
for both variants.
The Windows-only DirectStorage option below has not been run.

MSVC builds delay-load `nvcuda.dll`, so `--help` can run without an NVIDIA driver.
GPU execution still requires a compatible driver and device.

Users run `120a` builds from `scripts\build-native.bat` with Visual Studio 2026 Build Tools and
CUDA 13.4 on an RTX 5070 Ti, and `89` builds with CUDA 13.4 on RTX 4090s
([field reports](performance/field-reports.md)).

## Requirements

- Windows 11 x64 and a recent NVIDIA driver;
- Visual Studio 2022 with MSVC 19.44 (toolset 14.44, Visual Studio 17.14) and CUDA 12.8 or newer
  for `75`, `86` and `89`, or Visual Studio 2026 with CUDA 13.1 or newer for `120a` except 13.2.0 and 13.2.1
  ([CUDA toolkits](build-linux.md#cuda-toolkits));
- CMake, Ninja and [vcpkg](https://github.com/microsoft/vcpkg), which provides FFmpeg and curl
  through `vcpkg.json`;
- the Microsoft Visual C++ runtime to run the executables, and a `.ninfer` artifact.

Configure stops on an MSVC older than 19.44: cl 19.38 (Visual Studio 17.8) computes the overflow
flag of `core::wide_add` wrongly at `/O2` when its inputs are constants, so `ninfer_wide_math_test`
fails, and 19.39 to 19.43 were never qualified. Update Visual Studio 2022 to 17.14, or install its
14.44 build tools beside an older default toolset and open the environment with
`vcvars64.bat -vcvars_ver=14.44`, as `scripts\build.ps1` does.

CUDA 12.8 refuses MSVC v145 (`C1189: unsupported Microsoft Visual Studio version!`), the compiler
of Visual Studio 2026, and `nvcc` takes whatever `cl.exe` comes first on `PATH`. With both Visual
Studio versions installed, build `86` and `89` from the VS 2022 environment (its `vcvars64.bat`
imports the compiler together with `INCLUDE` and `LIB`; adding only the compiler's directory to
`PATH` leaves the standard headers missing).

## Build

`scripts\build.ps1` imports the VS 2022 BuildTools environment itself, forces CUDA 12.8 and the
Ninja generator, and builds into `build-ninja`:

```powershell
.\scripts\build.ps1                       # configure and build
.\scripts\build.ps1 -Test                 # then run the test suite
.\scripts\build.ps1 -Target ninfer-serve  # one target
.\scripts\build.ps1 -Arch 89              # another architecture (75, 80, 86 or 89)
```

The same build by hand:

```powershell
cmake -S . -B build-windows -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TOOLCHAIN_FILE=C:\path\to\vcpkg\scripts\buildsystems\vcpkg.cmake `
  -DVCPKG_TARGET_TRIPLET=x64-windows -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-windows --parallel
```

`scripts\build-native.bat` builds against a prebuilt vcpkg triplet tree instead of the vcpkg
toolchain, with Visual Studio 2026 and CUDA 13 by default, for `120a`. Run
`vcpkg install --triplet x64-windows` in the checkout once: in manifest mode vcpkg installs the
packages of `vcpkg.json` into `vcpkg_installed\x64-windows`, and the script takes that tree when it
holds FFmpeg. The vcpkg that Visual Studio bundles has only this mode. Otherwise the script takes
the classic tree of a standalone vcpkg, `%NINFER_VCPKG_ROOT%\installed\x64-windows` (`C:\vcpkg` by
default). Then run `scripts\build-native.bat configure`, then `build`, or `target ninfer-serve` for
one target. The script passes the tree to CMake as `NINFER_VCPKG_TREE`; `cmake\FindFFMPEG.cmake`
finds FFmpeg there, and the build stages the triplet's DLLs beside every executable.
`NINFER_VCVARS`, `NINFER_GENERATOR`, `NINFER_CUDA_PATH`, `NINFER_VCPKG_TREE`, `NINFER_VCPKG_ROOT`,
`NINFER_VCPKG_TRIPLET`, `NINFER_CUDA_ARCH` and `NINFER_BUILD_DIR` override its defaults. A
configure without the toolchain and without `NINFER_VCPKG_TREE` reads the classic tree that the
`VCPKG_ROOT` and `VCPKG_TARGET_TRIPLET` environment variables name.

A standalone `.cu` probe outside the build tree can be compiled with `nvcc` directly from
PowerShell (Git Bash rewrites MSVC-style flags into paths). Keep probe builds out of the
repository root: `nvcc` writes `.exp` and `.lib` files next to the output.

Windows builds differ from Linux builds in three places: they decode PNG images natively (a
prebuilt FFmpeg tree may have no PNG decoder), they pass the NVFP4 TMA descriptors through device
memory, since MSVC cannot pass them by value, and WDDM charges pinned host memory against the
card's memory. Every Windows build offers `--wddm-evictable-budget`, which sizes the KV cache against
the card's memory less the weights and a 512 MiB desktop floor instead of what is free at startup,
and commits the device arenas at startup. Another process that later takes memory makes WDDM page
some of this process's memory out, which slows generation until it comes back. WDDM may queue a
device's copies behind one another whatever their streams; the one place that put a host-to-device
copy behind a device-side wait for the host, the exchange of Flash-Next native host experts, hung
under WSL2 with queued copies; it uses mapped host memory and a copy kernel on Windows and under
WSL2 ([native host banks](qwen3-8-flash-next.md#the-expert-cache)). This repository has no Windows or
WSL2 host: `ninfer_qwen4_exp_hybrid_experts_test` runs both exchanges on Linux and only the mapped
one under WDDM, where it has not run yet. A process started
by a low-priority parent (a Task Scheduler task at its default priority, a service, some IDE
terminals) inherits I/O priority Low, and Windows throttles its reads; the Engine raises such a
priority to normal before it reads the artifact and logs that it did, or warns when Windows refuses
(`ninfer_io_priority_test`; not run on a Windows host). The Windows-only
build option, untested:

| option | effect |
|---|---|
| `NINFER_DIRECTSTORAGE=ON` | fetches the DirectStorage 1.3 runtime and offers `--disk-kv-directstorage` for disk-tier restores |

The other build options are in the [Linux guide](build-linux.md#build-options).

## Run

`scripts\download-model.bat qwen38-27b` fetches the pinned official Qwen3.8-27B artifact and checks
its size and SHA-256 (`qwen36-27b` and `qwen36-35b-a3b` work the same way), and
`scripts\run.bat qwen38-27b` serves it with the measured profile; the profiles and their overrides
are in [launcher profiles](maintainer/launcher-profiles.md). Any other artifact starts with
`ninfer-serve.exe` and the flags the [launch recipes](recipes.md) give for it, on the command line
or from their file in `configs\`, for example:

```powershell
.\build-ninja\apps\ninfer-serve.exe models\qwen3_8_27b.ninfer --model-id qwen3.8-27b `
  --max-context 131072 --kv-capacity 131072 --kv-dtype rk8v4 --gdn-state-fp16 `
  --spec mtp --draft-tokens 3
.\build-ninja\apps\ninfer-serve.exe --config configs\qwen3.8-27b-dflash2.ini
```

`--kv-capacity auto` sizes the KV pool from the memory left after startup but keeps 1 GiB free;
on a 24 GB card that serves a desktop, an explicit capacity is easier to predict. The
[HTTP serving guide](serving.md) and `ninfer-serve.exe --help` cover every option.

## Release archive

The `Native release` workflow builds two single-CUDA x64 packages with VS 2022 and Ninja, then assembles a universal package:

| Toolkit | Device images available to each executable |
|---|---|
| CUDA 12.9.2 | `70;75;80;86;89` (Volta through Ada) |
| CUDA 13.4.3 | `75;80;86;89;120a` (Turing through consumer Blackwell) |

Each Toolkit compiles its release objects in eight jobs. Every CUDA object retains the complete
architecture set above. Jobs checkpoint separate HF cache families; a final job restores them,
links the executables and verifies the package. At most eight object jobs run together across
both Toolkits. Each runner uses two compiler processes with two nvcc optimization threads each,
to limit the memory pressure observed on the four-CPU Windows runners. This does not establish
a bound on compilation time for the largest individual source files.

Windows release builds set `NINFER_CUDA_PRUNE_PTX=ON`. NVIDIA's `nvprune` removes PTX from
the CUDA static archives before linking, retaining machine code for every listed GPU architecture.
This reduces the image size without recompiling cached objects. These packages do not provide
PTX fallback for GPUs that cannot run any of their machine-code images. Local builds retain PTX
by default; enabling this option requires the Toolkit's `nvprune` component.

The workflow also sets `NINFER_WINDOWS_KERNEL_DLL=ON`: the GGML kernels and the INT8-family
small-T attention units reside in `ninfer_kernels.dll` inside the selected backend directory.
The DLL carries the same architecture set. Its GGML objects retain their PTX; the attention units
come from a pruned archive like the other CUDA archives. The split keeps every PE image within
what Windows loads: the loader refuses an image whose `SizeOfImage` exceeds `0x77000000`
(1904 MiB) with error 193, although the linker accepts up to 2 GiB. In the CUDA 12.9 runtime DLL
of revision `7a89828a3`, the attention units held 1068 of the 1841 MiB of device code. The DLL
uses the existing function-only GGML bridge, exports only the attention units' launch
instantiations, and is staged for apps, the product benchmark and tests. Local builds default to
the static libraries. The packager checks device images in the kernel DLL and the shared runtime
DLL.

Every push to `master` cancels the running build, which saves the objects it completed for the
next run. Feature branches build
only through a manual run, which builds the packages without publication by default. Select
`publish` on `master` to publish. Both final Windows and both final Linux jobs must pass before
publication. The workflow creates an source-identified `YYYY-MM-DD.N` release,
then marks it **latest** after checking every uploaded asset. It leaves an older latest release
available if compilation, packaging or upload fails. See [release process](maintainer/release-process.md).

For a local build, `scripts\package-release.ps1` (or `.\scripts\build.ps1 -Package`) derives the
Toolkit and device images from `CMakeCache.txt`. It requires `cuobjdump` in that Toolkit and `7z`
on `PATH` and `NINFER_SHARED_RUNTIME=ON` in the build. It packages all six command EXEs, their runtime DLLs,
licences and launchers into `.7z.001` volumes of at most 1900 MiB. Download every volume of one
package into the same folder and extract the first with 7-Zip. Keep the backend subdirectories intact.

`BUILD-INFO.json` records the version, source revision, source modifications, CUDA version and device
images. Archive names use the platform, CUDA version, GPU architectures and 12-character source commit,
for example `ninfer-windows-x64-cuda13.4-sm75-80-86-89-120a-0123456789ab.7z.001`.

The previous static-EXE layout produced CUDA 13.4 executables of about 1.95 GB each, plus a 313 MB
GGML DLL. Its package was 6.73 GB unpacked and 6.37 GB archived. The shared-runtime layout removes
this duplication between commands. Its actual linked sizes and GPU behavior still require native
validation. The packager reads each device module's `SizeOfImage` and rejects one above
`0x77000000`. Archive splitting only addresses GitHub's asset limit.

The universal name is `ninfer-windows-x64-cuda12.9-13.4-<commit12>`. Each small EXE chooses one backend
from `backends/cuda12/` and `backends/cuda13/`. Use `--cuda-backend auto|12|13` to override selection.
`--help` and `--version` need no driver. See the [release process](maintainer/release-process.md)
for selection, cache and verification details. A successful startup check does not establish GPU
correctness or throughput.

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
MSVC 14.44 for `70` (CUDA 12.9), `75`, `80`, `86`, `89` and `120a` (CUDA 13.4) on each change
to the build inputs (`scripts/check-host-compile.py`), but builds no device code and runs nothing
on Windows; the Windows-only build option below compiles against Windows headers and has not been
run.

## Requirements

- Windows 11 x64 and a recent NVIDIA driver;
- Visual Studio 2022 (MSVC v143) with CUDA 12.8 or newer for `75`, `86` and `89`, or Visual Studio
  2026 with CUDA 13.1 or newer for `120a`;
- CMake, Ninja and [vcpkg](https://github.com/microsoft/vcpkg), which provides FFmpeg and curl
  through `vcpkg.json`;
- the Microsoft Visual C++ runtime to run the executables, and a `.ninfer` artifact.

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
toolchain, with Visual Studio 2026 and CUDA 13 by default, for `120a`:
`scripts\build-native.bat configure`, then `build`, or `target ninfer-serve` for one target. It
exports `VCPKG_ROOT` and `VCPKG_TARGET_TRIPLET`, from which `cmake\FindFFMPEG.cmake` finds FFmpeg,
and stages the triplet's DLLs beside every executable; `NINFER_VCVARS`, `NINFER_GENERATOR`,
`NINFER_CUDA_PATH`, `NINFER_VCPKG_ROOT`, `NINFER_VCPKG_TRIPLET`, `NINFER_CUDA_ARCH` and
`NINFER_BUILD_DIR` override its defaults.

A standalone `.cu` probe outside the build tree can be compiled with `nvcc` directly from
PowerShell (Git Bash rewrites MSVC-style flags into paths). Keep probe builds out of the
repository root: `nvcc` writes `.exp` and `.lib` files next to the output.

Windows builds differ from Linux builds in three places: they decode PNG images natively (a
prebuilt FFmpeg tree may have no PNG decoder), they pass the NVFP4 TMA descriptors through device
memory, since MSVC cannot pass them by value, and WDDM charges pinned host memory against the
card's memory. Every Windows build offers `--wddm-evictable-budget`, which sizes the KV cache against
the card's memory less the weights and a 512 MiB desktop floor instead of what is free at startup,
and commits the device arenas at startup. Another process that later takes memory makes WDDM page
some of this process's memory out, which slows generation until it comes back. The Windows-only
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
`ninfer-serve.exe` and the flags the [launch recipes](recipes.md) give for it, for example:

```powershell
.\build-ninja\apps\ninfer-serve.exe models\qwen3_8_27b.ninfer --model-id qwen3.8-27b `
  --max-context 131072 --kv-capacity 131072 --kv-dtype rk8v4 --gdn-state-fp16 `
  --spec mtp --draft-tokens 3
```

`--kv-capacity auto` sizes the KV pool from the memory left after startup but keeps 1 GiB free;
on a 24 GB card that serves a desktop, an explicit capacity is easier to predict. The
[HTTP serving guide](serving.md) and `ninfer-serve.exe --help` cover every option.

## Release archive

`scripts\package-release.ps1` (or `.\scripts\build.ps1 -Package`) builds the Windows ZIP for the
version in `VERSION`; [cutting a release](maintainer/release-process.md) describes the flow.

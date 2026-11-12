# NInfer Linux archive

This archive contains prebuilt x64 binaries. Single-CUDA names record the Toolkit, GPU architectures and source commit.
Universal names use `cuda12.9-13.4-<commit12>` and contain both backends. `BUILD-INFO.json` records the version (the release
tag), complete revision, compiler, build system and runtime requirements.

## Extract and verify

Download every numbered volume and the matching `SHA256SUMS-*.txt` for one variant. For example,
replace `<commit12>` below with the commit in the downloaded filenames:

```bash
name='ninfer-linux-x64-cuda12.9-13.4-<commit12>'
sha256sum -c "SHA256SUMS-$name.txt"
cat "$name".tar.gz.[0-9][0-9][0-9] | tar -xz
cd "$name"
sha256sum -c SHA256SUMS.txt
```

CUDA 12.9 packages contain `sm_70`, `sm_75`, `sm_80`, `sm_86` and `sm_89`.
CUDA 13.4 packages contain `sm_75`, `sm_80`, `sm_86`, `sm_89` and `sm_120a`.
The universal package selects CUDA 12 for Volta and CUDA 13 for consumer Blackwell. On overlapping
targets it prefers CUDA 13 if the driver supports it, otherwise CUDA 12. Use `--cuda-backend 12`
or `--cuda-backend 13` to require one backend. Every GPU in `--devices` must fit that backend.
V100 plus Blackwell cannot share a process with these sets. Install a compatible NVIDIA driver
on the host. The minimum Driver API family is 12.0 for CUDA 12 and 13.0 for CUDA 13. This is a
compatibility admission check, not GPU qualification of every driver release.
The CUDA 12 build applies a glibc 2.41 exception-declaration patch to NVIDIA headers;
`BUILD-INFO.json` records it. Function bodies are not modified.

## Debian runtime

CI builds on Debian 13 (glibc 2.41). CUDA runtime libraries are included under `backends/cuda12/` or `backends/cuda13/`, with NVIDIA's EULA. FFmpeg and dav1d are linked into the runtime (see [media libraries](#media-libraries)).
The other libraries stay dynamically linked: glibc, the GCC runtime, zlib and libcurl, whose
SONAMEs (`libc.so.6`, `libstdc++.so.6`, `libz.so.1`, `libcurl.so.4`) Debian and Ubuntu keep across
releases. A newer distribution needs its own packages of these libraries and a glibc no older than
the build host's. Local builds can use another Debian-family host; check `BUILD-INFO.json`.

On Debian 13, install the recorded system runtime packages:

```bash
sudo apt-get update
xargs -a packages/multi.txt sudo apt-get install -y ca-certificates curl
./ninfer --help
./ninfer-serve --help
./ninfer_bench --help
```

This installs libcurl and the other resolved system dependencies. It does not install a
CUDA Toolkit or a driver. The build's temporary driver stub is never shipped. The executables load
one private runtime from the selected backend directory. All commands share that backend's kernel
images. Keep `backends/` and `help/` beside the executables when moving the installation.
`--help` and `--version` work without a driver. NVIDIA libraries remain subject to the
`NVIDIA-CUDA-EULA.txt` in their backend directory, separately from NInfer's Apache-2.0 licence.

## Media libraries

The runtime contains FFmpeg and dav1d, built from their release archives with only the decoders of
the [media formats](https://github.com/iamwavecut/ninfer-all/blob/master/docs/serving.md#media-formats)
NInfer reads. FFmpeg is licensed under the GNU LGPL 2.1 or later and dav1d under the BSD 2-Clause
licence. `LICENSE-ffmpeg.txt` and `LICENSE-dav1d.txt` hold the licence texts, `sources/` holds the
exact archives, and `media_libraries` in `BUILD-INFO.json` records their versions, SHA-256 digests
and build options. To run a modified FFmpeg, rebuild NInfer at the recorded `source_revision` with
the modified archive and its digest in `cmake/MediaSources.cmake`.

## Docker without compilation

From an extracted **single-CUDA** Debian 13 archive (the universal archive has no Dockerfile):

```bash
docker build -t ninfer:local .
docker run --rm --gpus all -p 8080:8080 -v ninfer-models:/models ninfer:local
```

The included Dockerfile installs runtime packages and copies these binaries, the licences and
`sources/`. It does not build NInfer again. The host needs NVIDIA Container Toolkit for GPU access. This image uses the same
entrypoint commands, `run.sh` launcher and model profiles as the source tree. The Container image
workflow packages both Native release archives as `cuda12.9` and `cuda13.4`, with matching
`cuda12.9-<commit12>` and `cuda13.4-<commit12>` tags, without recompilation.

## Start a model

```bash
./run.sh qwen38-27b
```

The launcher downloads the selected model on first use. Set `NINFER_MODEL_DIR` to choose storage,
`NINFER_MODEL` to use an existing artifact, and `NINFER_HOST`/`NINFER_PORT` for the server address.
`configs/` holds the [launch recipes](recipes.md) as
[launch configuration files](cli.md#launch-configuration-files):
`./ninfer-serve --config configs/<file>.ini` from the directory that holds `models/`.
CI checks device images, dynamic dependencies and `--help`; it does not run GPU inference or
measure performance. Verify the intended model and GPU before relying on a package for serving.

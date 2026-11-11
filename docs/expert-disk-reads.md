# Disk expert reads

With `--expert-residency disk` the routed experts stay in the artifact's files, and each call reads
the experts that its device cache lacks. Two options change how those reads reach the GPU:
`--expert-mirror` spreads them over a second drive, and `--expert-direct` lets the drive write them
into device memory. Both apply to every family that has disk experts: native Qwen MoE models,
Qwen3.8-Flash-Next, Gemma 4, Flint and DeepSeek-V4.1. Neither option changes a result: the experts
arrive byte for byte as stored, so tokens, log probabilities and perplexity keep their bits.

Both options require `--expert-residency disk`; startup refuses `--expert-mirror` and
`--expert-direct on` with another residency.

## A second drive: `--expert-mirror DIR`

`DIR` holds a copy of each of the artifact's files, under the same file name, on another drive.
Reads alternate by 1 MiB stripe of the file offset: a read that starts in an even stripe comes from
the artifact's file, a read that starts in an odd stripe from the copy. The misses of one call thus
spread over both drives. An expert's bytes always come from the same copy, so the OS page cache
holds them once.

At startup each copy must be a regular file of the original's size whose first and last 64 KiB
equal the original's. Startup refuses a missing copy or one that differs there. A copy that differs
elsewhere is not detected, so verify a copy after you make it:

```bash
cp model.ninfer /mnt/nvme1/mirror/ && cmp model.ninfer /mnt/nvme1/mirror/model.ninfer
```

The tests read every disk expert reader's experts through a mirror of real copies and compare them
with the file's bytes. An end-to-end check with Qwen3.8-Flash-Next's disk experts gave the same
bytes with and without a mirror (21 of 21 answers). That mirror was a link to the artifact's own
file, so the speed of two drives is not measured.

## Direct reads into device memory: `--expert-direct auto|on|off`

On Linux, with NInfer's two kernel modules loaded, the NVMe drive writes the bytes of the missing
experts straight into a window of device memory (peer-to-peer DMA across PCIe), and a CUDA copy
moves them from the window to the expert bank or cache. Host memory carries none of these bytes, and
no host bounce copy runs.

| Mode | Behavior |
|---|---|
| `auto` (default) | The direct path when the modules are loaded and a self-test read over the window returns the same bytes as a normal read; otherwise reads go through host memory. Startup prints the reason on stderr unless the modules are not loaded. |
| `on` | The direct path; startup fails with the reason when it is not available. |
| `off` | Reads go through host memory. |

The self-test reads up to 1 MiB of each file at an offset inside a block and compares the bytes
with a normal read. When the direct path is in use, the startup log shows one line for each reader
and device.

### State

The direct path has not run on hardware. The tests run every reader through a host window instead:
pinned host memory that ordinary reads fill. That covers the 4 KiB covers, the window offsets, the
CUDA copies out of the window and the self-test. It leaves out the NVIDIA driver calls that export
and register the window, the file extent lookup (unit tests check the extent arithmetic on worked
cases), the kernel modules and the drive's DMA. The kernel modules compile against Ubuntu's Linux
6.8, 7.0 and 7.3 headers (7.x with GCC 14). The NVIDIA driver interfaces were checked against the
headers of drivers 595.99.02 and 620.70.06; other driver versions are not checked.

### Requirements

- Linux with `CONFIG_PCI_P2PDMA=y`. `load.sh` checks the kernel configuration.
- An NVIDIA driver that exports `nvidia_p2p_get_pages_persistent`.
- Resizable BAR enabled. Each window must fit in the GPU's BAR1 aperture, which is 256 MiB without
  Resizable BAR, and the driver keeps part of that.
- The GPU and the NVMe drive on a PCIe path that allows peer-to-peer DMA. The kernel refuses a pair
  of devices that cannot reach each other.
- The artifact's files (and the mirror's) on an NVMe drive, in a filesystem whose extents are
  aligned to 4 KiB, such as ext4 or XFS. Compressed, encrypted, inline or unwritten extents refuse
  the direct path.
- Read access to the raw block device (`/dev/nvme0n1` and so on): membership in group `disk`. That
  group can read and write every drive on the machine, so give it only to a user that you trust with
  the whole machine.
- Secure Boot: the kernel loads only signed modules. Sign the modules with a key that the machine
  trusts (MOK), or turn Secure Boot off.

### Window sizes

Each reader takes a window of device memory on each device, in addition to its usual memory, and
then uses no pinned host staging:

| Reader | Families | Window per device |
|---|---|---|
| Expert delivery | native Qwen MoE, Gemma 4, Flint, DeepSeek-V4.1 | 128 MiB, plus the larger of 8 MiB and the largest expert (rounded up to 4 KiB) |
| Expert stream | Qwen3.8-Flash-Next, native experts | 256 MiB |
| Expert misses | Qwen3.8-Flash-Next, GGUF experts | the larger of 64 MiB and the largest expert projection |

The CPU share of a call (`--expert-dma-share`) still reads its experts through host memory.

### Build and load

The modules are in [`kernel/ninfer-direct`](../kernel/ninfer-direct/README.md) (GPL-2.0-only):

```bash
sudo apt install build-essential linux-headers-$(uname -r)
make -C kernel/ninfer-direct
sudo install -m 0644 kernel/ninfer-direct/99-ninfer-direct.rules /etc/udev/rules.d/
sudo udevadm control --reload
sudo kernel/ninfer-direct/load.sh
sudo usermod -aG disk "$USER"
```

Log in again after the `usermod` command. `sudo kernel/ninfer-direct/load.sh unload` removes the
modules.

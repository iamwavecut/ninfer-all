# ninfer-direct kernel modules

Two Linux kernel modules for NInfer's direct expert reads (`--expert-direct`): an NVMe drive writes
disk experts straight into a window of GPU memory, and host memory carries none of their bytes.
[Disk expert reads](../../docs/expert-disk-reads.md) describes the option, its requirements and
its state. License: GPL-2.0-only ([LICENSE](LICENSE)); the ioctl header `ninfer_direct_uapi.h` is
also available under Apache-2.0, so that NInfer's user-space side can include it.

| Module | Device | What it does |
|---|---|---|
| `ninfer_pin` | `/dev/ninfer_pin` | Pins a window of GPU memory, which the calling process registered for third-party P2P access, into the GPU's BAR1 with `nvidia_p2p_get_pages_persistent()`, and reports the window's bus address. The window stays pinned until the file closes. |
| `ninfer_p2p` | `/dev/ninfer_p2p` | Registers the BAR1 blocks that cover the window as PCI P2PDMA memory, claims exactly the window's pages, and maps them into the process with `mmap()`. An `O_DIRECT` read into that mapping makes the drive write into GPU memory. |

The work is split in two modules because the kernel refuses a module that uses GPL-only kernel
symbols together with the symbols of a proprietary module, and the NVIDIA driver can be the
proprietary one. `ninfer_pin` calls the NVIDIA driver and no GPL-only symbol; `ninfer_p2p` uses the
GPL-only P2PDMA symbols and no NVIDIA symbol.

## Files

| File | Content |
|---|---|
| `ninfer_pin.c`, `ninfer_p2p.c` | the two modules |
| `ninfer_direct_uapi.h` | the ioctl interface; `src/core/expert_direct.cpp` includes it |
| `nv_p2p_abi.h` | the part of NVIDIA's `nv-p2p.h` that `ninfer_pin` calls |
| `Makefile` | builds both modules for the running kernel, or for `KDIR` |
| `load.sh` | checks the kernel configuration, reports each GPU's BAR1 size, loads the modules; `load.sh unload` removes them |
| `99-ninfer-direct.rules` | a udev rule that gives both devices to group `disk` |

## Build and load

```bash
sudo apt install build-essential linux-headers-$(uname -r)
make -C kernel/ninfer-direct
sudo install -m 0644 kernel/ninfer-direct/99-ninfer-direct.rules /etc/udev/rules.d/
sudo udevadm control --reload
sudo kernel/ninfer-direct/load.sh
```

With the NVIDIA driver's `Module.symvers` (found in its DKMS tree, or set `NV_SYMVERS`), the build
checks the calls into the driver; without it, they resolve when `ninfer_pin` loads. With Secure Boot
on, sign both modules with a key that the machine trusts before you load them.

## State

The modules compile against Ubuntu's Linux 6.8, 7.0 and 7.3 headers (7.x with GCC 14). They have
not run on hardware: NInfer's tests run every user-space step of the direct path through a host
window instead ([state](../../docs/expert-disk-reads.md#state)).

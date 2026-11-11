/* SPDX-License-Identifier: (GPL-2.0-only WITH Linux-syscall-note) OR Apache-2.0 */
/*
 * The ioctl interface of ninfer_pin and ninfer_p2p, NInfer's direct expert reads: an NVMe drive
 * writes routed experts straight into a window of GPU memory.
 *
 * NInfer's user-space side (src/core/expert_direct.cpp) includes this header; a change
 * here changes both, and NINFER_DIRECT_ABI with them.
 */
#ifndef NINFER_DIRECT_UAPI_H
#define NINFER_DIRECT_UAPI_H

#include <linux/ioctl.h>
#include <linux/types.h>

#define NINFER_DIRECT_ABI 1

/*
 * ninfer_pin, /dev/ninfer_pin: NINFER_PIN_HOLD pins `bytes` of GPU memory registered for
 * third-party P2P access under `key` (an NV50_THIRD_PARTY_P2P registration of the calling
 * process) into the GPU's BAR1, with nvidia_p2p_get_pages_persistent(), and keeps it pinned until
 * the file is closed. One window per open file. `key` and `bytes` are multiples of 64 KiB.
 *
 * Out: `status` is the NVIDIA driver's return value (0 on success); `bus` the PCI bus address of
 * the window's first byte; `pages` the count of 64 KiB pages; `contiguous` 1 when the pages lie
 * at consecutive bus addresses, which ninfer_p2p requires.
 */
struct ninfer_pin_hold {
    __u32 abi; /* in: NINFER_DIRECT_ABI */
    __u32 reserved;
    __u64 key;        /* in */
    __u64 bytes;      /* in */
    __s32 status;     /* out */
    __u32 pages;      /* out */
    __u64 bus;        /* out */
    __u32 contiguous; /* out */
    __u32 reserved2;
};

#define NINFER_PIN_HOLD _IOWR('N', 1, struct ninfer_pin_hold)

/*
 * ninfer_p2p, /dev/ninfer_p2p: NINFER_P2P_MAP hands the BAR1 range [bus, bus + bytes) of the GPU
 * at PCI address domain:bus_number:devfn to the kernel's PCI peer-to-peer DMA layer and claims
 * exactly the range's pages; a following mmap() of the whole range (offset 0, MAP_SHARED) maps
 * them in order. An O_DIRECT read into that mapping is then written by the drive straight into
 * the GPU's memory. One window per open file; the pages return when the mapping and the file
 * close. `bus` and `bytes` are page aligned.
 *
 * Out: `status` 0 or a negative errno; `pages` the CPU pages claimed.
 */
struct ninfer_p2p_map {
    __u32 abi;        /* in: NINFER_DIRECT_ABI */
    __u32 domain;     /* in: the GPU's PCI domain */
    __u32 bus_number; /* in: the GPU's PCI bus */
    __u32 devfn;      /* in: the GPU's PCI_DEVFN(slot, function) */
    __u64 bus;        /* in: the window's first bus address (NINFER_PIN_HOLD's `bus`) */
    __u64 bytes;      /* in */
    __s32 status;     /* out */
    __u32 pages;      /* out */
};

#define NINFER_P2P_MAP _IOWR('N', 2, struct ninfer_p2p_map)

#endif /* NINFER_DIRECT_UAPI_H */

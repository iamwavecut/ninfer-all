/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * The part of the NVIDIA driver's third-party P2P interface ninfer_pin calls: the persistent
 * page-table calls and the page table they fill, as NVIDIA's nv-p2p.h declares them (MIT,
 * open-gpu-kernel-modules; the same in drivers 595 and 620). nvidia.ko exports both calls.
 */
#ifndef NINFER_NV_P2P_ABI_H
#define NINFER_NV_P2P_ABI_H

#include <linux/types.h>

enum nvidia_p2p_page_size_type {
    NVIDIA_P2P_PAGE_SIZE_4KB = 0,
    NVIDIA_P2P_PAGE_SIZE_64KB,
    NVIDIA_P2P_PAGE_SIZE_128KB,
    NVIDIA_P2P_PAGE_SIZE_COUNT
};

struct nvidia_p2p_page {
    uint64_t physical_address;

    union nvidia_p2p_request_registers {
        struct {
            uint32_t wreqmb_h;
            uint32_t rreqmb_h;
            uint32_t rreqmb_0;
            uint32_t reserved[3];
        } fermi;
    } registers;
};

/* A table whose major version (the upper 16 bits) is this one's lays out as below. */
#define NVIDIA_P2P_PAGE_TABLE_VERSION 0x00020000
#define NINFER_NV_P2P_MAJOR(version)  ((version) >> 16)

struct nvidia_p2p_page_table {
    uint32_t version;
    uint32_t page_size; /* enum nvidia_p2p_page_size_type */
    struct nvidia_p2p_page** pages;
    uint32_t entries;
    uint8_t* gpu_uuid;
    uint32_t flags;
};

int nvidia_p2p_get_pages_persistent(uint64_t virtual_address, uint64_t length,
                                    struct nvidia_p2p_page_table** page_table, uint32_t flags);
int nvidia_p2p_put_pages_persistent(uint64_t virtual_address,
                                    struct nvidia_p2p_page_table* page_table, uint32_t flags);

#endif /* NINFER_NV_P2P_ABI_H */

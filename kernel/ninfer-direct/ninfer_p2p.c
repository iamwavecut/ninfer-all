// SPDX-License-Identifier: GPL-2.0-only
/*
 * ninfer_p2p: hands a pinned BAR1 window of an NVIDIA GPU to the kernel's PCI peer-to-peer DMA
 * layer and maps it, so an O_DIRECT read writes from the NVMe drive straight into GPU memory.
 *
 * ninfer_pin keeps the window in BAR1 and reports its bus range. NINFER_P2P_MAP registers the
 * BAR1 blocks that cover the range as P2PDMA memory (ZONE_DEVICE pages), at most once per boot
 * each, and claims exactly the range's pages from the GPU's P2PDMA pool; mmap() maps them in
 * order, as the kernel's p2pmem allocate file maps its pages. The NVMe driver then DMAs into
 * them through the block layer's P2PDMA support, which also refuses a pair of devices the PCI
 * topology does not let reach each other.
 *
 * The module uses no NVIDIA symbol (see ninfer_pin.c for why the work is split in two).
 */
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/mm.h>
#include <linux/memremap.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/pci-p2pdma.h>
#include <linux/pci.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/version.h>

#include "ninfer_direct_uapi.h"

/*
 * BAR1 is registered in blocks of this size on a grid from the BAR's start, only the blocks a
 * window touches. Registrations last until the GPU's driver unbinds, and the driver places each
 * window anew; a fixed grid keeps two registrations from ever overlapping in part (the memory
 * hotplug layer refuses that), and a block costs 8 MiB of page structures where the whole BAR
 * (32 GiB and more with Resizable BAR) would cost 512 MiB.
 */
#define NINFER_P2P_BLOCK (512ULL << 20)
/* Pool pages parked while the window's own are claimed: 8 GiB of 4 KiB pages. */
#define NINFER_P2P_PARKED_MAX (2U << 20)
#define NINFER_P2P_MAGIC      0x4e503250U

#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 18, 0)
#    define ninfer_get_dev_pagemap(pfn) get_dev_pagemap(pfn)
#else
#    define ninfer_get_dev_pagemap(pfn) get_dev_pagemap(pfn, NULL)
#endif

struct ninfer_p2p_window {
    u32 magic;
    struct pci_dev* gpu; /* holds a reference */
    u64 bus;
    u64 bytes;
    u32 pages;
    bool mapped;
    void** pages_kva; /* the claimed pages, by bus address */
};

/* Serializes registration, claims and release. */
static DEFINE_MUTEX(ninfer_p2p_lock);

static void ninfer_p2p_free(struct ninfer_p2p_window* window) {
    u32 i;

    if (!window) return;
    if (window->magic != NINFER_P2P_MAGIC) {
        WARN_ONCE(1, "ninfer_p2p: free of a window it did not create\n");
        return;
    }
    window->magic = 0;
    if (window->pages_kva) {
        /* A mapped window's pages went back to the pool as the mapping released them. */
        if (!window->mapped) {
            for (i = 0; i < window->pages; ++i) {
                if (window->pages_kva[i])
                    pci_free_p2pmem(window->gpu, window->pages_kva[i], PAGE_SIZE);
            }
        }
        kvfree(window->pages_kva);
    }
    pci_dev_put(window->gpu);
    kfree(window);
}

/* Registers each grid block of BAR1 that [offset, offset + bytes) touches and no earlier call
 * registered. */
static int ninfer_p2p_register(struct pci_dev* gpu, u64 offset, u64 bytes) {
    const u64 start  = pci_resource_start(gpu, 1);
    const u64 length = pci_resource_len(gpu, 1);
    u64 block;

    for (block = round_down(offset, NINFER_P2P_BLOCK); block < offset + bytes;
         block += NINFER_P2P_BLOCK) {
        const u64 size          = min_t(u64, NINFER_P2P_BLOCK, length - block);
        struct dev_pagemap* map = ninfer_get_dev_pagemap(PHYS_PFN(start + block));
        int status;

        if (map) {
            put_dev_pagemap(map);
            continue;
        }
        map = ninfer_get_dev_pagemap(PHYS_PFN(start + block + size - 1));
        if (map) {
            /* Not on the grid: a registration this module did not make. */
            put_dev_pagemap(map);
            pr_err("ninfer_p2p: BAR1 block 0x%llx is partly registered\n", start + block);
            return -EEXIST;
        }
        status = pci_p2pdma_add_resource(gpu, 1, size, block);
        pr_info("ninfer_p2p: %s BAR1 block 0x%llx (+0x%llx) registered: %d\n", pci_name(gpu),
                start + block, size, status);
        if (status) return status;
    }
    return 0;
}

/* Claims the window's pages from the GPU's pool one at a time, parks the others the pool hands
 * out first, and returns those. */
static int ninfer_p2p_claim(struct ninfer_p2p_window* window) {
    void** parked;
    u32 count = 0, claimed = 0, i;

    parked = kvcalloc(NINFER_P2P_PARKED_MAX, sizeof(*parked), GFP_KERNEL);
    if (!parked) return -ENOMEM;
    while (claimed < window->pages) {
        void* page = pci_alloc_p2pmem(window->gpu, PAGE_SIZE);
        u64 bus;

        if (!page) break;
        bus = pci_p2pmem_virt_to_bus(window->gpu, page);
        if (bus >= window->bus && bus < window->bus + window->bytes &&
            !window->pages_kva[(bus - window->bus) >> PAGE_SHIFT]) {
            window->pages_kva[(bus - window->bus) >> PAGE_SHIFT] = page;
            ++claimed;
        } else if (count < NINFER_P2P_PARKED_MAX) {
            parked[count++] = page;
        } else {
            pci_free_p2pmem(window->gpu, page, PAGE_SIZE);
            break;
        }
    }
    for (i = 0; i < count; ++i) pci_free_p2pmem(window->gpu, parked[i], PAGE_SIZE);
    kvfree(parked);
    if (claimed != window->pages) {
        pr_err("ninfer_p2p: window 0x%llx (+0x%llx): %u of %u pages in the pool\n", window->bus,
               window->bytes, claimed, window->pages);
        return -ENOSPC;
    }
    return 0;
}

static long ninfer_p2p_map(struct file* file, void __user* argument) {
    struct ninfer_p2p_map request;
    struct ninfer_p2p_window* window = NULL;
    struct pci_dev* gpu;
    u64 bar_bus, bar_length;
    long result = 0;

    if (copy_from_user(&request, argument, sizeof(request))) return -EFAULT;
    if (request.abi != NINFER_DIRECT_ABI) return -EPROTO;
    if (!request.bytes || ((request.bus | request.bytes) & (PAGE_SIZE - 1)) ||
        (request.bytes >> PAGE_SHIFT) > U32_MAX)
        return -EINVAL;
    request.status = 0;
    request.pages  = 0;

    gpu = pci_get_domain_bus_and_slot(request.domain, request.bus_number, request.devfn);
    if (!gpu) return -ENODEV;
    if (gpu->vendor != PCI_VENDOR_ID_NVIDIA) {
        pci_dev_put(gpu);
        return -ENODEV;
    }
    bar_bus    = pci_bus_address(gpu, 1);
    bar_length = pci_resource_len(gpu, 1);
    if (!bar_length || request.bus < bar_bus ||
        request.bus + request.bytes > bar_bus + bar_length) {
        pci_dev_put(gpu);
        return -ERANGE;
    }

    mutex_lock(&ninfer_p2p_lock);
    if (file->private_data) {
        result = -EBUSY;
        goto fail;
    }
    result = ninfer_p2p_register(gpu, request.bus - bar_bus, request.bytes);
    if (result) goto fail;
    window = kzalloc(sizeof(*window), GFP_KERNEL);
    if (!window) {
        result = -ENOMEM;
        goto fail;
    }
    window->magic     = NINFER_P2P_MAGIC;
    window->gpu       = gpu;
    window->bus       = request.bus;
    window->bytes     = request.bytes;
    window->pages     = (u32)(request.bytes >> PAGE_SHIFT);
    window->pages_kva = kvcalloc(window->pages, sizeof(void*), GFP_KERNEL);
    if (!window->pages_kva) {
        result = -ENOMEM;
        goto fail_window;
    }
    result = ninfer_p2p_claim(window);
    if (result) goto fail_window;
    file->private_data = window;
    request.pages      = window->pages;
    mutex_unlock(&ninfer_p2p_lock);
    pr_info("ninfer_p2p: window at bus 0x%llx (+0x%llx) claimed\n", request.bus, request.bytes);
    return copy_to_user(argument, &request, sizeof(request)) ? -EFAULT : 0;

fail_window:
    ninfer_p2p_free(window); /* puts the GPU */
    mutex_unlock(&ninfer_p2p_lock);
    return result;
fail:
    mutex_unlock(&ninfer_p2p_lock);
    pci_dev_put(gpu);
    return result;
}

/*
 * Maps the claimed pages in bus order, as drivers/pci/p2pdma.c maps its pool's pages: each page
 * enters the mapping with a reference of its own, so the mapping's release of its last page
 * reference returns the page to the pool (gen_pool_free and the pool's percpu reference, which
 * the page's pci_alloc_p2pmem took). Some kernels keep a free device page at one reference
 * (6.8: drivers/pci/p2pdma.c inserts it as it is), newer ones at none (they set it to one
 * first); a page at none starts at one here.
 */
static int ninfer_p2p_mmap(struct file* file, struct vm_area_struct* vma) {
    struct ninfer_p2p_window* window;
    const unsigned long length = vma->vm_end - vma->vm_start;
    unsigned long offset;
    int result = 0;

    mutex_lock(&ninfer_p2p_lock);
    window = file->private_data;
    if (!window || window->magic != NINFER_P2P_MAGIC || window->mapped) {
        result = -ENXIO;
        goto out;
    }
    if ((vma->vm_flags & VM_MAYSHARE) != VM_MAYSHARE || vma->vm_pgoff || length != window->bytes) {
        result = -EINVAL;
        goto out;
    }
    for (offset = 0; offset < length; offset += PAGE_SIZE) {
        struct page* page       = virt_to_page(window->pages_kva[offset >> PAGE_SHIFT]);
        const bool unreferenced = page_ref_count(page) == 0;

        if (unreferenced) set_page_count(page, 1);
        result = vm_insert_page(vma, vma->vm_start + offset, page);
        if (result) {
            /* This page and the rest go back directly; the pages already inserted return
             * as the failed mapping is torn down. */
            if (unreferenced) set_page_count(page, 0);
            for (; offset < length; offset += PAGE_SIZE)
                pci_free_p2pmem(window->gpu, window->pages_kva[offset >> PAGE_SHIFT], PAGE_SIZE);
            break;
        }
        put_page(page);
    }
    window->mapped = true;
out:
    mutex_unlock(&ninfer_p2p_lock);
    return result;
}

static long ninfer_p2p_ioctl(struct file* file, unsigned int command, unsigned long argument) {
    if (command == NINFER_P2P_MAP) return ninfer_p2p_map(file, (void __user*)argument);
    return -ENOTTY;
}

/* misc_open() leaves the miscdevice in private_data; this module keeps its window there. */
static int ninfer_p2p_open(struct inode* inode, struct file* file) {
    file->private_data = NULL;
    return 0;
}

static int ninfer_p2p_release(struct inode* inode, struct file* file) {
    mutex_lock(&ninfer_p2p_lock);
    ninfer_p2p_free(file->private_data);
    file->private_data = NULL;
    mutex_unlock(&ninfer_p2p_lock);
    return 0;
}

static const struct file_operations ninfer_p2p_fops = {
    .owner          = THIS_MODULE,
    .open           = ninfer_p2p_open,
    .release        = ninfer_p2p_release,
    .unlocked_ioctl = ninfer_p2p_ioctl,
    .compat_ioctl   = ninfer_p2p_ioctl,
    .mmap           = ninfer_p2p_mmap,
};

static struct miscdevice ninfer_p2p_device = {
    .minor = MISC_DYNAMIC_MINOR,
    .name  = "ninfer_p2p",
    .fops  = &ninfer_p2p_fops,
    .mode  = 0600,
};

static int __init ninfer_p2p_init(void) { return misc_register(&ninfer_p2p_device); }

static void __exit ninfer_p2p_exit(void) { misc_deregister(&ninfer_p2p_device); }

module_init(ninfer_p2p_init);
module_exit(ninfer_p2p_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("NInfer direct expert reads: pinned GPU BAR1 windows as P2PDMA memory");

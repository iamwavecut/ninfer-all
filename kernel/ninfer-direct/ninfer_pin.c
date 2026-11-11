// SPDX-License-Identifier: GPL-2.0-only
/*
 * ninfer_pin: keeps a window of GPU memory in the GPU's BAR1 for NInfer's direct expert reads.
 *
 * NInfer allocates the window with CUDA, registers it for third-party P2P access through the
 * driver's NV50_THIRD_PARTY_P2P object under a key of its own, and asks this module to pin it.
 * nvidia_p2p_get_pages_persistent() maps the window into BAR1 and keeps it there until the file
 * is closed; the module reports the window's bus address, which ninfer_p2p then hands to the
 * kernel's PCI peer-to-peer DMA layer.
 *
 * The module calls the NVIDIA driver and uses no GPL-only kernel symbol: the kernel refuses a
 * module that takes GPL-only symbols together with symbols of a proprietary module, and the
 * NVIDIA driver may be the proprietary one. ninfer_p2p takes the GPL-only symbols and no NVIDIA
 * symbol.
 */
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/uaccess.h>

#include "ninfer_direct_uapi.h"
#include "nv_p2p_abi.h"

#define NINFER_PIN_PAGE (64ULL << 10)
/* Marks a window this module created: release() frees nothing else. */
#define NINFER_PIN_MAGIC 0x4e50494eU

struct ninfer_pin_window {
    u32 magic;
    u64 key;
    struct nvidia_p2p_page_table* table;
};

/* Serializes HOLD and release on one file. */
static DEFINE_MUTEX(ninfer_pin_lock);

static long ninfer_pin_hold(struct file* file, void __user* argument) {
    struct ninfer_pin_hold request;
    struct nvidia_p2p_page_table* table = NULL;
    struct ninfer_pin_window* window;
    long result = 0;
    u32 i;

    if (copy_from_user(&request, argument, sizeof(request))) return -EFAULT;
    if (request.abi != NINFER_DIRECT_ABI) return -EPROTO;
    if (!request.bytes || ((request.key | request.bytes) & (NINFER_PIN_PAGE - 1))) return -EINVAL;
    request.status     = 0;
    request.pages      = 0;
    request.bus        = 0;
    request.contiguous = 0;

    mutex_lock(&ninfer_pin_lock);
    if (file->private_data) {
        result = -EBUSY;
        goto out;
    }
    request.status = nvidia_p2p_get_pages_persistent(request.key, request.bytes, &table, 0);
    if (request.status || !table) {
        pr_info("ninfer_pin: pinning key 0x%llx (%llu bytes) failed: %d\n", request.key,
                request.bytes, request.status);
        goto out;
    }
    if (NINFER_NV_P2P_MAJOR(table->version) != NINFER_NV_P2P_MAJOR(NVIDIA_P2P_PAGE_TABLE_VERSION) ||
        table->page_size != NVIDIA_P2P_PAGE_SIZE_64KB || !table->entries ||
        (u64)table->entries * NINFER_PIN_PAGE != request.bytes) {
        pr_info("ninfer_pin: unexpected page table (version 0x%x, page size %u, %u pages)\n",
                table->version, table->page_size, table->entries);
        nvidia_p2p_put_pages_persistent(request.key, table, 0);
        result = -EPROTO;
        goto out;
    }
    request.pages      = table->entries;
    request.bus        = table->pages[0]->physical_address;
    request.contiguous = 1;
    for (i = 1; i < table->entries; ++i) {
        if (table->pages[i]->physical_address != request.bus + i * NINFER_PIN_PAGE) {
            request.contiguous = 0;
            break;
        }
    }
    window = kzalloc(sizeof(*window), GFP_KERNEL);
    if (!window) {
        nvidia_p2p_put_pages_persistent(request.key, table, 0);
        result = -ENOMEM;
        goto out;
    }
    window->magic      = NINFER_PIN_MAGIC;
    window->key        = request.key;
    window->table      = table;
    file->private_data = window;
    pr_info("ninfer_pin: key 0x%llx pinned: %u pages at bus 0x%llx, contiguous %u\n", request.key,
            request.pages, request.bus, request.contiguous);
out:
    mutex_unlock(&ninfer_pin_lock);
    if (result) return result;
    return copy_to_user(argument, &request, sizeof(request)) ? -EFAULT : 0;
}

static long ninfer_pin_ioctl(struct file* file, unsigned int command, unsigned long argument) {
    if (command == NINFER_PIN_HOLD) return ninfer_pin_hold(file, (void __user*)argument);
    return -ENOTTY;
}

/* misc_open() leaves the miscdevice in private_data; this module keeps its window there. */
static int ninfer_pin_open(struct inode* inode, struct file* file) {
    file->private_data = NULL;
    return 0;
}

static int ninfer_pin_release(struct inode* inode, struct file* file) {
    struct ninfer_pin_window* window;

    mutex_lock(&ninfer_pin_lock);
    window             = file->private_data;
    file->private_data = NULL;
    mutex_unlock(&ninfer_pin_lock);
    if (!window) return 0;
    if (window->magic != NINFER_PIN_MAGIC) {
        WARN_ONCE(1, "ninfer_pin: release of a window it did not create\n");
        return 0;
    }
    window->magic = 0;
    nvidia_p2p_put_pages_persistent(window->key, window->table, 0);
    kfree(window);
    return 0;
}

static const struct file_operations ninfer_pin_fops = {
    .owner          = THIS_MODULE,
    .open           = ninfer_pin_open,
    .release        = ninfer_pin_release,
    .unlocked_ioctl = ninfer_pin_ioctl,
    .compat_ioctl   = ninfer_pin_ioctl,
};

static struct miscdevice ninfer_pin_device = {
    .minor = MISC_DYNAMIC_MINOR,
    .name  = "ninfer_pin",
    .fops  = &ninfer_pin_fops,
    .mode  = 0600,
};

static int __init ninfer_pin_init(void) { return misc_register(&ninfer_pin_device); }

static void __exit ninfer_pin_exit(void) { misc_deregister(&ninfer_pin_device); }

module_init(ninfer_pin_init);
module_exit(ninfer_pin_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("NInfer direct expert reads: GPU memory windows pinned in BAR1");

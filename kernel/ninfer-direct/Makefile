# SPDX-License-Identifier: GPL-2.0-only
# Builds ninfer_pin.ko and ninfer_p2p.ko for the running kernel (or KDIR):
#   make -C kernel/ninfer-direct
# ninfer_pin calls the NVIDIA driver. With the driver's Module.symvers (NV_SYMVERS, or found in
# the DKMS tree) modpost checks those calls; without it they resolve when the module loads.

ifneq ($(KERNELRELEASE),)

obj-m := ninfer_pin.o ninfer_p2p.o

else

KDIR ?= /lib/modules/$(shell uname -r)/build
NV_SYMVERS ?= $(firstword $(wildcard \
	/var/lib/dkms/nvidia/*/build/Module.symvers \
	/var/lib/dkms/nvidia*/*/$(shell uname -r)/*/Module.symvers \
	/usr/src/nvidia-*/Module.symvers))

ifneq ($(NV_SYMVERS),)
EXTRA := KBUILD_EXTRA_SYMBOLS=$(NV_SYMVERS)
else
EXTRA := KBUILD_MODPOST_WARN=1
endif

all:
	$(MAKE) -C $(KDIR) M=$(CURDIR) $(EXTRA) modules

clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean

.PHONY: all clean

endif

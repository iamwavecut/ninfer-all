#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-only
# Loads ninfer_pin and ninfer_p2p after checking what direct expert reads need (run as root from
# the directory the modules were built in); `load.sh unload` removes them.
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "unload" ]]; then
  rmmod ninfer_p2p 2>/dev/null || true
  rmmod ninfer_pin 2>/dev/null || true
  echo "ninfer-direct: modules removed"
  exit 0
fi

if [[ $EUID -ne 0 ]]; then
  echo "ninfer-direct: run as root" >&2
  exit 1
fi

config="/boot/config-$(uname -r)"
if [[ -r "$config" ]] && ! grep -q '^CONFIG_PCI_P2PDMA=y' "$config"; then
  echo "ninfer-direct: this kernel has no CONFIG_PCI_P2PDMA" >&2
  exit 1
fi

if ! lsmod | grep -q '^nvidia '; then
  modprobe nvidia
fi

# Without Resizable BAR the BAR1 aperture is 256 MiB, of which the driver keeps a part: a window
# then has to stay small. Report it; NInfer sizes its window to what the pin allows.
if command -v nvidia-smi > /dev/null; then
  nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader | while IFS=, read -r index name bus; do
    bar=$(nvidia-smi -i "$index" -q | awk '/BAR1 Memory Usage/{found=1} found && /Total/{print $3, $4; exit}')
    echo "ninfer-direct: GPU $index ($name,$bus) BAR1 $bar"
  done
fi

for module in ninfer_pin ninfer_p2p; do
  if ! lsmod | grep -q "^$module "; then
    insmod "$here/$module.ko"
  fi
done
echo "ninfer-direct: modules loaded (/dev/ninfer_pin, /dev/ninfer_p2p)"

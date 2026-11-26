#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl python3-venv libgomp1
curl -LsSf https://astral.sh/uv/install.sh | sh
/root/.local/bin/uv python install 3.11
/root/.local/bin/uv venv --python 3.11 /workspace/ninfer-work/py311
/root/.local/bin/uv pip install --python /workspace/ninfer-work/py311/bin/python \
    numpy 'huggingface_hub==1.17.0' hf-xet
/workspace/ninfer-work/py311/bin/python --version
nvidia-smi --query-gpu=name,driver_version,memory.free --format=csv,noheader
/workspace/ninfer-work/py311/bin/python scripts/pods/host_gate.py

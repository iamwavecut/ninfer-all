#!/bin/sh
# Nanbeige's full 256k F16 KV requires 52 GiB RAM, separately from prefix images.
# The server's model router and this standalone launcher use the same GPU/port; run one at a time.
set -eu
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec "${NINFER_BIN:-/workspace/ninfer-deepseek-tuned/build/apps/ninfer-serve}" \
 "${NANBEIGE_MODEL:-/workspace/models/Nanbeige4.3-33B-IQ3_XXS-NInfer-v3/Nanbeige4.3-33B-IQ3_XXS.ninfer}" \
 --model-id nanbeige4.3-33b --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" \
 --max-context 262144 --kv-capacity 262144 --max-concurrency 1 \
 --prefill-chunk 512 --host-cache-mib 512 --no-cuda-graph "$@"

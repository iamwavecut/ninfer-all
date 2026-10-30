#!/bin/sh
# Fast interactive Nanbeige profile. The executor fits resident KV to available VRAM;
# 8k bounds growth of host transfers, but TPS still depends on actual prefix length.
set -eu
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec "${NINFER_BIN:-/workspace/ninfer-deepseek-tuned/build/apps/ninfer-serve}" \
 "${NANBEIGE_MODEL:-/workspace/models/Nanbeige4.3-33B-IQ3_XXS-NInfer-v3/Nanbeige4.3-33B-IQ3_XXS.ninfer}" \
 --model-id nanbeige4.3-33b --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" \
 --max-context 8192 --kv-capacity 8192 --max-concurrency 1 \
 --prefill-chunk 512 --host-cache-mib 2048 --no-cuda-graph \
 --no-thinking --default-max-tokens 2048 --lookup-ngram 4 "$@"

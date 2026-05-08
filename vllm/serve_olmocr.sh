#!/usr/bin/env bash
# olmOCR-2-7B FP8 — port 8003

vllm serve allenai/olmOCR-2-7B-1025-FP8 \
    --port 8003 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --max-model-len 4096

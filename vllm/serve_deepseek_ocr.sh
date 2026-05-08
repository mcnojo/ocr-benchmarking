#!/usr/bin/env bash
# DeepSeek-OCR-2 (3B) — port 8001

vllm serve deepseek-ai/DeepSeek-OCR-2 \
    --port 8001 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --max-model-len 4096

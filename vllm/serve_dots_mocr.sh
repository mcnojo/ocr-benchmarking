#!/usr/bin/env bash
# dots.mocr (3B) — port 8002

vllm serve rednote-hilab/dots.mocr \
    --port 8002 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --chat-template-content-format string \
    --trust-remote-code \
    --max-model-len 4096

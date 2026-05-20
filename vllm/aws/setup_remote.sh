#!/usr/bin/env bash
# Runs on EC2 — installs vLLM and starts serving.
# Called by launch.sh, not run directly.

set -euo pipefail

MODEL_ID="$1"
PORT="$2"
EXTRA_ARGS="${3:-}"

export PATH="$HOME/.local/bin:$PATH"

pip3 install --upgrade pip
pip3 install vllm openai

echo "Starting $MODEL_ID on :$PORT"

nohup "$HOME/.local/bin/vllm" serve "$MODEL_ID" \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --max-model-len 4096 \
    $EXTRA_ARGS \
    > /home/ubuntu/vllm_serve.log 2>&1 &

echo "PID: $!"

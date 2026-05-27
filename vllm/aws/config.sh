#!/usr/bin/env bash
# AWS config — edit before running launch/terminate

AWS_REGION="us-west-2"
INSTANCE_TYPE="g6.xlarge"               # 1x L4 24GB
# KEY_NAME="deepseek-ocr-key" # for personal mac
KEY_NAME="ocr-bench-mac"
KEY_FILE="$HOME/.ssh/${KEY_NAME}.pem"
SECURITY_GROUP_NAME="ocr-bench-vllm"

# Deep Learning Base (NVIDIA, Ubuntu 22.04) — us-west-2
AMI_ID="ami-0fa2d26ba6c91cd05"

INSTANCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/instances"

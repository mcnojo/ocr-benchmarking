#!/usr/bin/env bash
# AWS config — edit before running launch/terminate

AWS_REGION="us-east-1"
INSTANCE_TYPE="g6.xlarge"               # 1x L4 24GB
KEY_NAME="ocr-bench"
KEY_FILE="$HOME/.ssh/${KEY_NAME}.pem"
SECURITY_GROUP_NAME="ocr-bench-vllm"

# Deep Learning Base (NVIDIA, Ubuntu 22.04) — us-east-1
AMI_ID="ami-0a1ab4a3fcf997a9d"

INSTANCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/instances"

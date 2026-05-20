#!/usr/bin/env bash
# Launch EC2, install vLLM, start serving.
# Usage: ./launch.sh deepseek|dots|olmocr

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

MODEL_KEY="${1:-}"

case "$MODEL_KEY" in
    deepseek)
        MODEL_ID="deepseek-ai/DeepSeek-OCR-2"
        PORT=8001
        EXTRA=""
        ;;
    dots)
        MODEL_ID="rednote-hilab/dots.mocr"
        PORT=8002
        EXTRA="--chat-template-content-format string"
        ;;
    olmocr)
        MODEL_ID="allenai/olmOCR-2-7B-1025-FP8"
        PORT=8003
        EXTRA=""
        ;;
    *)
        echo "Usage: $0 <deepseek|dots|olmocr>"
        exit 1
        ;;
esac

echo "=== Launching $MODEL_KEY ==="
echo "Model:    $MODEL_ID"
echo "Port:     $PORT"
echo "Instance: $INSTANCE_TYPE"
echo "Region:   $AWS_REGION"
echo ""

# security group
SG_ID=$(aws ec2 describe-security-groups \
    --region "$AWS_REGION" \
    --group-names "$SECURITY_GROUP_NAME" \
    --query "SecurityGroups[0].GroupId" \
    --output text 2>/dev/null || true)

if [[ -z "$SG_ID" || "$SG_ID" == "None" ]]; then
    echo "Creating security group: $SECURITY_GROUP_NAME"
    SG_ID=$(aws ec2 create-security-group \
        --region "$AWS_REGION" \
        --group-name "$SECURITY_GROUP_NAME" \
        --description "vLLM OCR benchmark servers" \
        --query "GroupId" --output text)

    aws ec2 authorize-security-group-ingress \
        --region "$AWS_REGION" --group-id "$SG_ID" \
        --protocol tcp --port 22 --cidr 0.0.0.0/0

    for p in 8001 8002 8003; do
        aws ec2 authorize-security-group-ingress \
            --region "$AWS_REGION" --group-id "$SG_ID" \
            --protocol tcp --port "$p" --cidr 0.0.0.0/0
    done
    echo "Created: $SG_ID"
else
    echo "Using existing: $SG_ID"
fi

# launch
echo ""
echo "Launching instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --region "$AWS_REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=ocr-bench-${MODEL_KEY}},{Key=Project,Value=ocr-benchmarking}]" \
    --query "Instances[0].InstanceId" --output text)

echo "Instance: $INSTANCE_ID"
mkdir -p "$INSTANCE_DIR"
echo "$INSTANCE_ID" > "$INSTANCE_DIR/${MODEL_KEY}.id"
echo "$AWS_REGION" > "$INSTANCE_DIR/${MODEL_KEY}.region"

# wait for running
echo "Waiting for instance..."
aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances \
    --region "$AWS_REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query "Reservations[0].Instances[0].PublicIpAddress" --output text)

echo "IP: $PUBLIC_IP"
echo "$PUBLIC_IP" > "$INSTANCE_DIR/${MODEL_KEY}.ip"

# wait for ssh
echo "Waiting for SSH..."
for i in $(seq 1 30); do
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$KEY_FILE" \
        ubuntu@"$PUBLIC_IP" "echo ready" &>/dev/null; then
        break
    fi
    sleep 10
done

# deploy
scp -o StrictHostKeyChecking=no -i "$KEY_FILE" \
    "$SCRIPT_DIR/setup_remote.sh" ubuntu@"$PUBLIC_IP":/home/ubuntu/setup_remote.sh

echo "Installing vLLM + downloading weights (5-10 min)..."
ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" ubuntu@"$PUBLIC_IP" \
    "chmod +x /home/ubuntu/setup_remote.sh && /home/ubuntu/setup_remote.sh '$MODEL_ID' '$PORT' '$EXTRA'"

echo ""
echo "=== $MODEL_KEY starting on $PUBLIC_IP:$PORT ==="
echo ""
echo "Test:      python vllm/client.py <image> --model $MODEL_KEY"
echo "Logs:      ssh -i $KEY_FILE ubuntu@$PUBLIC_IP 'tail -f ~/vllm_serve.log'"
echo "Terminate: ./vllm/aws/terminate.sh $MODEL_KEY"

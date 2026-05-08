#!/usr/bin/env bash
# Show tracked instance status.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

if [[ ! -d "$INSTANCE_DIR" ]] || ! ls "$INSTANCE_DIR"/*.id &>/dev/null; then
    echo "No tracked instances."
    exit 0
fi

echo "OCR Instances"
echo "============="
echo ""

for f in "$INSTANCE_DIR"/*.id; do
    [[ -f "$f" ]] || continue
    name=$(basename "$f" .id)
    id=$(cat "$f")
    region="$AWS_REGION"
    [[ -f "$INSTANCE_DIR/${name}.region" ]] && region=$(cat "$INSTANCE_DIR/${name}.region")

    state=$(aws ec2 describe-instances \
        --region "$region" \
        --instance-ids "$id" \
        --query "Reservations[0].Instances[0].State.Name" --output text 2>/dev/null || echo "unknown")

    ip="n/a"
    [[ -f "$INSTANCE_DIR/${name}.ip" ]] && ip=$(cat "$INSTANCE_DIR/${name}.ip")

    printf "  %-10s %-20s %-16s %s\n" "$name" "$id" "$ip" "$state"
done

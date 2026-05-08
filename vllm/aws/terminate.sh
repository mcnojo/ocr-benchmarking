#!/usr/bin/env bash
# Terminate instances.
# Usage: ./terminate.sh deepseek|dots|olmocr|all|i-xxxxx

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

terminate_instance() {
    local key="$1"
    local id_file="$INSTANCE_DIR/${key}.id"

    if [[ ! -f "$id_file" ]]; then
        echo "No tracked instance for '$key'"
        return 1
    fi

    local id=$(cat "$id_file")
    local region="$AWS_REGION"
    [[ -f "$INSTANCE_DIR/${key}.region" ]] && region=$(cat "$INSTANCE_DIR/${key}.region")

    echo "Terminating $key ($id)..."
    aws ec2 terminate-instances \
        --region "$region" \
        --instance-ids "$id" \
        --query "TerminatingInstances[0].CurrentState.Name" --output text

    rm -f "$INSTANCE_DIR/${key}.id" "$INSTANCE_DIR/${key}.ip" "$INSTANCE_DIR/${key}.region"
    echo "Done."
}

TARGET="${1:-}"

if [[ -z "$TARGET" ]]; then
    echo "Usage: $0 <deepseek|dots|olmocr|all|INSTANCE_ID>"
    if [[ -d "$INSTANCE_DIR" ]]; then
        echo ""
        echo "Tracked:"
        for f in "$INSTANCE_DIR"/*.id 2>/dev/null; do
            [[ -f "$f" ]] || continue
            name=$(basename "$f" .id)
            id=$(cat "$f")
            ip="?"
            [[ -f "$INSTANCE_DIR/${name}.ip" ]] && ip=$(cat "$INSTANCE_DIR/${name}.ip")
            echo "  $name: $id ($ip)"
        done
    fi
    exit 1
fi

if [[ "$TARGET" == "all" ]]; then
    for f in "$INSTANCE_DIR"/*.id 2>/dev/null; do
        [[ -f "$f" ]] || continue
        terminate_instance "$(basename "$f" .id)"
    done
elif [[ "$TARGET" == i-* ]]; then
    echo "Terminating $TARGET..."
    aws ec2 terminate-instances \
        --region "$AWS_REGION" \
        --instance-ids "$TARGET" \
        --query "TerminatingInstances[0].CurrentState.Name" --output text
    echo "Done."
else
    terminate_instance "$TARGET"
fi

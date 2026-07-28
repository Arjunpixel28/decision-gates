#!/usr/bin/env bash
# Push this project from the Mac to the GPU server and (optionally) run a command.
#
#   GPU_HOST=user@gpu-box ./scripts/sync_to_gpu.sh            # sync only
#   GPU_HOST=user@gpu-box ./scripts/sync_to_gpu.sh "bash setup_env.sh"
#
# Or set a Host alias in ~/.ssh/config and use GPU_HOST=that-alias.
set -euo pipefail

: "${GPU_HOST:?Set GPU_HOST, e.g. GPU_HOST=arjun@192.168.1.50}"
REMOTE_DIR="${REMOTE_DIR:-~/decision-gates}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

rsync -avz --delete \
  --exclude data/ --exclude runs/ --exclude '__pycache__' \
  "$LOCAL_DIR"/ "$GPU_HOST:$REMOTE_DIR"/

echo "synced to $GPU_HOST:$REMOTE_DIR"

if [ $# -gt 0 ]; then
  ssh -t "$GPU_HOST" "cd $REMOTE_DIR && $*"
fi

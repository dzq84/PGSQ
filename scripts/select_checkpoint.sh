#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_DIR" >&2
  exit 2
fi

python -m dmel_codec.tools.select_checkpoint \
  --run-dir "$1" \
  --monitor val_loss \
  --mode min

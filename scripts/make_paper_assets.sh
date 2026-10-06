#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 RUN_DIR [RUN_DIR ...]" >&2
  exit 2
fi

: "${TEST_GATE:?set TEST_GATE to the schema-v3 TEST_GATE_FROZEN file}"

args=()
for run_dir in "$@"; do
  args+=(--run-dir "$run_dir")
done

python -m dmel_codec.tools.make_paper_assets \
  "${args[@]}" \
  --test-gate "${TEST_GATE}" \
  --output-dir paper

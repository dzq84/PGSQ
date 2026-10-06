#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 OURS_RUN DAC_Q2_RUN DAC_Q3_RUN REPAIR_ROUNDS OUTPUT" >&2
  exit 2
fi

python -m dmel_codec.tools.pilot_gate \
  --ours-run "$1" \
  --dac-q2-run "$2" \
  --dac-q3-run "$3" \
  --repair-rounds "$4" \
  --output "$5"

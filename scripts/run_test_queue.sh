#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "usage: TEST_QUEUE=/absolute/queue.json CUDA_VISIBLE_DEVICES=0,1,2,3 $0" >&2
  exit 2
fi

: "${TEST_QUEUE:?set TEST_QUEUE to FINAL_TEST_QUEUE_FROZEN.json}"
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the inference GPU list}"

python -m dmel_codec.tools.test_queue run \
  --queue "${TEST_QUEUE}" \
  --gpus "${CUDA_VISIBLE_DEVICES}" \
  --repository-root .

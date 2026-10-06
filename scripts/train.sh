#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: scripts/train.sh EXPERIMENT CODEC SEED [Hydra overrides...]" >&2
  exit 2
fi

experiment=$1
codec=$2
seed=$3
shift 3

: "${LIBRITTS_ROOT:?set the read-only LIBRITTS_ROOT}"
: "${DMEL_RUN_ROOT:?set an absolute writable root outside the repository}"
: "${BIGVGAN_CKPT:?set BIGVGAN_CKPT}"
: "${BIGVGAN_CONFIG:?set BIGVGAN_CONFIG}"
: "${CUDA_VISIBLE_DEVICES:?select GPUs outside the config with CUDA_VISIBLE_DEVICES}"

nproc=${NPROC:-1}
if [[ ! "${nproc}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC must be a positive integer, got: ${nproc}" >&2
  exit 2
fi

exec torchrun --standalone --nproc_per_node="${nproc}" \
  -m dmel_codec.train_codec \
  experiment="${experiment}" codec="${codec}" seed="${seed}" "$@"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: scripts/train_refinement.sh VARIANT SEED [Hydra overrides...]" >&2
  exit 2
fi

variant=$1
seed=$2
shift 2

case "${variant}" in
  mel_control|waveform_frozen|waveform_joint_tail|waveform_joint_gan) ;;
  *)
    echo "unknown refinement variant: ${variant}" >&2
    exit 2
    ;;
esac

: "${LIBRITTS_ROOT:?set the read-only LIBRITTS_ROOT}"
: "${DMEL_RUN_ROOT:?set an absolute writable root outside the repository}"
: "${BIGVGAN_CKPT:?set BIGVGAN_CKPT}"
: "${BIGVGAN_CONFIG:?set BIGVGAN_CONFIG}"
: "${OURS_REFINEMENT_INIT:?set the selected Ours-M checkpoint}"
: "${CUDA_VISIBLE_DEVICES:?select exactly one GPU outside the config}"

exec python -m dmel_codec.train_codec \
  experiment="refinement_${variant}" codec=ours_m seed="${seed}" "$@"

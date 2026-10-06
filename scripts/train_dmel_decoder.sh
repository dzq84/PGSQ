#!/usr/bin/env bash
set -euo pipefail

: "${DMEL_RUN_ROOT:?Set DMEL_RUN_ROOT to an absolute writable experiment root}"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES; GPU IDs are not embedded here}"

variant="${1:-l1}"
scope="${2:-pilot_clean100}"
seed="${DMEL_DECODER_SEED:-2026}"
case "${variant}" in
  l1)
    delta=0.0
    delta2=0.0
    ;;
  temporal)
    delta=0.5
    delta2=0.25
    ;;
  temporal_strong)
    delta=1.0
    delta2=0.5
    ;;
  *)
    echo "Unknown variant '${variant}'; expected l1, temporal, or temporal_strong" >&2
    exit 2
    ;;
esac

case "${scope}" in
  pilot_clean100)
    default_steps=10000
    default_val_interval=500
    train_manifest="${DMEL_RUN_ROOT}/manifests/libritts24k/train-clean-100_codec.jsonl.gz"
    val_manifest="${DMEL_RUN_ROOT}/diagnostics/dmel_decoder_manifests/dev-clean-first128.jsonl.gz"
    default_run_dir="${DMEL_RUN_ROOT}/dmel_decoder_pilot/${variant}/seed_${seed}"
    ;;
  full_libritts)
    default_steps=100000
    default_val_interval=2000
    train_manifest="${DMEL_RUN_ROOT}/manifests/libritts24k/train_codec.jsonl.gz"
    val_manifest="${DMEL_RUN_ROOT}/manifests/libritts24k/dev-clean_eval.jsonl.gz"
    default_run_dir="${DMEL_RUN_ROOT}/dmel_decoder_full_libritts/${variant}/seed_${seed}"
    ;;
  *)
    echo "Unknown scope '${scope}'; expected pilot_clean100 or full_libritts" >&2
    exit 2
    ;;
esac

run_dir="${DMEL_DECODER_RUN_DIR:-${default_run_dir}}"
steps="${DMEL_DECODER_STEPS:-${default_steps}}"
val_interval="${DMEL_DECODER_VAL_INTERVAL:-${default_val_interval}}"
arguments=(
  --train-manifest "${train_manifest}"
  --val-manifest "${val_manifest}"
  --range-metadata "${DMEL_RUN_ROOT}/assets/dmel_same_frontend/range_q000_995.json"
  --output-dir "${run_dir}"
  --seed "${seed}"
  --max-steps "${steps}"
  --val-check-interval "${val_interval}"
  --learning-rate "${DMEL_DECODER_LR:-0.001}"
  --delta-loss-weight "${delta}"
  --delta2-loss-weight "${delta2}"
)
if [[ -n "${DMEL_DECODER_RESUME_FROM:-}" ]]; then
  arguments+=(--resume-from "${DMEL_DECODER_RESUME_FROM}")
fi

exec python -m dmel_codec.train_dmel_decoder "${arguments[@]}"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_DIR" >&2
  exit 2
fi

run_dir=$1
records="${run_dir}/per_utterance.jsonl"
if [[ ! -f "${records}" ]]; then
  echo "missing per-utterance records: ${records}" >&2
  exit 2
fi

metrics_string=${DMEL_BOOTSTRAP_METRICS:-visqol_speech,pesq_wb,speaker_cosine,wer_reconstruction,wer_delta,encoder_rtf,decoder_rtf}
IFS=',' read -r -a metrics <<< "${metrics_string}"

python -m dmel_codec.evaluation.statistics \
  --records "${records}" \
  --output "${run_dir}/statistics.json" \
  --metrics "${metrics[@]}" \
  --resamples "${DMEL_BOOTSTRAP_RESAMPLES:-10000}" \
  --seed "${DMEL_BOOTSTRAP_SEED:-2027}"

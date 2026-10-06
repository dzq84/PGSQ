#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: scripts/evaluate.sh SYSTEM SPLIT RUN_ID [evaluator arguments...]" >&2
  exit 2
fi

system=$1
split=$2
run_id=$3
shift 3

: "${DMEL_RUN_ROOT:?set DMEL_RUN_ROOT outside the repository}"
: "${DMEL_CACHE_ROOT:?set DMEL_CACHE_ROOT outside the repository}"
: "${CUDA_VISIBLE_DEVICES:?select exactly one inference GPU outside the config}"
if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "Formal codec evaluation uses exactly one visible GPU" >&2
  exit 2
fi

manifest="${DMEL_RUN_ROOT}/manifests/libritts24k/${split}_eval.jsonl.gz"
model_cache="${DMEL_CACHE_ROOT}/models"
arguments=(
  --system "${system}"
  --manifest "${manifest}"
  --split "${split}"
  --run-id "${run_id}"
  --output-dir "${DMEL_RUN_ROOT}/evaluations"
  --cache-root "${DMEL_RUN_ROOT}/cache"
  --model-cache-root "${model_cache}"
  --metrics "${DMEL_METRICS:-stoi,estoi,pesq_wb,visqol_speech,speaker_cosine,si_sdr,mr_stft,ms_log_mel,f0_correlation,wer}"
  --metric-cpu-threads "${DMEL_METRIC_CPU_THREADS:-8}"
  --warmup-cuts 2
  --remeasure-rtf
)

case "${system}" in
  identity)
    ;;
  mel_oracle)
    : "${BIGVGAN_CKPT:?set BIGVGAN_CKPT}"
    : "${BIGVGAN_CONFIG:?set BIGVGAN_CONFIG}"
    arguments+=(--bigvgan-checkpoint "${BIGVGAN_CKPT}" --bigvgan-config "${BIGVGAN_CONFIG}")
    ;;
  dmel)
    : "${BIGVGAN_CKPT:?set BIGVGAN_CKPT}"
    : "${BIGVGAN_CONFIG:?set BIGVGAN_CONFIG}"
    arguments+=(
      --dmel-range "${DMEL_RANGE:-${DMEL_RUN_ROOT}/assets/dmel_same_frontend/range_q000_995.json}"
      --bigvgan-checkpoint "${BIGVGAN_CKPT}"
      --bigvgan-config "${BIGVGAN_CONFIG}"
    )
    ;;
  dmel_decoder)
    : "${DMEL_DECODER_CKPT:?set DMEL_DECODER_CKPT}"
    : "${BIGVGAN_CKPT:?set BIGVGAN_CKPT}"
    : "${BIGVGAN_CONFIG:?set BIGVGAN_CONFIG}"
    arguments+=(
      --dmel-range "${DMEL_RANGE:-${DMEL_RUN_ROOT}/assets/dmel_same_frontend/range_q000_995.json}"
      --dmel-decoder-checkpoint "${DMEL_DECODER_CKPT}"
      --bigvgan-checkpoint "${BIGVGAN_CKPT}"
      --bigvgan-config "${BIGVGAN_CONFIG}"
    )
    ;;
  dac)
    : "${DAC_CKPT:?set DAC_CKPT}"
    arguments+=(--dac-checkpoint "${DAC_CKPT}" --n-quantizers "${DAC_QUANTIZERS:-2}")
    ;;
  ours)
    : "${OURS_CKPT:?set OURS_CKPT}"
    : "${BIGVGAN_CKPT:?set BIGVGAN_CKPT}"
    : "${BIGVGAN_CONFIG:?set BIGVGAN_CONFIG}"
    cublas_workspace_config=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
    if [[ "${cublas_workspace_config}" != ":4096:8" ]]; then
      echo "Ours evaluation requires CUBLAS_WORKSPACE_CONFIG=:4096:8" >&2
      exit 2
    fi
    export CUBLAS_WORKSPACE_CONFIG="${cublas_workspace_config}"
    arguments+=(
      --ours-checkpoint "${OURS_CKPT}"
      --bigvgan-checkpoint "${BIGVGAN_CKPT}"
      --bigvgan-config "${BIGVGAN_CONFIG}"
    )
    ;;
  *)
    echo "unknown system: ${system}" >&2
    exit 2
    ;;
esac

asr_metadata="${ASR_MODEL_METADATA:-${DMEL_RUN_ROOT}/assets/evaluation_models/whisper-large-v3-turbo/model_metadata.json}"
speaker_metadata="${SPEAKER_MODEL_METADATA:-${DMEL_RUN_ROOT}/assets/evaluation_models/spkrec-ecapa-voxceleb/model_metadata.json}"
visqol_metadata="${VISQOL_METADATA:-${DMEL_RUN_ROOT}/assets/evaluation_models/visqol_v3.3.3/model_metadata.json}"
[[ -f "${asr_metadata}" ]] && arguments+=(--asr-model-metadata "${asr_metadata}")
[[ -f "${speaker_metadata}" ]] && arguments+=(--speaker-model-metadata "${speaker_metadata}")
[[ -f "${visqol_metadata}" ]] && arguments+=(--visqol-metadata "${visqol_metadata}")
[[ -n "${TEST_GATE:-}" ]] && arguments+=(--test-gate "${TEST_GATE}")

run_directory="${DMEL_RUN_ROOT}/evaluations/${run_id}"
mkdir -p "${run_directory}/logs"
invocation_id=$(date -u +%Y%m%dT%H%M%SZ)-$$
resource_receipt=${run_directory}/logs/evaluation_invocation_${invocation_id}.resource.json
evaluation_started=false
evaluation_finished=false
resource_receipt_written=false
record_unclosed_evaluation() {
  exit_code=$?
  if [[ "${evaluation_started}" != true || "${resource_receipt_written}" == true ]]; then
    return
  fi
  set +e
  end_time=$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')
  if [[ "${evaluation_finished}" == true ]]; then
    outcome=completed
    resource_exit_code=0
  elif (( exit_code == 130 || exit_code == 143 )); then
    outcome=aborted
    resource_exit_code=${exit_code}
  else
    outcome=failed
    resource_exit_code=${exit_code:-1}
    if (( resource_exit_code == 0 )); then
      resource_exit_code=1
    fi
  fi
  python -m dmel_codec.tools.resource_accounting \
    --record-job "${resource_receipt}" \
    --label "evaluation-${run_id}-${invocation_id}" \
    --category evaluation \
    --outcome "${outcome}" \
    --start-time "${start_time}" \
    --end-time "${end_time}" \
    --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
    --exit-code "${resource_exit_code}" >/dev/null
}
trap record_unclosed_evaluation EXIT

exec > >(tee -a "${run_directory}/logs/evaluate.log") 2>&1
printf '\n[%s] evaluation start pid=%s\n' "$(date --iso-8601=seconds)" "$$"
printf 'command: python -m dmel_codec.evaluation.run_codec_eval'
printf ' %q' "${arguments[@]}" "$@"
printf '\n'

start_time=$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')
evaluation_started=true
python -m dmel_codec.evaluation.run_codec_eval "${arguments[@]}" "$@"
evaluation_finished=true
end_time=$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')
python -m dmel_codec.tools.resource_accounting \
  --record-job "${resource_receipt}" \
  --label "evaluation-${run_id}-${invocation_id}" \
  --category evaluation \
  --outcome completed \
  --start-time "${start_time}" \
  --end-time "${end_time}" \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
  --exit-code 0 >/dev/null
resource_receipt_written=true
trap - EXIT

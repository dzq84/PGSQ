#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: scripts/train_ours_075.sh [resume_from=/absolute/checkpoint.ckpt]" >&2
  exit 2
fi

: "${DMEL_RUN_ROOT:?set an absolute writable root outside the repository}"
: "${OURS_075_RELEASE:?set the frozen OURS_075_FORMAL_RELEASE marker}"
: "${CUDA_VISIBLE_DEVICES:?select exactly one GPU outside the config}"

# Deterministic cuBLAS is required by formal evaluation, but is not part of the
# frozen training environment and substantially reduces training throughput.
unset CUBLAS_WORKSPACE_CONFIG

if [[ "${NPROC:-1}" != "1" ]]; then
  echo "Formal Ours-0.75 uses exactly one process/GPU" >&2
  exit 2
fi

repository=$(cd "$(dirname "$0")/.." && pwd)
python -m dmel_codec.tools.ours_075_release verify \
  --release "${OURS_075_RELEASE}" \
  --repository-root "${repository}" >/dev/null

run_dir=${DMEL_RUN_ROOT}/pilot_clean100_repair2_100k/ours_075/seed_2026
overrides=()
if [[ $# -eq 0 ]]; then
  if [[ -e "${run_dir}" ]]; then
    echo "Formal Ours-0.75 run already exists; resume it explicitly" >&2
    exit 2
  fi
else
  case "$1" in
    resume_from=/*) ;;
    *)
      echo "Only resume_from=/absolute/checkpoint.ckpt is accepted" >&2
      exit 2
      ;;
  esac
  checkpoint=${1#resume_from=}
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Resume checkpoint does not exist: ${checkpoint}" >&2
    exit 2
  fi
  resolved_checkpoint=$(realpath "${checkpoint}")
  resolved_run=$(realpath "${run_dir}")
  case "${resolved_checkpoint}" in
    "${resolved_run}"/checkpoints/*.ckpt) ;;
    *)
      echo "Resume checkpoint must belong to the formal Ours-0.75 run" >&2
      exit 2
      ;;
  esac
  overrides+=("resume_from=${resolved_checkpoint}")
fi

mkdir -p "${run_dir}/logs"
invocation_id=$(date -u +%Y%m%dT%H%M%SZ)-$$
resource_receipt=${run_dir}/logs/training_invocation_${invocation_id}.resource.json
training_started=false
training_finished=false
resource_receipt_written=false
record_training_invocation() {
  exit_code=$?
  if [[ "${training_started}" != true || "${resource_receipt_written}" == true ]]; then
    return
  fi
  set +e
  end_time=$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')
  if [[ "${training_finished}" == true ]]; then
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
    --label "ours-075-seed2026-${invocation_id}" \
    --category training \
    --outcome "${outcome}" \
    --start-time "${start_time}" \
    --end-time "${end_time}" \
    --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
    --exit-code "${resource_exit_code}" >/dev/null
}
trap record_training_invocation EXIT

start_time=$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')
training_started=true
"${repository}/scripts/train.sh" \
  pilot_clean100_repair2_100k ours_075 2026 "${overrides[@]}"
training_finished=true
end_time=$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')
python -m dmel_codec.tools.resource_accounting \
  --record-job "${resource_receipt}" \
  --label "ours-075-seed2026-${invocation_id}" \
  --category training \
  --outcome completed \
  --start-time "${start_time}" \
  --end-time "${end_time}" \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
  --exit-code 0 >/dev/null
resource_receipt_written=true
trap - EXIT

python -m dmel_codec.tools.select_checkpoint \
  --run-dir "${run_dir}" --monitor val_loss --mode min >/dev/null
python -m dmel_codec.tools.ours_075_release verify-final \
  --run-dir "${run_dir}" \
  --release "${OURS_075_RELEASE}" \
  --repository-root "${repository}" \
  --output "${run_dir}/training_complete.json"

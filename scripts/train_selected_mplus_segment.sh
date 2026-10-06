#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: CUDA_VISIBLE_DEVICES=N $0 SEED TARGET_STEP" >&2
  exit 2
fi

seed=$1
target_step=$2
if ! [[ "${seed}" =~ ^202(7|8)$ ]]; then
  echo "SEED must be one of 2027, 2028; seed 2026 is the selected direction run" >&2
  exit 2
fi
if ! [[ "${target_step}" =~ ^[0-9]+$ ]] ||
   (( target_step < 2000 || target_step > 100000 || target_step % 2000 != 0 )); then
  echo "TARGET_STEP must be a multiple of 2000 in [2000, 100000]" >&2
  exit 2
fi

: "${LIBRITTS_ROOT:?set the read-only LIBRITTS_ROOT}"
: "${DMEL_RUN_ROOT:?set an absolute writable root outside the repository}"
: "${DMEL_CACHE_ROOT:?set an absolute writable cache root}"
: "${BIGVGAN_CKPT:?set BIGVGAN_CKPT}"
: "${BIGVGAN_CONFIG:?set BIGVGAN_CONFIG}"
: "${CUDA_VISIBLE_DEVICES:?select exactly one GPU outside the config}"
: "${MPLUS_DIRECTION_SELECTION:?set the frozen MPLUS_DIRECTION_SELECTED marker}"

# Deterministic cuBLAS is required by formal evaluation, but is not part of the
# frozen training environment and substantially reduces training throughput.
unset CUBLAS_WORKSPACE_CONFIG

direction=$(jq -er '.selected_direction' "${MPLUS_DIRECTION_SELECTION}")
case "${direction}" in
  codec|vocoder|joint) ;;
  source)
    echo "The frozen decision retains source; no replication is allowed" >&2
    exit 3
    ;;
  *)
    echo "Invalid selected direction: ${direction}" >&2
    exit 2
    ;;
esac

marker_commit=$(jq -er '.git_commit' "${MPLUS_DIRECTION_SELECTION}")
run_name="mplus_selected_${marker_commit:0:12}_${direction}"
replication_root=${MPLUS_REPLICATION_ROOT:-${DMEL_RUN_ROOT}/decoder_refinement/${run_name}}
output=${replication_root}/seed_${seed}
preflight=${output}/replication_preflight.json
step_text=$(printf '%06d' "${target_step}")
receipt=${output}/logs/segment_step${step_text}.json
invocation_id=$(date -u +%Y%m%dT%H%M%SZ)-$$
log=${output}/logs/segment_step${step_text}_invocation_${invocation_id}.log
resource_receipt=${output}/logs/segment_step${step_text}_invocation_${invocation_id}.resource.json

if [[ ! -f "${preflight}" ]]; then
  if [[ -e "${output}/resolved_config.yaml" ]]; then
    echo "Refusing to create preflight over an existing formal run: ${output}" >&2
    exit 1
  fi
  python -m dmel_codec.tools.mplus_replication preflight \
    --selection "${MPLUS_DIRECTION_SELECTION}" \
    --seed "${seed}" \
    --run-dir "${output}" \
    --output "${preflight}"
else
  python -m dmel_codec.tools.mplus_replication validate-preflight \
    --selection "${MPLUS_DIRECTION_SELECTION}" \
    --preflight "${preflight}" >/dev/null
fi

experiment=$(jq -er '.experiment' "${preflight}")
source_checkpoint=$(jq -er '.source_checkpoint.path' "${preflight}")
mkdir -p "${output}/logs"
export DMEL_STOP_AFTER_VALIDATION_STEP=${target_step}
export PYTHONUNBUFFERED=1

overrides=(
  "experiment=${experiment}"
  codec=ours_m
  "seed=${seed}"
  "experiment.name=${run_name}"
  "experiment.run_dir=${output}"
)
resume_checkpoint=${output}/checkpoints/last.ckpt
if [[ -f "${resume_checkpoint}" ]]; then
  unset OURS_REFINEMENT_INIT || true
  overrides+=(
    experiment.init_from=null
    init_from=null
    "resume_from=${resume_checkpoint}"
  )
else
  export OURS_REFINEMENT_INIT=${source_checkpoint}
fi

training_started=false
training_finished=false
scientific_receipt_written=false
record_unclosed_invocation() {
  exit_code=$?
  if [[ "${training_started}" != true || "${scientific_receipt_written}" == true ]]; then
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
    --label "mplus-${direction}-seed${seed}-step${step_text}-${invocation_id}" \
    --category training \
    --outcome "${outcome}" \
    --start-time "${start_time}" \
    --end-time "${end_time}" \
    --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
    --exit-code "${resource_exit_code}" >/dev/null
}
trap record_unclosed_invocation EXIT

start_time=$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')
training_started=true
torchrun --standalone --nproc_per_node=1 \
  -m dmel_codec.train_codec "${overrides[@]}" >"${log}" 2>&1
training_finished=true
end_time=$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')

python -m dmel_codec.tools.mplus_replication verify-segment \
  --checkpoint "${output}/checkpoints/last.ckpt" \
  --preflight "${preflight}" \
  --target-step "${target_step}" \
  --start-time "${start_time}" \
  --end-time "${end_time}" \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
  --output "${receipt}"
scientific_receipt_written=true
trap - EXIT

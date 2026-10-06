#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: CUDA_VISIBLE_DEVICES=N $0 SEED" >&2
  exit 2
fi
seed=$1
if ! [[ "${seed}" =~ ^202(7|8)$ ]]; then
  echo "SEED must be one of 2027, 2028; seed 2026 is the selected direction run" >&2
  exit 2
fi
: "${DMEL_RUN_ROOT:?set an absolute writable root outside the repository}"
: "${MPLUS_DIRECTION_SELECTION:?set the frozen MPLUS_DIRECTION_SELECTED marker}"

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
launcher=$(dirname "$0")/train_selected_mplus_segment.sh

if [[ -f "${preflight}" ]]; then
  python -m dmel_codec.tools.mplus_replication validate-preflight \
    --selection "${MPLUS_DIRECTION_SELECTION}" \
    --preflight "${preflight}" >/dev/null
fi

target_step=100000
step_text=$(printf '%06d' "${target_step}")
receipt=${output}/logs/segment_step${step_text}.json
if [[ -f "${receipt}" ]]; then
  jq -e \
    --argjson seed "${seed}" \
    --argjson step "${target_step}" \
    --arg direction "${direction}" \
    '.status == "MPLUS_REPLICATION_SEGMENT_VERIFIED"
     and .seed == $seed and .target_step == $step
     and .direction == $direction' "${receipt}" >/dev/null
  echo "already verified ${direction} seed ${seed} through step ${target_step}"
else
  "${launcher}" "${seed}" "${target_step}"
fi

python -m dmel_codec.tools.select_checkpoint \
  --run-dir "${output}" --monitor val_loss --mode min >/dev/null
python -m dmel_codec.tools.mplus_replication verify-final \
  --run-dir "${output}" \
  --preflight "${preflight}" \
  --output "${output}/replication_complete.json"

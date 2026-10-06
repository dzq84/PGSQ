#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 FINAL_SYSTEM_REGISTRY.yaml" >&2
  exit 2
fi

: "${DMEL_RUN_ROOT:?set DMEL_RUN_ROOT outside the repository}"
: "${MPLUS_DIRECTION_SELECTION:?set the frozen M+ direction selection}"
: "${OURS_075_RELEASE:?set the formal Ours-0.75 release}"
: "${OURS_075_COMPLETE:?set the Ours-0.75 training completion receipt}"

registry=$(realpath "$1")
provenance_args=()
selected_direction=$(jq -er '.selected_direction' "${MPLUS_DIRECTION_SELECTION}")
case "${selected_direction}" in
  source) ;;
  codec|vocoder|joint)
    : "${MPLUS_REPLICATION_COMPLETE_2027:?set the seed-2027 M+ completion receipt}"
    : "${MPLUS_REPLICATION_COMPLETE_2028:?set the seed-2028 M+ completion receipt}"
    provenance_args+=(
      --mplus-replication-complete "${MPLUS_REPLICATION_COMPLETE_2027}"
      --mplus-replication-complete "${MPLUS_REPLICATION_COMPLETE_2028}"
    )
    ;;
  *)
    echo "Invalid selected M+ direction: ${selected_direction}" >&2
    exit 2
    ;;
esac

python -m dmel_codec.tools.freeze_test_gate \
  --system-registry "${registry}" \
  --mplus-direction-selection "${MPLUS_DIRECTION_SELECTION}" \
  --ours-075-release "${OURS_075_RELEASE}" \
  --ours-075-complete "${OURS_075_COMPLETE}" \
  "${provenance_args[@]}" \
  --manifest-summary "${DMEL_RUN_ROOT}/manifests/libritts24k/manifest_summary.json" \
  --output "${DMEL_RUN_ROOT}/TEST_GATE_FROZEN"

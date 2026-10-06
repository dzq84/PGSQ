#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 COMPILED_PDF OUTPUT_ZIP" >&2
  exit 2
fi

: "${TEST_GATE:?set TEST_GATE to the schema-v3 TEST_GATE_FROZEN file}"

pdf=$(realpath "$1")
archive=$(realpath -m "$2")
if [[ "${archive}" != *.zip ]]; then
  echo "OUTPUT_ZIP must end in .zip" >&2
  exit 2
fi
manifest="${archive%.zip}.manifest.json"

python -m dmel_codec.tools.package_paper \
  --paper-dir paper \
  --pdf "${pdf}" \
  --archive "${archive}" \
  --manifest "${manifest}" \
  --test-gate "${TEST_GATE}" \
  --repository-root .

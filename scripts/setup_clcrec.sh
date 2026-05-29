#!/usr/bin/env bash
# Clone CLCRec into baselines/CLCRec (official MM21 implementation).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${ROOT}/baselines/CLCRec/src"
if [[ -d "${TARGET}/.git" ]]; then
  echo "CLCRec already cloned at ${TARGET}"
  exit 0
fi
mkdir -p "${ROOT}/baselines/CLCRec"
git clone --depth 1 https://github.com/iLearn-Lab/MM21-CLCRec.git "${TARGET}"
echo "Cloned to ${TARGET}"
echo "See baselines/CLCRec/README.md for train commands."

#!/usr/bin/env bash
# Usage: bash config/run_param_sweep_parallel.sh

set -euo pipefail

PYTHON=python
BASE_CONFIG="config/base_pol.json"
OUTPUT_ROOT="outputs/bead_only_sweep"
mkdir -p "$OUTPUT_ROOT"

RQ=0.0
V0=0.0

NB=(1 4 9 16 25 36 49 64 81 100)
RM=(0.1 0.25 0.5 1.0 1.5)
SIGMA=(0.1 0.5 1.0 1.5 2.0)
DPSI=(1.0)
DIST=("homo" "pol")
IT=( $(seq 1 1 10) )

N_JOBS=8

run_one() {
  local dist="$1"
  local nb="$2"
  local rm="$3"
  local sig="$4"
  local dpsi="$5"
  local iter="$6"

  local run_name="dist_${dist}_nb_${nb}_rm_${rm}_sigma_${sig}_Dpsi_${dpsi}"
  local out_dir="${OUTPUT_ROOT}/${run_name}/iter_${iter}"

  echo "==== Starting ${run_name}_iter_${iter} ===="

  "$PYTHON" main.py \
    --config "$BASE_CONFIG" \
    --out "$out_dir" \
    --set \
      "beads.N_bd=$nb" \
      "beads.dist=$dist" \
      "beads.m_bd=$rm" \
      "beads.sig_bd=$sig" \
      "params.D_psi=$dpsi" \
      "params.rq=$RQ" \
      "params.v0=$V0"

  echo "==== Finished ${run_name}_iter_${iter} ===="
}

export -f run_one
export PYTHON BASE_CONFIG OUTPUT_ROOT RQ V0

parallel -j "$N_JOBS" \
  run_one \
  ::: "${DIST[@]}" \
  ::: "${NB[@]}" \
  ::: "${RM[@]}" \
  ::: "${SIGMA[@]}" \
  ::: "${DPSI[@]}" \
  ::: "${IT[@]}"

echo "Sweep complete. Results in $OUTPUT_ROOT/"
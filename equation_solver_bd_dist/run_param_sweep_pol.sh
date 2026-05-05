#!/usr/bin/env bash
# run_param_sweep.sh
# Example sweep script for equation_solver_v2 main.py
# Usage: bash config/run_param_sweep.sh

set -euo pipefail

PYTHON=python
# If you use a virtual environment, set e.g. PYTHON=~/.venv/bin/python

BASE_CONFIG="config/base_pol.json"
OUTPUT_ROOT="outputs/pol_speed_rq_sweep"
mkdir -p "$OUTPUT_ROOT"

# Parameter arrays (edit as needed)
NB=(4 9 16 25 36 49 64 81 100)
RQ=(0.0 1.0)
V0=(0.0 3.0 5.0)
IT=( $(seq 1 1 3) )

# Optional baseline override for all runs
#COMMON_OVERRIDES=("time.T=10" "domain.Nx=128" "domain.Ny=128")

# Run each combination
for nb in "${NB[@]}"; do
  for rq in "${RQ[@]}"; do
    for v0 in "${V0[@]}"; do
      RUN_NAME="nb_${nb}_rq_${rq}_v0_${v0}"
      for i in "${IT[@]}"; do
        iter="$i"
        OUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
      echo "==== Starting ${RUN_NAME}_iter_${iter} ===="
      "$PYTHON" main.py \
        --config "$BASE_CONFIG" \
        --out "$OUT_DIR" \
        --set "beads.N_bd=$nb" \
              "params.rq=$rq" \
              "params.v0=$v0"

        # "${COMMON_OVERRIDES[@]}" \  # Uncomment if using common overrides

      echo "==== Finished $RUN_NAME ===="
      echo
      done
    done
  done
 done

echo "Sweep complete. Results in $OUTPUT_ROOT/"

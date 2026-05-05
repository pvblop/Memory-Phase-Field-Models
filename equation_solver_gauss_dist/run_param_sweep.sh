#!/usr/bin/env bash
# run_param_sweep.sh
# Example sweep script for equation_solver_v2 main.py
# Usage: bash config/run_param_sweep.sh

set -euo pipefail

PYTHON=python
# If you use a virtual environment, set e.g. PYTHON=~/.venv/bin/python

BASE_CONFIG="config/base_pol.json"
OUTPUT_ROOT="outputs/pol_sweep"
mkdir -p "$OUTPUT_ROOT"

# Parameter arrays (edit as needed)
RBMS=( $(seq 0.1 0.1 0.5 ) )
SIG=( $(seq 5 5 15) )
IT=( $(seq 1 1 5) )


# Optional baseline override for all runs
#COMMON_OVERRIDES=("time.T=10" "domain.Nx=128" "domain.Ny=128")

# Run each combination
for sig in "${SIG[@]}"; do
  for rbm in "${RBMS[@]}"; do
      RUN_NAME="sig_${sig}_rbm_${rbm}"
      OUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
      for it in "${IT[@]}"; do
        OUT_DIR_name="$OUTPUT_ROOT/${RUN_NAME}_it_${it}"      
      echo "==== Starting $OUT_DIR_name ===="
      "$PYTHON" main.py \
        --config "$BASE_CONFIG" \
        --out "$OUT_DIR" \
        --set "beads.sigma=$sig" "beads.rbm=$rbm"

      echo "==== Finished $RUN_NAME ===="
      echo
    done
  done
 done

echo "Sweep complete. Results in $OUTPUT_ROOT/"

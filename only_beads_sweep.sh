#!/usr/bin/env bash
# run_param_sweep.sh
# Example sweep script for equation_solver_v2 main.py
# Usage: bash config/run_param_sweep.sh

set -euo pipefail

PYTHON=python
# If you use a virtual environment, set e.g. PYTHON=~/.venv/bin/python

BASE_CONFIG="config/base_pol.json"
OUTPUT_ROOT="outputs/differentiation_sweep_2"
mkdir -p "$OUTPUT_ROOT"
cp run_param_sweep.sh "$OUTPUT_ROOT/"  # Save a copy of this script for reference

# Parameter arrays (edit as needed)
RBMS=(0.1 0.2 0.3 0.5 0.7 1.0)
SIG=(5 10 15 20 25)
DPSI=(0.01 0.5 1.0 2.0)
X0=(12.5 25.0)
IT=(1)

# Bead-only regime
RQ=0.0
V0=0.0
ALPHA=0.0
LAMP=0.0



# Run each combination
for sig in "${SIG[@]}"; do
  for rbm in "${RBMS[@]}"; do
    for dpsi in "${DPSI[@]}"; do
      for x0 in "${X0[@]}"; do
        RUN_NAME="sig_${sig}_rbm_${rbm}_x0_${x0}_dpsi_${dpsi}"
        OUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
        for it in "${IT[@]}"; do
          OUT_DIR_name="$OUTPUT_ROOT/${RUN_NAME}"      
        echo "==== Starting $OUT_DIR_name ===="
        "$PYTHON" main.py \
          --config "$BASE_CONFIG" \
          --out "$OUT_DIR" \
          --set \
          "beads.dist=$dist" \
          "beads.sigma=$sig" \
          "beads.rbm=$rbm" \
          "params.dpsi=$dpsi"\
          "params.rq=$RQ" \
          "params.v0=$V0" \
          "params.alpha=$ALPHA" \
          "params.lam_p=$LAMP" \

        echo "==== Finished $RUN_NAME ===="
      done
    done
  done
 done

echo "Sweep complete. Results in $OUTPUT_ROOT/"

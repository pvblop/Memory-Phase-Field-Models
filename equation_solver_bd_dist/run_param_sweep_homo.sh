#!/usr/bin/env bash
# run_param_sweep.sh
# Example sweep script for equation_solver_v2 main.py
# Usage: bash config/run_param_sweep.sh

set -euo pipefail

PYTHON=python
# If you use a virtual environment, set e.g. PYTHON=~/.venv/bin/python

BASE_CONFIG="config/one_bead.json"
OUTPUT_ROOT="outputs/one_bead"
mkdir -p "$OUTPUT_ROOT"

# Parameter arrays (edit as needed)
# NB=(64 81 100)
RQ=(0.0 1.0)
V0=(0.0 3.0 5.0)
DPSI=(0.001 0.1 1.0)
RM=(0.1 0.2 0.3 0.5)
SIG=(0.01 0.5 1.0 2.0)
# IT=( $(seq 1 1 3) )

# Optional baseline override for all runs
#COMMON_OVERRIDES=("time.T=10" "domain.Nx=128" "domain.Ny=128")

# Run each combination
for dpsi in "${DPSI[@]}"; do
  for rm in "${RM[@]}"; do
    for sig in "${SIG[@]}"; do
      for rq in "${RQ[@]}"; do
        for v0 in "${V0[@]}"; do
          RUN_NAME="dpsi_${dpsi}_rm_${rm}_sig_${sig}_rq_${rq}_v0_${v0}"
          # for i in "${IT[@]}"; do
          #   iter="$i"
            OUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
          echo "==== Starting ${RUN_NAME} ===="
          "$PYTHON" main.py \
            --config "$BASE_CONFIG" \
            --out "$OUT_DIR" \
            --set "params.D_psi=$dpsi" \
              "params.rq=$rq" \
              "params.v0=$v0" \
              "beads.m_bd=$rm" \
              "beads.sig_bd=$sig"

        # "${COMMON_OVERRIDES[@]}" \  # Uncomment if using common overrides

          echo "==== Finished $RUN_NAME ===="
          echo
          # done
        done
      done
    done
  done
 done

echo "Sweep complete. Results in $OUTPUT_ROOT/"

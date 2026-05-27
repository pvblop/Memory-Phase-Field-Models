#!/usr/bin/env bash
# Usage: bash config/run_param_sweep.sh

set -euo pipefail

PYTHON=python
BASE_CONFIG="config/base_pol.json"
OUTPUT_ROOT="outputs/bead_only_sweep"
mkdir -p "$OUTPUT_ROOT"

# Bead-only regime
RQ=0.0
V0=0.0

# Sweep parameters
NB=(1 4 9 16 25 36 49 64 81 100)
RM=(0.1 0.25 0.5 1.0 1.5)
SIGMA=(0.1 0.5 1.0 1.5 2.0)
DPSI=(1.0)
DIST=("homo" "pol")

# Initial-condition / bead-position iterations
IT=( $(seq 1 1 10) )

for dist in "${DIST[@]}"; do
  for nb in "${NB[@]}"; do
    for rm in "${RM[@]}"; do
      for sig in "${SIGMA[@]}"; do
        for dpsi in "${DPSI[@]}"; do
          for iter in "${IT[@]}"; do

            RUN_NAME="dist_${dist}_nb_${nb}_rm_${rm}_sigma_${sig}_Dpsi_${dpsi}"
            OUT_DIR="$OUTPUT_ROOT/$RUN_NAME"

            echo "==== Starting $RUN_NAME ===="

            "$PYTHON" main.py \
              --config "$BASE_CONFIG" \
              --out "$OUT_DIR" \
              --set \
                "beads.N_bd=$nb" \
                "beads.dist=$dist" \
                "beads.m_bd=$rm" \
                "beads.sig_bd=$sig" \
                "params.D_psi=$dpsi" \
                "params.rq=$RQ" \
                "params.v0=$V0" \

            echo "==== Finished $RUN_NAME ===="
            echo

          done
        done
      done
    done
  done
done

echo "Sweep complete. Results in $OUTPUT_ROOT/"
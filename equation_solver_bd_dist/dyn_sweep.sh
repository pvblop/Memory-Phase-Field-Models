#!/usr/bin/env bash
# run_dynamics_sweep.sh
# Usage: bash config/run_dynamics_sweep.sh

set -euo pipefail

PYTHON=python
BASE_CONFIG="config/base_pol.json"
OUTPUT_ROOT="outputs/dynamics_sweep"
mkdir -p "$OUTPUT_ROOT"

# Fixed bead parameters from bead-only calibration
RM=1.0
SIGMA=2.0
DPSI=1.0

# Geometries
GEOMETRY=("homo" "pol")

# Dynamics sweep
V0=(0.5 1.0 3.0)
ALPHA=(0.1 0.5 1.0)
RQ=(0.0 0.5 1.0)
LAMP=(5.0)
DP=(0.2)

# Number of beads
NB=(4 9 16 25 36 49 64 81 100)


# Seeds / iterations
IT=( $(seq 1 1 3) )

for geom in "${GEOMETRY[@]}"; do
    for nb in "${NB[@]}"; do
        for v0 in "${V0[@]}"; do
            for alpha in "${ALPHA[@]}"; do
                for rq in "${RQ[@]}"; do
                    for lamp in "${LAMP[@]}"; do
                        for dp in "${DP[@]}"; do
                            for iter in "${IT[@]}"; do
                            RUN_NAME="dist_${geom}_n_${nb}_v0_${v0}_alpha_${alpha}_rq_${rq}_lamp_${lamp}_Dp_${dp}"
                            NAME="dist_${geom}_n_${nb}_v0_${v0}_alpha_${alpha}_rq_${rq}_lamp_${lamp}_Dp_${dp}_iter_${iter}"
                            OUT_DIR="$OUTPUT_ROOT/$RUN_NAME"

                            echo "==== Starting $NAME ===="

                            "$PYTHON" main.py \
                                --config "$BASE_CONFIG" \
                                --out "$OUT_DIR" \
                                --set \
                                "beads.N_bd=$nb" \
                                "beads.m_bd=$RM" \
                                "beads.sig_bd=$SIGMA" \
                                "beads.dist=$geom" \
                                "params.D_psi=$DPSI" \
                                "params.v0=$v0" \
                                "params.alpha=$alpha" \
                                "params.rq=$rq" \
                                "params.lam_p=$lamp" \
                                "params.D_p=$dp" \
                                "seed=$iter"

                            echo "==== Finished $NAME ===="
                            echo

                            done
                        done
                    done
                done
            done
        done
    done
done

echo "Sweep complete. Results in $OUTPUT_ROOT/"
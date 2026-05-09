#!/usr/bin/env bash
# run_dynamics_sweep_parallel.sh
# Usage: bash config/run_dynamics_sweep_parallel.sh

set -euo pipefail

export NUMBA_NUM_THREADS=1

PYTHON=python
BASE_CONFIG="config/base_pol.json"
OUTPUT_ROOT="outputs/dynamics_sweep"
mkdir -p "$OUTPUT_ROOT"

# Number of simulations running simultaneously
N_JOBS=8

# Fixed bead parameters
RM=0.5
SIGMA=2.0
DPSI=1.0

# Geometries
GEOMETRY=("homo" "pol")

# Dynamics sweep
V0=(0.5 1.0 3.0)
ALPHA=(0.1 0.5 1.0)
RQ=(0.0 0.5 1.0)
LAMP=(5.0)
DP=(1.0)

# Number of beads
NB=(4 9 16 25 36 49 64 81 100)

# iterations
IT=( $(seq 1 1 1) )

run_one() {
    local geom="$1"
    local nb="$2"
    local v0="$3"
    local alpha="$4"
    local rq="$5"
    local lamp="$6"
    local dp="$7"
    local iter="$8"

    local RUN_NAME="dist_${geom}_n_${nb}_v0_${v0}_alpha_${alpha}_rq_${rq}_lamp_${lamp}_Dp_${dp}"

    local OUT_DIR="$OUTPUT_ROOT/$RUN_NAME"

    mkdir -p "$OUT_DIR"

    echo "==== Starting ${RUN_NAME} iter ${iter} ===="

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

    echo "==== Finished ${RUN_NAME} iter ${iter} ===="
}

export -f run_one
export PYTHON BASE_CONFIG OUTPUT_ROOT RM SIGMA DPSI

parallel -j "$N_JOBS" --halt soon,fail=1 \
    run_one ::: "${GEOMETRY[@]}" ::: "${NB[@]}" ::: "${V0[@]}" ::: "${ALPHA[@]}" ::: "${RQ[@]}" ::: "${LAMP[@]}" ::: "${DP[@]}" ::: "${IT[@]}"

echo "Sweep complete. Results in $OUTPUT_ROOT/"

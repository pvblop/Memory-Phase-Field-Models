#!/usr/bin/env bash
# run_dynamics_sweep_parallel.sh
# Usage: bash config/run_dynamics_sweep_parallel.sh

set -euo pipefail

export NUMBA_NUM_THREADS=1

PYTHON=python
BASE_CONFIG="config/base_two_beads.json"
OUTPUT_ROOT="outputs/two_beads"
mkdir -p "$OUTPUT_ROOT"

# Number of simulations running simultaneously
N_JOBS=8

# Fixed bead parameters
RM=0.5
SIGMA=2.0
DPSI=1.0

# Geometries
GEOMETRY=("two")

# Dynamics sweep
RM=(0.2 0.3 0.5 1.0)
SIGMA=(0.5 1.0 2.0)
L=(0.5 1.0 2.0 5.0 10.0)
V0=(0.0 0.5 1.0 3.0 5.0)
RQ=(0.0 0.5 1.0)
ALPHA=(0.0 0.5 1.0)
LAMP=(5.0)
DP=(1.0)
DPSI=(1.0)

run_one() {
    local v0="$1"
    local alpha="$2"
    local rq="$3"
    local lamp="$4"
    local dp="$5"
    local dpsi="$6"
    local rm="$7"
    local sig="$8"
    local l="$9"
    local iter="${10}"

    local RUN_NAME="rm_${rm}_sig_${sig}_l_${l}_v0_${v0}_alpha_${alpha}_rq_${rq}_lamp_${lamp}_Dp_${dp}_Dpsi_${dpsi}"
    local OUT_DIR="$OUTPUT_ROOT/$RUN_NAME/iter_${iter}"

    mkdir -p "$OUT_DIR"

    echo "==== Starting ${RUN_NAME} iter ${iter} ===="

    "$PYTHON" main.py \
        --config "$BASE_CONFIG" \
        --out "$OUT_DIR" \
        --set \
        "beads.m_bd=$rm" \
        "beads.sig_bd=$sig" \
        "beads.l=$l" \
        "params.v0=$v0" \
        "params.lam_p=$lamp" \
        "params.D_p=$dp" \
        "params.D_psi=$dpsi" \
        "params.alpha=$alpha" \
        "params.rq=$rq" \
        "seed=$iter"

    echo "==== Finished ${RUN_NAME} iter ${iter} ===="
}

export -f run_one
export PYTHON BASE_CONFIG OUTPUT_ROOT

parallel -j "$N_JOBS" --halt soon,fail=1 \
    run_one ::: "${V0[@]}" ::: "${ALPHA[@]}" ::: "${RQ[@]}" ::: "${LAMP[@]}" ::: "${DP[@]}" ::: "${DPSI[@]}" ::: "${RM[@]}" ::: "${SIGMA[@]}" ::: "${L[@]}" ::: {1..3}
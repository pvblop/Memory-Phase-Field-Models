#!/usr/bin/env bash
set -euo pipefail

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PYTHON=python
BASE_CONFIG="config/base_pol.json"
OUTPUT_ROOT="outputs/differentiation_sweep"
mkdir -p "$OUTPUT_ROOT"

N_JOBS=8

# RBMS=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
RBMS=`seq 0.1 0.1 1.0`
# SIG=(3.0 3.5 4.0 4.5 5.0 5.5 7.5 10.0 12.5 15.0)
SIG=`seq 0.5 0.5 15.0`
DPSI=(1.0 3.0 5.0 7.0 10.0 15.0 20.0 25.0 30.0)
X0=(25.0)
IT=(1)

DIST=("pol")

RQ=0.0
V0=0.0
ALPHA=0.0
LAMP=0.0

run_one() {
    local sig="$1"
    local rbm="$2"
    local dpsi="$3"
    local x0="$4"
    local dist="$5"
    local it="$6"

    local RUN_NAME="dist_${dist}_sig_${sig}_rbm_${rbm}_x0_${x0}_dpsi_${dpsi}"
    local OUT_DIR="$OUTPUT_ROOT/$RUN_NAME"

    mkdir -p "$OUT_DIR"

    echo "==== Starting ${RUN_NAME} iter ${it} ===="

    "$PYTHON" main.py \
        --config "$BASE_CONFIG" \
        --out "$OUT_DIR" \
        --set \
        "beads.dist=$dist" \
        "beads.sigma=$sig" \
        "beads.rbm=$rbm" \
        "beads.x0=$x0" \
        "params.dpsi=$dpsi" \
        "params.rq=$RQ" \
        "params.v0=$V0" \
        "params.alpha=$ALPHA" \
        "params.lam_p=$LAMP" \
        "seed=$it"

    echo "==== Finished ${RUN_NAME} iter ${it} ===="
}

export -f run_one
export PYTHON BASE_CONFIG OUTPUT_ROOT RQ V0 ALPHA LAMP

parallel -j "$N_JOBS" --halt soon,fail=1 \
    run_one ::: "${SIG[@]}" ::: "${RBMS[@]}" ::: "${DPSI[@]}" ::: "${X0[@]}" ::: "${DIST[@]}" ::: "${IT[@]}"

echo "Sweep complete. Results in $OUTPUT_ROOT/"
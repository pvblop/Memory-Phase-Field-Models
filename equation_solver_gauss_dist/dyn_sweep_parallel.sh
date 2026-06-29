#!/usr/bin/env bash
set -euo pipefail

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PYTHON=python
BASE_CONFIG="config/base_pol.json"
OUTPUT_ROOT="outputs/only_diff_v0"
mkdir -p "$OUTPUT_ROOT"

N_JOBS=8

RBMS=(0.26 0.78)
DPSI=(1.0)
SIG=(25.0 10.0)
X0=(12.5 25.0)

DIST=("pol")

RQ=(0.0)
V0=(0.5 1.0 3.0 5.0)
ALPHA=0.5
LAMP=5.0

IT=(1 2 3 4 5)

run_one() {
    local sig="$1"
    local x0="$2"
    local rbm="$3"
    local dpsi="$4"
    local rq="$5"
    local v0="$6"
    local dist="$7"
    local it="$8"
    local RUN_NAME="dist_${dist}_sig_${sig}_rbm_${rbm}_x0_${x0}_dpsi_${dpsi}_rq_${rq}_v0_${v0}"
    local OUT_DIR="$OUTPUT_ROOT"

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
        "params.dpsi=$D_psi" \
        "params.rq=$rq" \
        "params.v0=$v0" \
        "params.alpha=$ALPHA" \
        "params.lam_p=$LAMP"

    echo "==== Finished ${RUN_NAME} iter ${it} ===="
}

export -f run_one
export PYTHON BASE_CONFIG OUTPUT_ROOT RQ V0 ALPHA LAMP

parallel -j "$N_JOBS" \
    run_one \
    :::+ "${SIG[@]}" \
    :::+ "${X0[@]}" \
    :::+ "${RBMS[@]}" \
    ::: "${DPSI[@]}" \
    ::: "${RQ[@]}" \
    ::: "${V0[@]}" \
    ::: "${DIST[@]}"\
    ::: "${IT[@]}"

echo "Sweep complete. Results in $OUTPUT_ROOT/"
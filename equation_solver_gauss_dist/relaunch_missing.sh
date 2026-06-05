#!/usr/bin/env bash
set -euo pipefail

MISSING_FILE="missing.txt"

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

N_JOBS=16


PYTHON=python
BASE_CONFIG="config/base_pol.json"
OUTPUT_ROOT="outputs/differentiation_sweep"

RQ="${RQ:-1.0}"
V0="${V0:-1.0}"
ALPHA="${ALPHA:-1.0}"
LAMP="${LAMP:-1.0}"
IT=(1)

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

parallel -j "$N_JOBS" --colsep ' ' \
    run_one {2} {3} {5} {4} {1} "$IT" \
    :::: "$MISSING_FILE"
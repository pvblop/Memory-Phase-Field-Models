#!/usr/bin/env bash
set -euo pipefail

# Sweep: dynamics ON, memory OFF, initial polarization noise varied.
# This assumes main.py accepts overrides through:
#   python main.py --config CONFIG --out OUTDIR --set key=value ...

export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

PYTHON=${PYTHON:-python}
BASE_CONFIG=${BASE_CONFIG:-config/config.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/dyn_no_q_sweep2}
N_JOBS=${N_JOBS:-12}

mkdir -p "$OUTPUT_ROOT"

# -----------------------------------------------------------------------------
# Fixed differentiation / bead cases.
# Each row is one physical passive differentiation point.
# Format:
#   "dist rbm sigma D_psi x0 y0"
#
# Important: these values are kept paired. GNU parallel will NOT mix rbm, sigma,
# x0, etc. across different rows.
# -----------------------------------------------------------------------------
DIFF_CASES=(
    # dist  rbm   sigma  D_psi  x0    y0
    "pol   1.0  10.0   1.0    12.5  25.0"
    "pol   0.3  25.0   1.0    25.0  25.0"
)

# Config compatibility:
# - The config you attached uses beads.type.
# - Some older sweep scripts used beads.dist.
# Run with BEAD_GEOM_KEY=beads.dist if your current main.py expects that key.
BEAD_GEOM_KEY=${BEAD_GEOM_KEY:-beads.type}

# -----------------------------------------------------------------------------
# Dynamics without memory.
# a_p > 0 keeps p linearly stable, so the initial p-noise is a transient source.
# -----------------------------------------------------------------------------
V0S=(0.1 1.0 5.0 10.0)
LAMPS=(0.01 0.1 1.0 5.0 10.0)
TDECS=(10.0 20.0 50.0 100.0)
PNOISES=(0.01 0.1 1.0)

# Keep these fixed in the first sweep to isolate v0, lam_p, t_dec and p-noise.
DP=1.0
AP=1.0
BP=1.0

# Memory OFF.
RQ=0.0
ALPHA=0.0
CHI=0.0
UQ=0.0

# Repetitions / seeds. Start with one seed; use e.g. IT=(1 2 3) for statistics.
IT=(1)

run_one() {
    local diff_case="$1"
    local pnoise="$2"
    local v0="$3"
    local lamp="$4"
    local tdec="$5"
    local it="$6"

    # Parse the differentiation case row.
    local dist rbm sig dpsi x0 y0
    read -r dist rbm sig dpsi x0 y0 <<< "$diff_case"

    local RUN_NAME="dist_${dist}_sig_${sig}_rbm_${rbm}_x0_${x0}_dpsi_${dpsi}_v0_${v0}_lamp_${lamp}_tdec_${tdec}_pnoise_${pnoise}_seed_${it}"
    local OUT_DIR="$OUTPUT_ROOT"

    mkdir -p "$OUT_DIR"

    echo "==== Starting ${RUN_NAME} ===="

    "$PYTHON" main.py \
        --config "$BASE_CONFIG" \
        --out "$OUT_DIR" \
        --set \
        "${BEAD_GEOM_KEY}=${dist}" \
        "beads.sigma=${sig}" \
        "beads.rbm=${rbm}" \
        "beads.x0=${x0}" \
        "beads.y0=${y0}" \
        "beads.t_dec=${tdec}" \
        "params.D_psi=${dpsi}" \
        "params.v0=${v0}" \
        "params.lam_p=${lamp}" \
        "params.D_p=${DP}" \
        "params.a_p=${AP}" \
        "params.b_p=${BP}" \
        "params.rq=${RQ}" \
        "params.alpha=${ALPHA}" \
        "params.chi=${CHI}" \
        "params.u_q=${UQ}" \
        "ic.noise_mag=${pnoise}"

    echo "==== Finished ${RUN_NAME} ===="
}

export -f run_one
export PYTHON BASE_CONFIG OUTPUT_ROOT BEAD_GEOM_KEY DP AP BP RQ ALPHA CHI UQ

parallel -j "$N_JOBS" --halt soon,fail=1 --joblog "$OUTPUT_ROOT/joblog.tsv" \
    run_one ::: "${DIFF_CASES[@]}" \
            ::: "${PNOISES[@]}" \
            ::: "${V0S[@]}" \
            ::: "${LAMPS[@]}" \
            ::: "${TDECS[@]}" \
            ::: "${IT[@]}"

echo "Sweep complete. Results in $OUTPUT_ROOT/"
echo "Job log: $OUTPUT_ROOT/joblog.tsv"
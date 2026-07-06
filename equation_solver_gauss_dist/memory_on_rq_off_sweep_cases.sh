#!/usr/bin/env bash
set -euo pipefail

# Sweep: memory field ON, but memory-driven differentiation OFF.
#
# Interpretation:
#   - q is deposited: params.alpha > 0
#   - q can feed back on p: params.chi > 0
#   - q does NOT directly promote differentiation: params.rq = 0
#
# Fixed for this step:
#   - v0
#   - lam_p
#   - beads.t_dec
#
# This assumes main.py accepts overrides through:
#   python main.py --config CONFIG --out OUTDIR --set key=value ...
#
# IMPORTANT:
# - This script does NOT pass ic.seed.
# - The code is assumed to assign seeds automatically.

export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

PYTHON=${PYTHON:-python}
BASE_CONFIG=${BASE_CONFIG:-config/config.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/memory_on_rq_off_sweep}
N_JOBS=${N_JOBS:-12}

mkdir -p "$OUTPUT_ROOT"

# -----------------------------------------------------------------------------
# Fixed differentiation / bead cases.
#
# Each row is one physical differentiation case:
#   dist rbm sigma D_psi x0 y0
#
# This avoids the Cartesian product between rbm, sigma and x0.
# -----------------------------------------------------------------------------
DIFF_CASES=(
    # dist  rbm   sigma  D_psi  x0    y0
    "pol   0.98  10.0   1.0    12.5  25.0"
    "pol   0.3  25.0   1.0    25.0  25.0"
)

# Config compatibility:
# - Your config uses beads.type.
# - Some older scripts used beads.dist.
# Run with BEAD_GEOM_KEY=beads.dist if your current main.py expects that key.
BEAD_GEOM_KEY=${BEAD_GEOM_KEY:-beads.type}

# -----------------------------------------------------------------------------
# Fixed dynamics for this step.
# Edit these values based on the representative regime found in the previous sweep.
# -----------------------------------------------------------------------------
V0=5.0
TDEC=50.0

# Initial polarization perturbation.
# Keep this fixed here; choose a value from the previous p-noise sweep.
PNOISE=0.1

# Polarization parameters.
DP=1.0
AP=1.0
BP=1.0

# -----------------------------------------------------------------------------
# Memory ON, memory-driven differentiation OFF.
# -----------------------------------------------------------------------------

# rq = 0 means q does not directly enter the psi differentiation term.
RQ=0.0

# alpha controls deposition/build-up of q.
# Include alpha=0 as a control. Remove it if you only want memory-on runs.
ALPHAS=(0.0 0.01 0.05 0.1 0.5 1.0)

# chi controls feedback from q to p.
# Include chi=0 as a diagnostic control: q exists but does not affect p.
CHIS=(0.0 0.1 0.5 1.0 2.0)
LAMP=(0.1 1.0 5.0)


# Memory saturation / relaxation parameter, depending on your implementation.
# Keep fixed unless you explicitly want to study memory persistence.
UQ=0.1

# Repetitions.
# Since the code assigns seeds automatically, REP is only a repetition label.
REPS=(1)

run_one() {
    local diff_case="$1"
    local alpha="$2"
    local chi="$3"
    local lamp="$4"
    local rep="$5"

    read -r dist rbm sig dpsi x0 y0 <<< "$diff_case"

    local RUN_NAME="dist_${dist}_sig_${sig}_rbm_${rbm}_x0_${x0}_dpsi_${dpsi}_v0_${V0}_lamp_${lamp}_tdec_${TDEC}_pnoise_${PNOISE}_alpha_${alpha}_chi_${chi}_rq_${RQ}_rep_${rep}"
    local OUT_DIR="$OUTPUT_ROOT/$RUN_NAME"

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
        "beads.t_dec=${TDEC}" \
        "params.D_psi=${dpsi}" \
        "params.v0=${V0}" \
        "params.lam_p=${lamp}" \
        "params.D_p=${DP}" \
        "params.a_p=${AP}" \
        "params.b_p=${BP}" \
        "params.rq=${RQ}" \
        "params.alpha=${alpha}" \
        "params.chi=${chi}" \
        "params.u_q=${UQ}" \
        "ic.noise_mag=${PNOISE}"

    echo "==== Finished ${RUN_NAME} ===="
}

export -f run_one
export PYTHON BASE_CONFIG OUTPUT_ROOT BEAD_GEOM_KEY
export V0 LAMP TDEC PNOISE DP AP BP RQ UQ

parallel -j "$N_JOBS" --halt soon,fail=1 --joblog "$OUTPUT_ROOT/joblog.tsv" \
    run_one ::: "${DIFF_CASES[@]}" \
            ::: "${ALPHAS[@]}" \
            ::: "${CHIS[@]}" \
            ::: "${LAMP[@]}" \
            ::: "${REPS[@]}"

echo "Sweep complete. Results in $OUTPUT_ROOT/"
echo "Job log: $OUTPUT_ROOT/joblog.tsv"

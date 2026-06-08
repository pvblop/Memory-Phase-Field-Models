#!/usr/bin/env bash

OUTDIR="outputs/differentiation_sweep"

RBMS=($(seq 0.1 0.1 1.0))
SIG=($(seq 0.5 0.5 15.0))
DPSI=(1.0 3.0 5.0 7.0 10.0 15.0 20.0 25.0 30.0)
X0=(25.0)
DIST=("pol")   # adjust to your sweep

MISSING="missing.txt"
> "$MISSING"

for dist in "${DIST[@]}"; do
for sig in "${SIG[@]}"; do
for rbm in "${RBMS[@]}"; do
for x0 in "${X0[@]}"; do
for dpsi in "${DPSI[@]}"; do

    sim="dist_${dist}_sig_${sig}_rbm_${rbm}_x0_${x0}_dpsi_${dpsi}"
    sim_txt="${dist} ${sig} ${rbm} ${x0} ${dpsi}"

    if ! find "${OUTDIR}/${sim}" -path '*/data.h5' -print -quit | grep -q .; then
    echo "$sim_txt" >> "$MISSING"
    fi

done
done
done
done
done

echo "Missing simulations:"
wc -l "$MISSING"
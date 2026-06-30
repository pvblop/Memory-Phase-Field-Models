"""GPU-batched parameter-sweep runner.

Runs many simulations together on the GPU as an ensemble batch (leading axis),
which is the only way the small 100x100 grid saturates the device.  Produces
exactly the same per-simulation ``data.h5`` / ``config_used.json`` layout as
main.py, so existing analysis tooling keeps working.

Examples
--------
Cartesian sweep of v0 x sigma x seed:

    python gpu_sweep.py --config config/config.json --out outputs/sweep \\
        --sweep params.v0=0.5,1.0,3.0,5.0 beads.sigma=10.0,25.0 ic.seed=1,2,3

Fixed overrides (applied to every sim) use --set, like main.py:

    python gpu_sweep.py --config config/config.json --out outputs/run \\
        --set params.alpha=0.5 --sweep params.v0=1.0,5.0 ic.seed=1,2,3,4,5

Notes
-----
* All sims share one grid, T, and time step (dt = the smallest stable dt over
  the whole sweep) so the batch marches in lockstep.
* --dtype float32 (default) is strongly recommended on consumer/laptop GPUs;
  float64 is far slower there and halves batch capacity.
* Batch size is auto-chosen from free VRAM and a host-RAM budget; override with
  --batch.
"""
import argparse
import json
import os
import itertools
from copy import deepcopy
from datetime import datetime

import numpy as np

from config_io import set_in_dict, parse_value, write_h5, SOLUTION_KEYS
from operators import suggest_dt


# ----------------------------------------------------------------------------
# Parameter extraction (mirrors main.py so CPU and GPU agree on defaults)
# ----------------------------------------------------------------------------
def extract_scalars(cfg):
    """Resolve the per-sim scalar parameters from a config dict."""
    p = cfg["params"]
    s = {
        "lam_psi": p["lam_psi"], "D_psi": p["D_psi"], "v0": p["v0"],
        "lam_p": p["lam_p"], "D_p": p["D_p"],
        "a_p": p.get("a_p", p.get("gamma", 1.0)),
        "b_p": p.get("b_p", 1.0),
        "chi": p.get("chi", p.get("gamma", 0.0)),
        "alpha": p["alpha"], "u_q": p.get("u_q", 1.0),
        "rq": p["rq"], "mu": p["mu"],
        "t_dec": cfg["beads"]["t_dec"],
        "rbm": cfg["beads"]["rbm"],
        "bd_type": cfg["beads"]["type"],
        "x0": cfg["beads"].get("x0", 0.0),
        "y0": cfg["beads"].get("y0", 0.0),
        "sigma": cfg["beads"].get("sigma", 1.0),
        "noise_mag": cfg["ic"]["noise_mag"],
        "seed": cfg["ic"]["seed"],
    }
    return s


# Parameter names passed to the GPU RHS as (B,1,1) arrays.
GPU_PARAM_KEYS = ("lam_psi", "D_psi", "v0", "lam_p", "D_p", "a_p", "b_p",
                  "chi", "alpha", "u_q", "rq", "mu", "t_dec")


def expand_sweep(base_cfg, sweep_specs):
    """Return a list of (resolved_cfg, swept_leaf_dict) for the cartesian
    product of the sweep specs.  Each spec is 'key.path=v1,v2,...'."""
    keys, value_lists = [], []
    for spec in sweep_specs:
        k, vals = spec.split("=", 1)
        keys.append(k)
        value_lists.append([parse_value(v) for v in vals.split(",")])

    cfgs = []
    combos = list(itertools.product(*value_lists)) if keys else [()]
    for combo in combos:
        cfg = deepcopy(base_cfg)
        leaf = {}
        for k, v in zip(keys, combo):
            set_in_dict(cfg, k, v)
            leaf[k.split(".")[-1]] = v
        # every sim needs a concrete seed for its noise field
        if "seed" not in cfg["ic"]:
            cfg["ic"]["seed"] = int(np.random.randint(0, 2**31 - 1))
        cfgs.append((cfg, leaf))
    return cfgs


def build_ic(s, X, Y, Nx, Ny):
    """Initial condition for one sim (numpy), matching main.py exactly."""
    psi0 = -np.ones((Ny, Nx))
    rng = np.random.default_rng(s["seed"])
    pmag = s["noise_mag"]
    px0 = pmag * (2 * rng.random((Ny, Nx)) - 1)
    py0 = pmag * (2 * rng.random((Ny, Nx)) - 1)
    px0[0, :] = px0[-1, :] = 0; px0[:, 0] = px0[:, -1] = 0
    py0[0, :] = py0[-1, :] = 0; py0[:, 0] = py0[:, -1] = 0
    qx0 = np.zeros((Ny, Nx)); qy0 = np.zeros((Ny, Nx))
    return psi0, px0, py0, qx0, qy0


def auto_batch(n_sims, Ny, Nx, nsaved, itemsize, free_vram, host_budget,
               vram_frac, n_save_fields, sweet_spot=128):
    """Pick a batch size that fits both VRAM and a host-RAM budget.

    On a small (laptop) GPU this grid saturates the device at B~128 and total
    sweep time is ~proportional to per-sim step time, so going bigger than the
    compute sweet spot does not help throughput (and only costs host RAM).  On
    a larger GPU raise --batch to push the sweet spot up.
    """
    # ~80 transient (B,Ny,Nx) arrays peak inside an RK4 step (functional style)
    per_sim_vram = 80 * Ny * Nx * itemsize
    vram_B = max(1, int(free_vram * vram_frac / per_sim_vram))
    per_sim_host = nsaved * Ny * Nx * 4 * n_save_fields   # float32 history
    host_B = max(1, int(host_budget / per_sim_host))
    return max(1, min(n_sims, vram_B, host_B, sweet_spot))


def run_name(leaf):
    if not leaf:
        return "run"
    return "_".join(f"{k}_{v}" for k, v in leaf.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="outputs/gpu_sweep")
    ap.add_argument("--set", nargs="*", default=[],
                    help="Fixed overrides applied to every sim, e.g. params.alpha=0.5")
    ap.add_argument("--sweep", nargs="*", default=[],
                    help="Swept params, e.g. params.v0=0.5,1.0 ic.seed=1,2,3 (cartesian product)")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--batch", type=int, default=0, help="Force batch size (0 = auto)")
    ap.add_argument("--host-budget-gb", type=float, default=8.0)
    ap.add_argument("--vram-frac", type=float, default=0.5)
    args = ap.parse_args()

    import cupy as cp
    from gpu.solver_gpu import simulate_rk4_batched_gpu
    from gpu.operators_gpu import beads_dist as beads_gpu

    dtype = np.float32 if args.dtype == "float32" else np.float64

    with open(args.config) as f:
        base_cfg = json.load(f)
    for item in args.set:
        k, v = item.split("=", 1)
        set_in_dict(base_cfg, k, parse_value(v))

    sim_cfgs = expand_sweep(base_cfg, args.sweep)
    n_sims = len(sim_cfgs)

    # --- grid (must be shared across the whole sweep) ---
    d = base_cfg["domain"]
    Nx, Ny, Lx, Ly = d["Nx"], d["Ny"], d["Lx"], d["Ly"]
    for cfg, _ in sim_cfgs:
        cd = cfg["domain"]
        if (cd["Nx"], cd["Ny"], cd["Lx"], cd["Ly"]) != (Nx, Ny, Lx, Ly):
            raise ValueError("All sims in a batch must share the same grid; "
                             "do not sweep domain.* with the GPU runner.")
    dx, dy = Lx / Nx, Ly / Ny
    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="xy")
    T = base_cfg["time"]["T"]

    bd_type = sim_cfgs[0][0]["beads"]["type"]
    if any(c["beads"]["type"] != bd_type for c, _ in sim_cfgs):
        raise ValueError("All sims in a batch must share beads.type.")

    # --- shared, conservative time step over the whole sweep ---
    scalars = [extract_scalars(cfg) for cfg, _ in sim_cfgs]
    def sim_dt(s):
        Dmax = max(s["lam_psi"] * s["D_psi"], s["lam_p"] * s["D_p"])
        return min(suggest_dt(dx, dy, Dmax, safety=0.2),
                   0.05 * min(dx, dy) / (abs(s["v0"]) + 1e-12))
    dt = min(sim_dt(s) for s in scalars)
    nsteps = int(np.round(T / dt))
    save_every = max(1, nsteps // 100)
    nsaved = nsteps // save_every + 1
    print(f"{n_sims} sims | grid {Ny}x{Nx} | T={T} | dt={dt:.4e} | "
          f"nsteps={nsteps} | save_every={save_every} | dtype={args.dtype}")

    free_vram = cp.cuda.runtime.memGetInfo()[0]
    B = args.batch or auto_batch(n_sims, Ny, Nx, nsaved, np.dtype(dtype).itemsize,
                                 free_vram, args.host_budget_gb * 1e9,
                                 args.vram_frac, len(SOLUTION_KEYS))
    print(f"batch size B={B}  ({-(-n_sims // B)} chunk(s))")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Xg = cp.asarray(X[None].astype(dtype))
    Yg = cp.asarray(Y[None].astype(dtype))

    written = 0
    for start in range(0, n_sims, B):
        chunk = list(range(start, min(start + B, n_sims)))
        bs = [scalars[i] for i in chunk]
        b = len(chunk)

        # initial conditions + bead fields
        ic = [build_ic(s, X, Y, Nx, Ny) for s in bs]
        psi0 = cp.asarray(np.stack([c[0] for c in ic]).astype(dtype))
        px0 = cp.asarray(np.stack([c[1] for c in ic]).astype(dtype))
        py0 = cp.asarray(np.stack([c[2] for c in ic]).astype(dtype))
        qx0 = cp.asarray(np.stack([c[3] for c in ic]).astype(dtype))
        qy0 = cp.asarray(np.stack([c[4] for c in ic]).astype(dtype))

        def col(key):
            return cp.asarray(np.array([s[key] for s in bs], dtype=dtype).reshape(b, 1, 1))

        if bd_type == "pol":
            rb = beads_gpu("pol", col("rbm"), Xg, Yg, col("x0"), col("y0"), col("sigma"))
        else:
            rb = beads_gpu("homo", col("rbm"), Xg, Yg, None, None, None)
        Nbd = cp.asnumpy(rb.sum(axis=(-2, -1)) * dx * dy)  # per-sim bead mass

        params = {k: col(k) for k in GPU_PARAM_KEYS}

        t, hist = simulate_rk4_batched_gpu(psi0, px0, py0, qx0, qy0, rb,
                                           params, dx, dy, dt, nsteps, save_every,
                                           progress=True)

        rb_host = cp.asnumpy(rb)
        # write one data.h5 per sim in this chunk
        for j, gi in enumerate(chunk):
            cfg = sim_cfgs[gi][0]
            leaf = sim_cfgs[gi][1]
            s = bs[j]
            folder = os.path.join(args.out, run_name(leaf),
                                  f"sim_data_{timestamp}_it_{s['seed']}")
            os.makedirs(folder, exist_ok=True)

            cfg_used = deepcopy(cfg)
            cfg_used["time"]["dt_used"] = dt
            cfg_used["time"]["save_every"] = save_every
            cfg_used["domain"]["dx"] = dx
            cfg_used["domain"]["dy"] = dy
            with open(os.path.join(folder, "config_used.json"), "w") as f:
                json.dump(cfg_used, f, indent=2)

            solution = {key: hist[key][:, j] for key in SOLUTION_KEYS}
            grid = {"x": x, "y": y, "rb": rb_host[j], "Lx": Lx, "Ly": Ly,
                    "Nx": Nx, "Ny": Ny, "dx": dx, "dy": dy, "Nbd": float(Nbd[j])}
            params_out = {k: s[k] for k in ("lam_psi", "D_psi", "v0", "lam_p",
                          "D_p", "a_p", "b_p", "chi", "alpha", "u_q", "rq", "rbm")}
            params_out.update(T_final=T, dt=dt, save_every=save_every)
            write_h5(folder, solution=solution, t=t, grid=grid,
                     params=params_out, timestamp=timestamp)
            written += 1

        # free this chunk's GPU memory before the next one
        del psi0, px0, py0, qx0, qy0, rb, params, hist
        cp.get_default_memory_pool().free_all_blocks()
        print(f"  wrote {written}/{n_sims} sims")

    print(f"Done. {written} simulations saved under {args.out}")


if __name__ == "__main__":
    main()

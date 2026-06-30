import numpy as np
from operators import suggest_dt, beads_dist
from solver_RK4 import simulate_rk4_numba
from operators import suggest_dt
from config_io import set_in_dict, parse_value, write_h5
import argparse, json, os
from copy import deepcopy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="outputs/run")
    ap.add_argument("--set", nargs="*", default=[], help="Overrides like params.rq=0.2 domain.Nx=128")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    # apply overrides
    for item in args.set:
        k, v = item.split("=", 1)
        set_in_dict(cfg, k, parse_value(v))

    ### Extract parameters from config ###
    Nx = cfg["domain"]["Nx"]
    Ny = cfg["domain"]["Ny"]
    Lx = cfg["domain"]["Lx"]
    Ly = cfg["domain"]["Ly"]

    T          = cfg["time"]["T"]

    bd_type    = cfg["beads"]["type"]
    rbm        = cfg["beads"]["rbm"]
    t_dec      = cfg["beads"]["t_dec"]
    if bd_type == "pol":
        x0         = cfg["beads"]["x0"]
        y0         = cfg["beads"]["y0"]
        sigma      = cfg["beads"]["sigma"]


    lam_psi    = cfg["params"]["lam_psi"]
    D_psi      = cfg["params"]["D_psi"]
    v0         = cfg["params"]["v0"]
    lam_p      = cfg["params"]["lam_p"]
    D_p        = cfg["params"]["D_p"]
    a_p        = cfg["params"].get("a_p", cfg["params"].get("gamma", 1.0))
    b_p        = cfg["params"].get("b_p", 1.0)
    chi        = cfg["params"].get("chi", cfg["params"].get("gamma", 0.0))
    alpha      = cfg["params"]["alpha"]
    u_q        = cfg["params"].get("u_q", 1.0)
    rq         = cfg["params"]["rq"]
    mu         = cfg["params"]["mu"]

    p_mag = cfg["ic"]["noise_mag"]

    # Generate seed if not provided in config
    if "seed" not in cfg["ic"]:
        cfg["ic"]["seed"] = np.random.randint(0, 2**31 - 1)
    
    seed  = cfg["ic"]["seed"]

    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    dx = Lx / Nx
    dy = Ly / Ny
    X, Y = np.meshgrid(x, y, indexing="xy")

    ### INITIAL CONDITIONS ###
    psi0 = -np.ones((Ny, Nx))
    # psi0 = 2 * np.exp(-((X - 0)**2 + (Y - 0.5)**2) / (2*10**2)) - 1

    # psi0 = np.zeros((Ny, Nx))
    # psi0 = 1 * (np.tanh((X - 4.5) / np.sqrt(2)))
    
    rng = np.random.default_rng(seed)
    px0 = p_mag * (2*rng.random((Ny, Nx)) - 1)
    py0 = p_mag * (2*rng.random((Ny, Nx)) - 1)

    # px0 = -np.zeros((Ny, Nx))
    # py0 = np.zeros((Ny, Nx))


    qx0 = np.zeros((Ny, Nx))
    qy0 = np.zeros((Ny, Nx))

    # clamp p on boundary
    px0[0,:]=px0[-1,:]=0; px0[:,0]=px0[:,-1]=0
    py0[0,:]=py0[-1,:]=0; py0[:,0]=py0[:,-1]=0

    # generate rb field
    rb = beads_dist(bd_type, rbm, X, Y, x0, y0, sigma) if bd_type == "pol" else beads_dist(bd_type, rbm, X, Y, None, None, None)
    Nbd = np.sum(rb) * dx * dy

    Dmax = max(lam_psi * D_psi, lam_p * D_p)
    dt_diff = suggest_dt(dx, dy, Dmax, safety=0.2)
    dt = min(dt_diff, 0.05 * min(dx,dy)/(abs(v0)+1e-12))
    print(f"Suggested dt based on diffusion: {dt_diff:.4e}, using dt={dt:.4e}")
    save_every = max(1, int(np.round(T / dt)) // 100)

    ### COPY JSON CONFIG WITH USED PARAMETERS TO OUTPUT FOLDER ###
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"{args.out}/sim_data_{timestamp}_it_{seed}"
    cfg_used = deepcopy(cfg)
    cfg_used["time"]["dt_used"] = dt
    cfg_used["time"]["save_every"] = save_every
    cfg_used["domain"]["dx"] = dx
    cfg_used["domain"]["dy"] = dy
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "config_used.json"), "w") as f:
        json.dump(cfg_used, f, indent=2)


    ### MAIN SIMULATION LOOP ###
    params = (lam_psi, D_psi, v0, lam_p, D_p, a_p, b_p, chi, alpha, u_q, rq, mu, t_dec, dx, dy, dt)

    t, psi_hist, px_hist, py_hist, qx_hist, qy_hist, P_hist, gradPx_hist, gradPy_hist, divJrho_hist = simulate_rk4_numba(
        psi0, px0, py0, qx0, qy0,
        rb, Lx, Ly, T, dt, save_every,
        params
    )


    ### SAVE DATA ###

    solution = {
        "psi_hist": psi_hist, "px_hist": px_hist, "py_hist": py_hist,
        "qx_hist": qx_hist, "qy_hist": qy_hist, "P_hist": P_hist,
        "gradPx_hist": gradPx_hist, "gradPy_hist": gradPy_hist,
        "divJrho_hist": divJrho_hist,
    }
    grid = {
        "x": x, "y": y, "rb": rb, "Lx": Lx, "Ly": Ly, "Nx": Nx, "Ny": Ny,
        "dx": dx, "dy": dy, "Nbd": Nbd,
    }
    params_out = {
        "lam_psi": lam_psi, "D_psi": D_psi, "v0": v0, "lam_p": lam_p,
        "D_p": D_p, "a_p": a_p, "b_p": b_p, "chi": chi, "alpha": alpha,
        "u_q": u_q, "rq": rq, "rbm": rbm,
        "T_final": T, "dt": dt, "save_every": save_every,
    }
    path = write_h5(folder, solution=solution, t=t, grid=grid,
                    params=params_out, timestamp=timestamp)
    print(f"Data saved to: {path}")

if __name__ == "__main__":
    main()
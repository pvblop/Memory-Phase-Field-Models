import numpy as np
from operators import suggest_dt, beads_dist
from solver_RK4 import simulate_rk4_numba
from operators import suggest_dt
import argparse, json, os
from copy import deepcopy

def set_in_dict(d, key_path, value):
    """
    key_path like 'params.rq' or 'domain.Nx'
    """
    keys = key_path.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = value

def parse_value(s):
    # tries int, float, bool, else string
    sl = s.lower()
    if sl in ("true", "false"):
        return sl == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s

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
    gamma      = cfg["params"]["gamma"]
    alpha      = cfg["params"]["alpha"]
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
    save_every = int(np.round(T / dt)) // 4

    ### COPY JSON CONFIG WITH USED PARAMETERS TO OUTPUT FOLDER ###
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"{args.out}/sim_data_{timestamp}"
    cfg_used = deepcopy(cfg)
    cfg_used["time"]["dt_used"] = dt
    cfg_used["time"]["save_every"] = save_every
    cfg_used["domain"]["dx"] = dx
    cfg_used["domain"]["dy"] = dy
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "config_used.json"), "w") as f:
        json.dump(cfg_used, f, indent=2)


    ### MAIN SIMULATION LOOP ###
    params = (lam_psi, D_psi, v0, lam_p, D_p, gamma, alpha, rq, mu, t_dec, dx, dy, dt)

    t, psi_hist, px_hist, py_hist, qx_hist, qy_hist, P_hist, gradPx_hist, gradPy_hist, divJrho_hist = simulate_rk4_numba(
        psi0, px0, py0, qx0, qy0,
        rb, Lx, Ly, T, dt, save_every,
        params
    )


    ### SAVE DATA ###

    import h5py
        # Create filename with timestamp
    filename = f"data.h5"
    os.makedirs(folder, exist_ok=True)

    # Save data to HDF5 file
    with h5py.File(f"{folder}/{filename}", 'w') as f:
        # Create groups
        solution_group = f.create_group('solution')
        params_group = f.create_group('parameters')
        grid_group = f.create_group('grid')
        
        # Save solution data
        solution_group.create_dataset('psi_hist', data=psi_hist)
        solution_group.create_dataset('px_hist', data=px_hist)
        solution_group.create_dataset('py_hist', data=py_hist)
        solution_group.create_dataset('qx_hist', data=qx_hist)
        solution_group.create_dataset('qy_hist', data=qy_hist)
        solution_group.create_dataset('P_hist', data=P_hist)
        solution_group.create_dataset('gradPx_hist', data=gradPx_hist)
        solution_group.create_dataset('gradPy_hist', data=gradPy_hist)
        solution_group.create_dataset('divJrho_hist', data=divJrho_hist)

        solution_group.create_dataset('t', data=t)
        
        # Save grid information
        grid_group.create_dataset('x', data=x)
        grid_group.create_dataset('y', data=y)
        grid_group.create_dataset('rb', data=rb)
        grid_group.attrs['Lx'] = Lx
        grid_group.attrs['Ly'] = Ly
        grid_group.attrs['Nx'] = Nx
        grid_group.attrs['Ny'] = Ny
        grid_group.attrs['dx'] = dx
        grid_group.attrs['dy'] = dy
        grid_group.attrs['Nbd'] = Nbd
        
        # Save equation parameters
        params_group.attrs['lam_psi'] = lam_psi
        params_group.attrs['D_psi'] = D_psi
        params_group.attrs['v0'] = v0
        params_group.attrs['lam_p'] = lam_p
        params_group.attrs['D_p'] = D_p
        params_group.attrs['gamma'] = gamma
        params_group.attrs['alpha'] = alpha
        params_group.attrs['rq'] = rq
        params_group.attrs['rbm'] = rbm

        # Save time parameters
        params_group.attrs['T_final'] = T
        params_group.attrs['dt'] = dt
        params_group.attrs['save_every'] = save_every
        
        # Add metadata
        f.attrs['created'] = timestamp
        f.attrs['description'] = 'Phase field simulation with coupled psi, p, q fields'

    print(f"Data saved to: {filename}")

if __name__ == "__main__":
    main()
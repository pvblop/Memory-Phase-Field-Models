"""Shared config parsing and HDF5 output helpers.

Used by both the CPU entry point (main.py) and the GPU batched sweep
runner (gpu_sweep.py) so the on-disk layout is identical.
"""
import os
import h5py


def set_in_dict(d, key_path, value):
    """key_path like 'params.rq' or 'domain.Nx'."""
    keys = key_path.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = value


def parse_value(s):
    """Tries int, float, bool, else string."""
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


# Solution-history dataset names, written under the 'solution' group.
SOLUTION_KEYS = (
    "psi_hist", "px_hist", "py_hist", "qx_hist", "qy_hist",
    "P_hist", "gradPx_hist", "gradPy_hist", "divJrho_hist",
)

# Scalar equation parameters written as attrs on the 'parameters' group.
PARAM_KEYS = (
    "lam_psi", "D_psi", "v0", "lam_p", "D_p", "a_p", "b_p",
    "chi", "alpha", "u_q", "rq", "rbm",
)


def write_h5(folder, *, solution, t, grid, params, timestamp,
             filename="data.h5"):
    """Write one simulation to ``folder/filename`` in the canonical layout.

    Parameters
    ----------
    folder : str
        Output directory (created if needed).
    solution : dict
        Maps each name in :data:`SOLUTION_KEYS` to an array of shape
        ``(nsaved, Ny, Nx)``.
    t : array
        Saved time points, shape ``(nsaved,)``.
    grid : dict
        Must contain ``x, y, rb`` arrays and scalars
        ``Lx, Ly, Nx, Ny, dx, dy, Nbd``.
    params : dict
        Must contain every name in :data:`PARAM_KEYS` plus the time
        parameters ``T_final, dt, save_every``.
    timestamp : str
        Creation timestamp stored in file metadata.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)

    with h5py.File(path, "w") as f:
        solution_group = f.create_group("solution")
        params_group = f.create_group("parameters")
        grid_group = f.create_group("grid")

        for key in SOLUTION_KEYS:
            solution_group.create_dataset(key, data=solution[key])
        solution_group.create_dataset("t", data=t)

        grid_group.create_dataset("x", data=grid["x"])
        grid_group.create_dataset("y", data=grid["y"])
        grid_group.create_dataset("rb", data=grid["rb"])
        for k in ("Lx", "Ly", "Nx", "Ny", "dx", "dy", "Nbd"):
            grid_group.attrs[k] = grid[k]

        for k in PARAM_KEYS:
            params_group.attrs[k] = params[k]
        for k in ("T_final", "dt", "save_every"):
            params_group.attrs[k] = params[k]

        f.attrs["created"] = timestamp
        f.attrs["description"] = "Phase field simulation with coupled psi, p, q fields"

    return path

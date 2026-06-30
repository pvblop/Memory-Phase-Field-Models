"""Batched RK4 time integrator on the GPU (CuPy).

Advances ``B`` simulations together (leading batch axis) with a shared time
step.  All parameters are ``(B, 1, 1)`` arrays so each sim uses its own values.
The solution history is streamed back to host memory at every ``save_every``
step, so VRAM only ever holds the working state.
"""
import numpy as np
import cupy as cp

from .RHS_gpu import compute_rhs, solve_gradP_dct, get_dct_denom
from .operators_gpu import clamp_zero_boundary, check_current_rho

# Fields streamed to host, in the order expected by config_io.write_h5.
_SAVE_FIELDS = ("psi", "px", "py", "qx", "qy", "P", "gradPx", "gradPy", "divJrho")


def _rhs(state, rb, p, dx, dy, t, denom):
    psi, px, py, qx, qy = state
    return compute_rhs(psi, px, py, qx, qy, rb, p, dx, dy, t, denom)


def simulate_rk4_batched_gpu(psi0, px0, py0, qx0, qy0, rb, params,
                             dx, dy, dt, nsteps, save_every,
                             progress=True):
    """Integrate the batch and return (t, history_dict).

    history_dict maps 'psi_hist', 'px_hist', ... 'divJrho_hist' to host
    arrays of shape (nsaved, B, Ny, Nx); ``t`` is (nsaved,).
    """
    psi = psi0.copy(); px = px0.copy(); py = py0.copy()
    qx = qx0.copy(); qy = qy0.copy()
    B, Ny, Nx = psi.shape
    dtype = psi.dtype
    denom = get_dct_denom(Ny, Nx, dx, dy, dtype)
    v0 = params["v0"]; mu = params["mu"]

    nsaved = nsteps // save_every + 1
    hist = {f"{f}_hist": np.empty((nsaved, B, Ny, Nx), dtype=np.float32)
            for f in _SAVE_FIELDS}
    out_t = np.empty(nsaved, dtype=np.float64)

    def save(s, tval):
        out_t[s] = tval
        P, gPx, gPy = solve_gradP_dct(psi, px, py, dx, dy, v0, mu, denom)
        divJ = check_current_rho(psi, px, py, gPx, gPy, v0, mu, dx, dy)
        vals = (psi, px, py, qx, qy, P, gPx, gPy, divJ)
        for f, arr in zip(_SAVE_FIELDS, vals):
            hist[f"{f}_hist"][s] = cp.asnumpy(arr.astype(np.float32))

    save(0, 0.0)

    steps = range(1, nsteps + 1)
    if progress:
        from tqdm import tqdm
        steps = tqdm(steps)

    s = 0
    t = 0.0
    half = dtype.type(0.5 * dt)
    full = dtype.type(dt)
    sixth = dtype.type(dt / 6.0)
    for n in steps:
        st = (psi, px, py, qx, qy)
        k1 = _rhs(st, rb, params, dx, dy, t, denom)[:5]
        tmp = [psi + half * k1[0], px + half * k1[1], py + half * k1[2],
               qx + half * k1[3], qy + half * k1[4]]
        clamp_zero_boundary(tmp[1]); clamp_zero_boundary(tmp[2])

        k2 = _rhs(tmp, rb, params, dx, dy, t + 0.5 * dt, denom)[:5]
        tmp = [psi + half * k2[0], px + half * k2[1], py + half * k2[2],
               qx + half * k2[3], qy + half * k2[4]]
        clamp_zero_boundary(tmp[1]); clamp_zero_boundary(tmp[2])

        k3 = _rhs(tmp, rb, params, dx, dy, t + 0.5 * dt, denom)[:5]
        tmp = [psi + full * k3[0], px + full * k3[1], py + full * k3[2],
               qx + full * k3[3], qy + full * k3[4]]
        clamp_zero_boundary(tmp[1]); clamp_zero_boundary(tmp[2])

        k4 = _rhs(tmp, rb, params, dx, dy, t + dt, denom)[:5]

        psi = psi + sixth * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        px = px + sixth * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        py = py + sixth * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        qx = qx + sixth * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
        qy = qy + sixth * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4])
        clamp_zero_boundary(px); clamp_zero_boundary(py)
        t += dt

        if n % save_every == 0:
            s += 1
            save(s, t)

    return out_t, hist

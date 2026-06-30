import os
from numba import njit, prange
import numba
import numpy as np
from RHS_eqs import compute_rhs, solve_gradP_dct
from operators import clamp_zero_boundary, check_current_rho
from tqdm import trange

# On the typical grids used here (~100x100) the per-operator parallel region
# launch overhead dominates the tiny amount of work, so a single thread is
# fastest.  The sweep scripts also run many simulations concurrently, where
# extra threads would only oversubscribe the cores.  Override with the
# SIM_THREADS environment variable for large single runs (e.g. 512x512).
_SIM_THREADS = max(1, int(os.environ.get("SIM_THREADS", "1")))
numba.set_num_threads(_SIM_THREADS)


@njit(cache=True, fastmath=True, parallel=True)
def _rk4_combine(psi, px, py, qx, qy,
                  k1_psi, k1_px, k1_py, k1_qx, k1_qy,
                  k2_psi, k2_px, k2_py, k2_qx, k2_qy,
                  k3_psi, k3_px, k3_py, k3_qx, k3_qy,
                  k4_psi, k4_px, k4_py, k4_qx, k4_qy,
                  dt):
    """Final RK4 weighted combination, fused into a single parallel loop."""
    c = dt / 6.0
    ny, nx = psi.shape
    for j in prange(ny):
        for i in range(nx):
            psi[j, i] += c * (k1_psi[j,i] + 2.0*k2_psi[j,i] + 2.0*k3_psi[j,i] + k4_psi[j,i])
            px [j, i] += c * (k1_px [j,i] + 2.0*k2_px [j,i] + 2.0*k3_px [j,i] + k4_px [j,i])
            py [j, i] += c * (k1_py [j,i] + 2.0*k2_py [j,i] + 2.0*k3_py [j,i] + k4_py [j,i])
            qx [j, i] += c * (k1_qx [j,i] + 2.0*k2_qx [j,i] + 2.0*k3_qx [j,i] + k4_qx [j,i])
            qy [j, i] += c * (k1_qy [j,i] + 2.0*k2_qy [j,i] + 2.0*k3_qy [j,i] + k4_qy [j,i])


@njit(cache=True, fastmath=True, parallel=True)
def _rk4_stage_all(psi, px, py, qx, qy,
                   kpsi, kpx, kpy, kqx, kqy,
                   coeff,
                   tpsi, tpx, tpy, tqx, tqy):
    """All five fields' stage update fused into one parallel loop:
        t* = state + coeff * k*
    Replaces five separate _stage_update dispatches per RK4 stage."""
    ny, nx = psi.shape
    for j in prange(ny):
        for i in range(nx):
            tpsi[j, i] = psi[j, i] + coeff * kpsi[j, i]
            tpx [j, i] = px [j, i] + coeff * kpx [j, i]
            tpy [j, i] = py [j, i] + coeff * kpy [j, i]
            tqx [j, i] = qx [j, i] + coeff * kqx [j, i]
            tqy [j, i] = qy [j, i] + coeff * kqy [j, i]


def rk4_step(
    psi, px, py, qx, qy,
    params,
    k1_psi, k1_px, k1_py, k1_qx, k1_qy,
    k2_psi, k2_px, k2_py, k2_qx, k2_qy,
    k3_psi, k3_px, k3_py, k3_qx, k3_qy,
    k4_psi, k4_px, k4_py, k4_qx, k4_qy,
    tmp_psi, tmp_px, tmp_py, tmp_qx, tmp_qy,
    gradPx, gradPy, Ax, Ay, divA, Fx, Fy, divF,
    gx, gy, rb, gradxpx, gradypx, gradxpy, gradypy,
    rhsP, P, tmp,
    dx, dy, v0, mu, t
):
    """
    One RK4 step.
    The Poisson solve (DCT, not iterative) is called once per stage
    *outside* the @njit boundary so scipy DCT can be used.
    
    t parameter is the current time (used for bead decay calculation).
    """
    lam_psi, D_psi, _, lam_p, D_p, a_p, b_p, chi, alpha, u_q, rq, _, t_dec, _, _, dt = params
    # Build a params tuple as expected by compute_rhs.
    rhs_params = (lam_psi, D_psi, v0, lam_p, D_p, a_p, b_p, chi, alpha, u_q, rq, mu, t_dec, dx, dy)

    # ----- k1 -----
    solve_gradP_dct(psi, px, py, gradPx, gradPy,
                    Ax, Ay, divA, rhsP, P, tmp, dx, dy, v0, mu)
    compute_rhs(psi, px, py, qx, qy,
                k1_psi, k1_px, k1_py, k1_qx, k1_qy,
                rb, gradPx, gradPy, Ax, Ay, divA, Fx, Fy, divF, gx, gy,
                gradxpx, gradypx, gradxpy, gradypy, rhsP, P, tmp,
                rhs_params, t)

    _rk4_stage_all(psi, px, py, qx, qy,
                   k1_psi, k1_px, k1_py, k1_qx, k1_qy, 0.5 * dt,
                   tmp_psi, tmp_px, tmp_py, tmp_qx, tmp_qy)
    clamp_zero_boundary(tmp_px)
    clamp_zero_boundary(tmp_py)

    # ----- k2 -----
    solve_gradP_dct(tmp_psi, tmp_px, tmp_py, gradPx, gradPy,
                    Ax, Ay, divA, rhsP, P, tmp, dx, dy, v0, mu)
    compute_rhs(tmp_psi, tmp_px, tmp_py, tmp_qx, tmp_qy,
                k2_psi, k2_px, k2_py, k2_qx, k2_qy,
                rb, gradPx, gradPy, Ax, Ay, divA, Fx, Fy, divF, gx, gy,
                gradxpx, gradypx, gradxpy, gradypy, rhsP, P, tmp,
                rhs_params, t + 0.5 * dt)

    _rk4_stage_all(psi, px, py, qx, qy,
                   k2_psi, k2_px, k2_py, k2_qx, k2_qy, 0.5 * dt,
                   tmp_psi, tmp_px, tmp_py, tmp_qx, tmp_qy)
    clamp_zero_boundary(tmp_px)
    clamp_zero_boundary(tmp_py)

    # ----- k3 -----
    solve_gradP_dct(tmp_psi, tmp_px, tmp_py, gradPx, gradPy,
                    Ax, Ay, divA, rhsP, P, tmp, dx, dy, v0, mu)
    compute_rhs(tmp_psi, tmp_px, tmp_py, tmp_qx, tmp_qy,
                k3_psi, k3_px, k3_py, k3_qx, k3_qy,
                rb, gradPx, gradPy, Ax, Ay, divA, Fx, Fy, divF, gx, gy,
                gradxpx, gradypx, gradxpy, gradypy, rhsP, P, tmp,
                rhs_params, t + 0.5 * dt)

    _rk4_stage_all(psi, px, py, qx, qy,
                   k3_psi, k3_px, k3_py, k3_qx, k3_qy, dt,
                   tmp_psi, tmp_px, tmp_py, tmp_qx, tmp_qy)
    clamp_zero_boundary(tmp_px)
    clamp_zero_boundary(tmp_py)

    # ----- k4 -----
    solve_gradP_dct(tmp_psi, tmp_px, tmp_py, gradPx, gradPy,
                    Ax, Ay, divA, rhsP, P, tmp, dx, dy, v0, mu)
    compute_rhs(tmp_psi, tmp_px, tmp_py, tmp_qx, tmp_qy,
                k4_psi, k4_px, k4_py, k4_qx, k4_qy,
                rb, gradPx, gradPy, Ax, Ay, divA, Fx, Fy, divF, gx, gy,
                gradxpx, gradypx, gradxpy, gradypy, rhsP, P, tmp,
                rhs_params, t + dt)

    # ----- final weighted update -----
    _rk4_combine(psi, px, py, qx, qy,
                 k1_psi, k1_px, k1_py, k1_qx, k1_qy,
                 k2_psi, k2_px, k2_py, k2_qx, k2_qy,
                 k3_psi, k3_px, k3_py, k3_qx, k3_qy,
                 k4_psi, k4_px, k4_py, k4_qx, k4_qy,
                 dt)
    clamp_zero_boundary(px)
    clamp_zero_boundary(py)


def simulate_rk4_numba(
    psi0, px0, py0, qx0, qy0,
    rb, Lx, Ly, T, dt, save_every,
    params
):
    lam_psi, D_psi, v0, lam_p, D_p, a_p, b_p, chi, alpha, u_q, rq, mu, t_dec, dx, dy, dt = params

    psi = psi0.copy()
    px  = px0.copy()
    py  = py0.copy()
    qx  = qx0.copy()
    qy  = qy0.copy()

    Ny, Nx = psi.shape
    dx = Lx / Nx
    dy = Ly / Ny

    nsteps = int(np.round(T / dt))
    nsaved = nsteps // save_every + 1

    # output arrays
    out_t      = np.empty(nsaved,           dtype=np.float64)
    out_psi    = np.empty((nsaved, Ny, Nx), dtype=np.float64)
    out_px     = np.empty((nsaved, Ny, Nx), dtype=np.float64)
    out_py     = np.empty((nsaved, Ny, Nx), dtype=np.float64)
    out_qx     = np.empty((nsaved, Ny, Nx), dtype=np.float64)
    out_qy     = np.empty((nsaved, Ny, Nx), dtype=np.float64)
    out_P      = np.empty((nsaved, Ny, Nx), dtype=np.float64)
    out_gradPx = np.empty((nsaved, Ny, Nx), dtype=np.float64)
    out_gradPy = np.empty((nsaved, Ny, Nx), dtype=np.float64)
    out_divJrho = np.empty((nsaved, Ny, Nx), dtype=np.float64)

    def zeros(): return np.zeros((Ny, Nx), dtype=np.float64)

    k1_psi,k1_px,k1_py,k1_qx,k1_qy = zeros(),zeros(),zeros(),zeros(),zeros()
    k2_psi,k2_px,k2_py,k2_qx,k2_qy = zeros(),zeros(),zeros(),zeros(),zeros()
    k3_psi,k3_px,k3_py,k3_qx,k3_qy = zeros(),zeros(),zeros(),zeros(),zeros()
    k4_psi,k4_px,k4_py,k4_qx,k4_qy = zeros(),zeros(),zeros(),zeros(),zeros()
    tmp_psi,tmp_px,tmp_py,tmp_qx,tmp_qy = zeros(),zeros(),zeros(),zeros(),zeros()

    Fx, Fy, divF = zeros(), zeros(), zeros()
    gx, gy       = zeros(), zeros()
    P    = zeros()
    rhsP = zeros()
    Ax, Ay, divA = zeros(), zeros(), zeros()
    tmp  = zeros()
    gradPx, gradPy = zeros(), zeros()
    gradxpx, gradypx = zeros(), zeros()
    gradxpy, gradypy = zeros(), zeros()

    # Save initial state
    s = 0
    out_t[s] = 0.0
    out_psi[s] = psi; out_px[s] = px; out_py[s] = py
    out_qx[s]  = qx;  out_qy[s] = qy
    out_P[s] = P; out_gradPx[s] = gradPx; out_gradPy[s] = gradPy
    #out_divJrho[s] = check_current_rho(psi, px, py, gradPx, gradPy, v0, mu, dx, dy)

    t = 0.0
    for n in trange(1, nsteps + 1):
        rk4_step(
            psi, px, py, qx, qy,
            params,
            k1_psi, k1_px, k1_py, k1_qx, k1_qy,
            k2_psi, k2_px, k2_py, k2_qx, k2_qy,
            k3_psi, k3_px, k3_py, k3_qx, k3_qy,
            k4_psi, k4_px, k4_py, k4_qx, k4_qy,
            tmp_psi, tmp_px, tmp_py, tmp_qx, tmp_qy,
            gradPx, gradPy, Ax, Ay, divA, Fx, Fy, divF,
            gx, gy, rb, gradxpx, gradypx, gradxpy, gradypy,
            rhsP, P, tmp,
            dx, dy, v0, mu, t
        )
        t += dt

        if n % save_every == 0:
            s += 1
            out_t[s] = t
            out_psi[s] = psi; out_px[s] = px; out_py[s] = py
            out_qx[s]  = qx;  out_qy[s] = qy
            out_P[s] = P; out_gradPx[s] = gradPx; out_gradPy[s] = gradPy

            divJrho = check_current_rho(psi, px, py, gradPx, gradPy, v0, mu, dx, dy)
            out_divJrho[s] = divJrho
            # print(np.max(np.abs(divJrho)))

    return out_t, out_psi, out_px, out_py, out_qx, out_qy, out_P, out_gradPx, out_gradPy, out_divJrho

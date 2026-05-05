import numpy as np
from numba import njit, prange
from operators import *
from scipy.fft import dctn, idctn

# Precomputed DCT eigenvalue arrays cached globally (set on first call)
_dct_cache = {}

def _get_dct_denom(ny, nx, dx, dy):
    """Cache the DCT denominator array keyed by grid shape and spacing."""
    key = (ny, nx, dx, dy)
    if key not in _dct_cache:
        kx = np.arange(nx)
        ky = np.arange(ny)
        lam_x = (2.0 * np.cos(np.pi * kx / nx) - 2.0) / dx**2
        lam_y = (2.0 * np.cos(np.pi * ky / ny) - 2.0) / dy**2
        denom = lam_y[:, None] + lam_x[None, :]
        denom[0, 0] = 1.0   # avoid divide-by-zero; overwritten below
        _dct_cache[key] = denom
    return _dct_cache[key]


def poisson_neumann_dct(rhsP, dx, dy, denom):
    """
    Solve lap(P) = rhsP on a uniform grid with homogeneous Neumann BC
    using a DCT-based fast Poisson solver.  O(N log N), no iteration.

    Returns P with zero mean.  denom is precomputed by _get_dct_denom.
    """
    rhsP = rhsP - rhsP.mean()          # ensure compatibility condition

    rhs_hat = dctn(rhsP, type=2, norm='ortho')

    P_hat = rhs_hat / denom            # vectorised divide (denom[0,0]==1)
    P_hat[0, 0] = 0.0                  # zero-mean gauge

    P = idctn(P_hat, type=2, norm='ortho')
    P -= P.mean()
    return P


@njit(cache=True, fastmath=True, parallel=True)
def compute_rhsP(psi, px, py, Ax, Ay, divA, rhsP, tmp, dx, dy, v0, mu):
    """
    Build the right-hand side of the Poisson equation for P:
        rhsP = -(1/mu) * div(v0 * m(psi) * p)
    with m(psi) = (1+psi)/2.
    """
    ny, nx = psi.shape

    # A = v0 * m * p  (fused loop, parallel)
    for j in prange(ny):
        for i in range(nx):
            m = 0.5 * (1.0 + psi[j, i])
            Ax[j, i] = v0 * m * px[j, i]
            Ay[j, i] = v0 * m * py[j, i]

    gradx_neumann(Ax, divA, dx)
    grady_neumann(Ay, tmp, dy)

    inv = -1.0 / mu
    for j in prange(ny):
        for i in range(nx):
            rhsP[j, i] = inv * (divA[j, i] + tmp[j, i])

    subtract_mean(rhsP)


def solve_gradP_dct(psi_, px_, py_, gradPx, gradPy,
                    Ax, Ay, divA, rhsP, P, tmp,
                    dx, dy, v0, mu):
    """
    Compute grad(P) satisfying  div(v0*m(psi)*p + mu*grad P) = 0
    using the fast DCT Poisson solver.  Updates gradPx/gradPy in-place.
    """
    ny, nx = psi_.shape
    denom = _get_dct_denom(ny, nx, dx, dy)

    compute_rhsP(psi_, px_, py_, Ax, Ay, divA, rhsP, tmp, dx, dy, v0, mu)

    P_new = poisson_neumann_dct(rhsP, dx, dy, denom)
    P[:, :] = P_new

    gradx_neumann(P, gradPx, dx)
    grady_neumann(P, gradPy, dy)


@njit(cache=True, fastmath=True, parallel=True)
def compute_rhs(psi_, px_, py_, qx_, qy_,
                out_psi, out_px, out_py, out_qx, out_qy,
                rb, gradPx, gradPy,
                Ax, Ay, divA,
                Fx, Fy, divF,
                gx, gy,
                gradxpx, gradypx, gradxpy, gradypy,
                rhsP, P, tmp,
                params, t):
    """
    Compute all RHS terms.  The Poisson solve for P is done *outside*
    this function (in solve_gradP_dct) and the resulting gradPx/gradPy
    are passed in — this keeps compute_rhs as a pure @njit kernel.
    """
    lam_psi, D_psi, v0, lam_p, D_p, gamma, alpha, rq, mu, t_dec, dx, dy = params
    ny, nx = psi_.shape

    # --- |q| norm (gx reused as qnorm) ---
    for j in prange(ny):
        for i in range(nx):
            gx[j, i] = np.sqrt(qx_[j, i]*qx_[j, i] + qy_[j, i]*qy_[j, i])

    # --- dpsi: flux divergence + diffusion + source ---
    laplacian_neumann(psi_, tmp, dx, dy)

    # Flux  F = v0/2*(1+psi)*p + mu*psi*gradP
    for j in prange(ny):
        for i in range(nx):
            half_v0_rho = 0.5 * v0 * (1.0 + psi_[j, i])
            Fx[j, i] = half_v0_rho * px_[j, i] + mu * (psi_[j, i]) * gradPx[j, i]
            Fy[j, i] = half_v0_rho * py_[j, i] + mu * (psi_[j, i]) * gradPy[j, i]

    gradx_neumann(Fx, divF, dx)
    grady_neumann(Fy, gy, dy)

    # Compute bead decay factor (exponential decay over time)
    bd_dec = bd_decay(t, t_dec)

    for j in prange(ny):
        for i in range(nx):
            q_norm  = gx[j, i]
            psi_val = psi_[j, i]
            term    = psi_val * (psi_val*psi_val - 1.0) \
                      - bd_dec * (rq * q_norm + rb[j, i]) * (1.0 - psi_val)
            out_psi[j, i] = -(divF[j, i] + gy[j, i]) \
                             + D_psi * tmp[j, i] \
                             - lam_psi * term

    # --- gradients of p (needed for advection) ---
    gradx_neumann(px_, gradxpx, dx)
    grady_neumann(px_, gradypx, dy)
    gradx_neumann(py_, gradxpy, dx)
    grady_neumann(py_, gradypy, dy)

    laplacian_neumann(px_, Fx, dx, dy)   # reuse Fx as lap(px)
    laplacian_neumann(py_, Fy, dx, dy)   # reuse Fy as lap(py)

    # --- p equation ---
    # Velocity field u = v0/2*(1+psi)*p + mu*gradP
    for j in prange(ny):
        for i in range(nx):
            half_v0_rho = 0.5 * v0 * (1.0 + psi_[j, i])
            ux = half_v0_rho * px_[j, i] + mu * gradPx[j, i]
            uy = half_v0_rho * py_[j, i] + mu * gradPy[j, i]

            adv_px = ux * gradxpx[j, i] + uy * gradypx[j, i]
            adv_py = ux * gradxpy[j, i] + uy * gradypy[j, i]

            # BUG FIX: original code used px_**2 + px_**2 instead of px_**2 + py_**2
            p2 = px_[j, i]*px_[j, i] + py_[j, i]*py_[j, i]

            out_px[j, i] = -lam_p * (
                px_[j, i] * (p2 - 1.0)
                - D_p * Fx[j, i]
                + gamma * (px_[j, i]- qx_[j, i])
            ) - adv_px

            out_py[j, i] = -lam_p * (
                py_[j, i] * (p2 - 1.0)
                - D_p * Fy[j, i]
                + gamma * (py_[j, i]- qy_[j, i])
            ) - adv_py

    clamp_zero_boundary(out_px)
    clamp_zero_boundary(out_py)

    # --- q equation ---
    for j in prange(ny):
        for i in range(nx):
            factor = alpha * 0.5 * (1.0 + psi_[j, i]) * (1.0 - gx[j, i])
            out_qx[j, i] = factor * px_[j, i]
            out_qy[j, i] = factor * py_[j, i]

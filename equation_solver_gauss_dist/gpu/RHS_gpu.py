"""Batched GPU right-hand side and DCT Poisson solver (CuPy).

Mirrors RHS_eqs.py but every field carries a leading batch axis and every
scalar equation parameter is a ``(B, 1, 1)`` array, so each simulation in the
batch uses its own parameters via broadcasting.

Model (Sec. 3.2 of the notes):
    dt psi = -div(F) + lam_psi*D_psi*lap(psi) - lam_psi*[ psi(psi^2-1)
                                       - bd_dec*(rq|q| + rb)(1-psi) ]
        with F = v0/2*(1+psi)*p + mu*psi*gradP
    dt p + (u.grad)p = lam_p*(-a_p p - b_p|p|^2 p + chi q + D_p lap p),  u = v0 p + mu gradP
    dt q = alpha*m(psi)*p - u_q|q|^2 q,   m(psi) = (1+psi)/2
The incompressibility pressure P solves lap(P) = -(1/mu) div(v0 m(psi) p)
with homogeneous Neumann BC, via a batched DCT fast-Poisson solve.
"""
import cupy as cp
import cupyx.scipy.fft as cfft

from .operators_gpu import (
    laplacian_neumann, gradx_neumann, grady_neumann,
    clamp_zero_boundary, subtract_mean,
)

# DCT Laplacian-eigenvalue denominators, cached per (grid, spacing, dtype).
_dct_cache = {}


def get_dct_denom(ny, nx, dx, dy, dtype):
    """Neumann-Laplacian eigenvalues for the DCT-II solve, shape (1, ny, nx)."""
    key = (ny, nx, float(dx), float(dy), cp.dtype(dtype).str)
    if key not in _dct_cache:
        kx = cp.arange(nx)
        ky = cp.arange(ny)
        lam_x = (2.0 * cp.cos(cp.pi * kx / nx) - 2.0) / dx ** 2
        lam_y = (2.0 * cp.cos(cp.pi * ky / ny) - 2.0) / dy ** 2
        denom = (lam_y[:, None] + lam_x[None, :]).astype(dtype)
        denom[0, 0] = 1.0          # avoid divide-by-zero; DC gauged to 0 below
        _dct_cache[key] = denom[None, :, :]
    return _dct_cache[key]


def poisson_neumann_dct(rhsP, denom):
    """Solve lap(P)=rhsP (Neumann, zero mean) for every sim in the batch.

    rhsP must already be zero-mean per sim (compute_rhsP enforces it).
    """
    rhs_hat = cfft.dctn(rhsP, type=2, norm="ortho", axes=(-2, -1), overwrite_x=True)
    rhs_hat /= denom
    rhs_hat[:, 0, 0] = 0.0          # zero-mean gauge per sim
    return cfft.idctn(rhs_hat, type=2, norm="ortho", axes=(-2, -1), overwrite_x=True)


def solve_gradP_dct(psi, px, py, dx, dy, v0, mu, denom):
    """Return (P, gradPx, gradPy) enforcing div(v0 m(psi) p + mu gradP)=0."""
    m = 0.5 * (1.0 + psi)
    Ax = v0 * m * px
    Ay = v0 * m * py
    rhsP = (-1.0 / mu) * (gradx_neumann(Ax, dx) + grady_neumann(Ay, dy))
    subtract_mean(rhsP)
    P = poisson_neumann_dct(rhsP, denom)
    return P, gradx_neumann(P, dx), grady_neumann(P, dy)


def compute_rhs(psi, px, py, qx, qy, rb, p, dx, dy, t, denom):
    """All RHS terms for the batch.  ``p`` is the parameter dict of (B,1,1)
    arrays.  Returns (out_psi, out_px, out_py, out_qx, out_qy, P, gradPx,
    gradPy) — P/gradP are returned so the saver can record them."""
    lam_psi = p["lam_psi"]; D_psi = p["D_psi"]; v0 = p["v0"]
    lam_p = p["lam_p"]; D_p = p["D_p"]; a_p = p["a_p"]; b_p = p["b_p"]
    chi = p["chi"]; alpha = p["alpha"]; u_q = p["u_q"]; rq = p["rq"]
    mu = p["mu"]; t_dec = p["t_dec"]

    P, gradPx, gradPy = solve_gradP_dct(psi, px, py, dx, dy, v0, mu, denom)

    # --- psi: flux divergence + diffusion + source ---
    lap_psi = laplacian_neumann(psi, dx, dy)
    half_v0_rho = 0.5 * v0 * (1.0 + psi)
    Fx = half_v0_rho * px + mu * psi * gradPx
    Fy = half_v0_rho * py + mu * psi * gradPy
    divF = gradx_neumann(Fx, dx) + grady_neumann(Fy, dy)

    q_norm = cp.sqrt(qx * qx + qy * qy)
    bd_dec = (t < t_dec)                       # (B,1,1) boolean -> broadcasts
    term = psi * (psi * psi - 1.0) - bd_dec * (rq * q_norm + rb) * (1.0 - psi)
    out_psi = -divF + lam_psi * D_psi * lap_psi - lam_psi * term

    # --- p: advection + relaxation + coupling + diffusion ---
    ux = v0 * px + mu * gradPx
    uy = v0 * py + mu * gradPy
    adv_px = ux * gradx_neumann(px, dx) + uy * grady_neumann(px, dy)
    adv_py = ux * gradx_neumann(py, dx) + uy * grady_neumann(py, dy)
    lap_px = laplacian_neumann(px, dx, dy)
    lap_py = laplacian_neumann(py, dx, dy)
    p2 = px * px + py * py
    out_px = lam_p * (-a_p * px - b_p * p2 * px + chi * qx + D_p * lap_px) - adv_px
    out_py = lam_p * (-a_p * py - b_p * p2 * py + chi * qy + D_p * lap_py) - adv_py
    clamp_zero_boundary(out_px)
    clamp_zero_boundary(out_py)

    # --- q: memory ---
    m = 0.5 * (1.0 + psi)
    q2 = qx * qx + qy * qy
    out_qx = alpha * m * px - u_q * q2 * qx
    out_qy = alpha * m * py - u_q * q2 * qy

    return out_psi, out_px, out_py, out_qx, out_qy, P, gradPx, gradPy

import numpy as np
from numba import njit, prange

# -----------------------
# Helpers: index clamping for Neumann BC
# -----------------------
@njit(inline="always")
def clamp(i, lo, hi):
    if i < lo:
        return lo
    if i > hi:
        return hi
    return i

# -----------------------
# Operators (Neumann via clamped indexing)
# -----------------------
@njit(cache=True, fastmath=True, parallel=True)
def laplacian_neumann(u, out, dx, dy):
    ny, nx = u.shape
    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)
    for j in prange(ny):
        jm = clamp(j - 1, 0, ny - 1)
        jp = clamp(j + 1, 0, ny - 1)
        for i in range(nx):
            im = clamp(i - 1, 0, nx - 1)
            ip = clamp(i + 1, 0, nx - 1)
            out[j, i] = (u[j, ip] - 2.0*u[j, i] + u[j, im]) * idx2 + (u[jp, i] - 2.0*u[j, i] + u[jm, i]) * idy2

@njit(cache=True, fastmath=True, parallel=True)
def gradx_neumann(u, out, dx):
    "compute gradient in x direction with Neumann BC via clamped indexing"
    ny, nx = u.shape
    inv2dx = 1.0 / (2.0 * dx)
    for j in prange(ny):
        for i in range(nx):
            im = clamp(i - 1, 0, nx - 1)
            ip = clamp(i + 1, 0, nx - 1)
            out[j, i] = (u[j, ip] - u[j, im]) * inv2dx

@njit(cache=True, fastmath=True, parallel=True)
def grady_neumann(u, out, dy):
    "compute gradient in y direction with Neumann BC via clamped indexing"
    ny, nx = u.shape
    inv2dy = 1.0 / (2.0 * dy)
    for j in prange(ny):
        jm = clamp(j - 1, 0, ny - 1)
        jp = clamp(j + 1, 0, ny - 1)
        for i in range(nx):
            out[j, i] = (u[jp, i] - u[jm, i]) * inv2dy

@njit(cache=True, fastmath=True)
def clamp_zero_boundary(u):
    "Set u=0 at boundaries (Dirichlet BC)"
    ny, nx = u.shape
    for i in range(nx):
        u[0, i] = 0.0
        u[ny-1, i] = 0.0
    for j in range(ny):
        u[j, 0] = 0.0
        u[j, nx-1] = 0.0

@njit(cache=True, fastmath=True)
def subtract_mean(u):
    "subtract mean(u) from u in-place"
    ny, nx = u.shape
    s = 0.0
    for j in range(ny):
        for i in range(nx):
            s += u[j, i]
    mean = s / (ny * nx)
    for j in range(ny):
        for i in range(nx):
            u[j, i] -= mean

@njit(cache=True, fastmath=True)
def check_current_rho(psi, px, py, gradPx, gradPy, v0, mu, dx, dy):
    """Check current Jrho = v0 * (1+psi)/2 * (px + py) + mu * gradPx + mu * gradPy and its divergence, for debugging."""
    ny, nx = psi.shape
    Jxrho = np.zeros((ny, nx))
    Jyrho = np.zeros((ny, nx))

    divJrho = np.zeros((ny, nx))
    tmp = np.zeros((ny, nx))

    for j in range(ny):
        for i in range(nx):
            Jxrho[j, i] = 0.5 * v0 * (1.0 + psi[j, i]) * px[j, i] + mu * gradPx[j, i]
            Jyrho[j, i] = 0.5 * v0 * (1.0 + psi[j, i]) * py[j, i] + mu * gradPy[j, i]

    # compute divergence of Jrho
    gradx_neumann(Jxrho, divJrho, dx)     # divJrho = dJxrho/dx
    grady_neumann(Jyrho, tmp, dy)       # tmp  = dJyrho/dy
    for j in range(ny):
        for i in range(nx):
            divJrho[j, i] += tmp[j, i]
    return divJrho

def suggest_dt(dx, dy, Dmax, safety=0.2):
    # For explicit schemes in 2D diffusion: dt <= 1/(2D(1/dx^2+1/dy^2))
    return safety / (2.0 * Dmax * (1.0/dx**2 + 1.0/dy**2))

def beads_dist(bd, rbm, X, Y, x0, y0, sigma):
    if bd == 'pol':
        return rbm * np.exp(-((X - x0)**2)/ (2*sigma**2) - ((Y - y0)**2 / (2*sigma**2)))
    elif bd == 'homo':
        return rbm * np.ones_like(X)
    else:
        raise ValueError(f"Unknown bd type: {bd}")

@njit
def bd_decay(t, t_dec):
    if t < t_dec:
        return 1
    else:
        return 0
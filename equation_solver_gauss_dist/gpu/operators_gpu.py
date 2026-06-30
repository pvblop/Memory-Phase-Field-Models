"""Batched, GPU (CuPy) versions of the finite-difference operators.

Every array has a leading *batch* axis: shape ``(B, Ny, Nx)``.  The spatial
stencils act on the last two axes and broadcast over the batch, so ``B``
independent simulations (one per parameter set in a sweep) advance together.

The clamped-index Neumann boundary of the CPU code (operators.py) is
reproduced exactly with ``cupy.pad(..., mode='edge')``: replicating the edge
value makes the central differences collapse to the same one-sided form the
CPU `clamp` produced (e.g. at i=0 the x-gradient becomes (u[1]-u[0])/2dx).
"""
import cupy as cp


def laplacian_neumann(u, dx, dy):
    """Neumann (zero-flux) Laplacian on the last two axes."""
    up = cp.pad(u, ((0, 0), (1, 1), (1, 1)), mode="edge")
    d2x = (up[:, 1:-1, 2:] - 2.0 * u + up[:, 1:-1, :-2]) / (dx * dx)
    d2y = (up[:, 2:, 1:-1] - 2.0 * u + up[:, :-2, 1:-1]) / (dy * dy)
    return d2x + d2y


def gradx_neumann(u, dx):
    """Central x-derivative with clamped (Neumann) boundary."""
    up = cp.pad(u, ((0, 0), (0, 0), (1, 1)), mode="edge")
    return (up[:, :, 2:] - up[:, :, :-2]) / (2.0 * dx)


def grady_neumann(u, dy):
    """Central y-derivative with clamped (Neumann) boundary."""
    up = cp.pad(u, ((0, 0), (1, 1), (0, 0)), mode="edge")
    return (up[:, 2:, :] - up[:, :-2, :]) / (2.0 * dy)


def clamp_zero_boundary(u):
    """Set u=0 on the spatial boundary (Dirichlet BC), in place."""
    u[:, 0, :] = 0.0
    u[:, -1, :] = 0.0
    u[:, :, 0] = 0.0
    u[:, :, -1] = 0.0
    return u


def subtract_mean(u):
    """Subtract the per-simulation spatial mean, in place."""
    u -= u.mean(axis=(-2, -1), keepdims=True)
    return u


def beads_dist(bd_type, rbm, X, Y, x0, y0, sigma):
    """Batched bead source field ``rb`` of shape ``(B, Ny, Nx)``.

    ``X``/``Y`` are ``(1, Ny, Nx)`` meshgrids; ``rbm, x0, y0, sigma`` are
    per-simulation arrays of shape ``(B, 1, 1)`` (or scalars).
    """
    if bd_type == "pol":
        return rbm * cp.exp(-((X - x0) ** 2) / (2 * sigma ** 2)
                            - ((Y - y0) ** 2) / (2 * sigma ** 2))
    elif bd_type == "homo":
        return rbm * cp.ones_like(X)
    else:
        raise ValueError(f"Unknown bd type: {bd_type}")


def check_current_rho(psi, px, py, gradPx, gradPy, v0, mu, dx, dy):
    """Divergence of the density current Jrho (diagnostic, batched)."""
    Jxrho = 0.5 * v0 * (1.0 + psi) * px + mu * gradPx
    Jyrho = 0.5 * v0 * (1.0 + psi) * py + mu * gradPy
    return gradx_neumann(Jxrho, dx) + grady_neumann(Jyrho, dy)

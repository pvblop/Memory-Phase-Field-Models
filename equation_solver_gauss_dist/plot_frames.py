import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import h5py


def parse_args():
    parser = argparse.ArgumentParser(description="Make frames from data.")
    parser.add_argument("--dir", "-d", type=str, required=True,
                        help="Directory containing data.h5")
    parser.add_argument("--step", type=int, default=1, help='Frame step')
    return parser.parse_args()


args = parse_args()

h5_path = os.path.join(args.dir, 'data.h5')
if os.path.isfile(h5_path):
    print(f"loading simulation data from {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        print("available keys:", list(f.keys()))
        sols = f['solution']
        psi_hist = sols['psi_hist'][:]
        px_hist = sols['px_hist'][:]
        py_hist = sols['py_hist'][:]
        qx_hist = sols['qx_hist'][:]
        qy_hist = sols['qy_hist'][:]
        divJrho_hist = sols['divJrho_hist'][:]
        t = sols['t'][:]
        params = f['grid']
        x = params['x'][:]
        y = params['y'][:]
        Lx = float(params.attrs['Lx'])
        Ly = float(params.attrs['Ly'])
        # Load bead positions if available
        if 'bead_positions' in params:
            bead_positions = params['bead_positions'][:]
        else:
            bead_positions = None
else:
    raise FileNotFoundError(f"Could not find data.h5 or data.npz inside {args.dir}")

# prepare output directory based on timestamp
base = args.dir
filename = f"{args.dir}/frames/"
os.makedirs(filename, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# rojo → amarillo
colors = ["#b2182b", "#ffffff", "#e0e000e1"]
cmap = LinearSegmentedColormap.from_list("stem_to_diff", colors)


# iterate frames
for i in range(0, len(psi_hist), args.step):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f'Time = {t[i]:.2f}')
    
    stride = 8
    Y_q, X_q = np.meshgrid(y[::stride], x[::stride], indexing='ij')

    # meso-endoderm mask: show polarization mainly where psi ~ differentiated
    m = 0.5 * (1.0 + psi_hist[i])

    # psi field + polarization direction
    im0 = axs[0].imshow(
        psi_hist[i],
        origin='lower',
        extent=[0, Lx, 0, Ly],
        cmap=cmap,
        vmin=-1,
        vmax=1
    )

    p_quiver_x = px_hist[i] * m
    p_quiver_y = py_hist[i] * m

    axs[0].quiver(
        X_q, Y_q,
        p_quiver_x[::stride, ::stride],
        p_quiver_y[::stride, ::stride],
        color='k',
        alpha=0.65,
        scale_units='xy',
        scale=0.1,
        width=0.004
    )

    if bead_positions is not None:
        axs[0].scatter(bead_positions[:, 0], bead_positions[:, 1], c='w', marker='o', alpha=0.5, edgecolors='w')

    axs[0].set_title(r'$\psi$ with $\mathbf{p}$')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    plt.colorbar(im0, ax=axs[0], label=r'$\psi$')

    # q magnitude + q direction
    q_mag = np.sqrt(qx_hist[i] ** 2 + qy_hist[i] ** 2)
    im1 = axs[1].imshow(
        q_mag,
        origin='lower',
        extent=[0, Lx, 0, Ly],
        cmap='Reds',
        vmin=0,
        vmax=1
    )

    axs[1].quiver(
        X_q, Y_q,
        qx_hist[i][::stride, ::stride],
        qy_hist[i][::stride, ::stride],
        color='k',
        alpha=0.65,
        scale_units='xy',
        scale=0.1,
        width=0.004
    )
    axs[1].set_title(r'$|\mathbf{q}|$ with $\mathbf{q}$')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    plt.colorbar(im1, ax=axs[1], label=r'$|\mathbf{q}|$')

    for ax in axs:
        ax.set_aspect('equal')

    plt.tight_layout()
    
    fig_name = f'{filename}/frame_{i:04d}.png'
    plt.savefig(fig_name, dpi=100)
    plt.close(fig)

    # plt.imshow(divJrho_hist[i], origin='lower', extent=[0, Lx, 0, Ly], cmap='Reds')
    # plt.colorbar(label=r'$\nabla \cdot J_\rho$')
    # plt.savefig(f'{filename}/divJrho_{i:04d}.png', dpi=300)
    # plt.close()

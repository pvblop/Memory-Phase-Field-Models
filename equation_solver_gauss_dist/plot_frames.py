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
else:
    raise FileNotFoundError(f"Could not find data.h5 or data.npz inside {args.dir}")

# prepare output directory based on timestamp
base = args.dir
filename = f"{args.dir}/frames/"
os.makedirs(filename, exist_ok=True)

# iterate frames
for i in range(0, len(psi_hist), args.step):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Time = {t[i]:.2f}')


    stride = 8
    Y_q, X_q = np.meshgrid(y[::stride], x[::stride], indexing='ij')

    # psi field
    im0 = axs[0].imshow(psi_hist[i], origin='lower', extent=[0, Lx, 0, Ly], cmap='RdBu', vmin=-1, vmax=1)
    axs[0].set_title(r'$\psi$')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    plt.colorbar(im0, ax=axs[0], label=r'$\psi$')

    # p field with quiver
    mag = np.sqrt(px_hist[i] ** 2 + py_hist[i] ** 2)
    m = 0.5 * (1.0 + psi_hist[i])
    # m = 1
    quiver_x = (px_hist[i] * m)
    quiver_y = (py_hist[i] * m)

    axs[1].quiver(X_q, Y_q, quiver_x[::stride, ::stride], quiver_y[::stride, ::stride], alpha=0.6, scale_units='xy', scale=0.1)
    im1 = axs[1].imshow(m*mag, origin='lower', extent=[0, Lx, 0, Ly], cmap='Reds', vmin=0, vmax=1)
    axs[1].set_title(r'$0.5(1 + \psi) \mathbf{p}$')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    plt.colorbar(im1, ax=axs[1], label=r'$|0.5(1 + \psi)\mathbf{p}|$')

    # q field with quiver
    im2 = axs[2].imshow(np.sqrt(qx_hist[i] ** 2 + qy_hist[i] ** 2), origin='lower', extent=[0, Lx, 0, Ly], cmap='Reds', vmin=0, vmax=1)
    qx_quiver = qx_hist[i] 
    qy_quiver = qy_hist[i] 
    axs[2].quiver(X_q, Y_q, qx_quiver[::stride, ::stride], qy_quiver[::stride, ::stride], alpha=0.6,  scale_units='xy', scale=0.1)
    axs[2].set_title(r'$\mathbf{q}$')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y')
    plt.colorbar(im2, ax=axs[2], label=r'$|\mathbf{q}|$')

    plt.tight_layout()
    
    fig_name = f'{filename}/frame_{i:04d}.png'
    plt.savefig(fig_name, dpi=300)
    plt.close(fig)

    plt.imshow(divJrho_hist[i], origin='lower', extent=[0, Lx, 0, Ly], cmap='Reds')
    plt.colorbar(label=r'$\nabla \cdot J_\rho$')
    plt.savefig(f'{filename}/divJrho_{i:04d}.png', dpi=300)
    plt.close()

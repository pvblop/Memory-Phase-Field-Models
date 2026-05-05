# Memory Phase Field Models — Code Summary

## Overview

This repository contains two related 2D phase-field solvers that model differentiation, polarization, and memory effects in active matter systems.

There are two main implementations:

- `equation_solver_bd_dist/`  
  Uses discrete Gaussian beads as the differentiation-triggering field.

- `equation_solver_gauss_dist/`  
  Uses a continuous field (either homogeneous or Gaussian) as the differentiation trigger.

---

## Fields in the model

The simulation evolves the following fields:

- `psi`: differentiation / order parameter  
- `p = (px, py)`: polarization field  
- `q = (qx, qy)`: memory field  
- `P`: pressure-like field enforcing a divergence constraint  

---

## Differentiation triggering field
The field that trigger the differentiation can be modeled in several ways. For all cases the field is active for a time `t_decay`. After this time this field, and `rq` are set to zero.
### Individual beads (bd_dist)
For these simulations, the differentiation field is given by individual beads which are modeled as a small Gaussian. Each bead contribution is summed. The number of beads `N_bd`, their magnitude `m_bd` and width `sig_bd` can be set in the config JSON file. The spatial distribution `dist` of the beads can be `homo`, so they are homogenously distributed across the domain or `pol`, so they are concentrated in the bottom left quarter of the domain. There is also `one`, which generates one bead at the center of the domain. 

### Gauss field (gauss_dist)
For these simulations, the differentiation field is given by a bigger continous field. There are two `types` of field, it can be Gaussian `pol` with a magnitude `rbm` and width `sigma` centered at `x0`,`y0`. Or it can be homogeneous `homo` with a value `rbm`.  

## Workflow

The simulation pipeline is:

```
JSON config → main.py → RK4 solver → data.h5 → frames → movie
```

1. Parameters are read from a JSON config file  
2. Initial conditions are generated  
3. The system is evolved using an RK4 solver  
4. Results are saved in HDF5 format  
5. Frames and movies can be generated from the output  

---

## Running a simulatuion
Use:
- `python main.py --config config/base_config.json --out outputs/run`

the JSON file must contain all simulation parameters. Note that the simulation parameters change between simulation types (see below). Parameters from the JSON config file can be override using --set:
- `python main.py --config config/base_config.json --out outputs/run --set domain.Nx=128`

The simulation creates a folder of the form:
`outputs/run/sim_data_YYYYMMDD_HHMMSS/`

which contains
- `data.h5`
- `config_used.json`

`config_used.json` is a copy of the parameter metadata. The file `data.h5` contains the simulation fields, grid, bead positions, and parameter metadata.

## Make movies and frames
To generate frames from a simulation folder:
- `python plot_frames.py --dir outputs/run/sim_data_YYYYMMDD_HHMMSS`

To generate a movie directly from a simulation folder:
- `bash make_movie.sh outputs/run/sim_data_YYYYMMDD_HHMMSS`

You also need `ffmpeg` to create movies and the `plot_frames.py` script accesible from the bash shell script.

## Main files of the code

### `main.py`
- Entry point of the simulation  
- Loads configuration from JSON  
- Applies command-line overrides (`--set`)  
- Initializes fields (`psi`, `p`, `q`)  
- Builds the differentiation field  
- Computes timestep  
- Runs the solver  
- Saves results to `data.h5` and `config_used.json`  

---

### `RHS_eqs.py`
- Defines the dynamical equations  
- Computes RHS for `psi`, `p`, and `q`  
- Solves a Poisson equation for `P` using a DCT-based solver  
- Uses `grad(P)` to enforce approximate divergence-free current  

---

### `solver_RK4.py`
- Implements explicit RK4 time stepping  
- Calls RHS at each stage  
- Uses Numba for performance  

---

### `operators.py`
- Finite-difference operators (with Neumann-like boundary conditions)  
- Timestep estimation  
- Bead/field generation  
- Boundary handling and diagnostics  

---

### `plot_frames.py`
- Reads `data.h5`  
- Generates visualization frames of:
  - `psi`
  - polarization field
  - memory field  

---

### `make_movie.sh`
- Calls `plot_frames.py`  
- Uses `ffmpeg` to generate `movie.mp4`  
- Optionally removes temporary frames  

---

## Numerical methods

The implementation uses:

- Finite differences on a 2D grid  
- Homogeneous Neumann-type boundary conditions  
- Explicit RK4 time integration  
- Numba acceleration  
- DCT-based Poisson solver  
- HDF5 for data storage  

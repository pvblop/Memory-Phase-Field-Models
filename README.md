# Memory Phase Field SImulations

This repository contains a collection of 2D phase-field simulations with coupled fields:

- `psi`: differentiation / order parameter  
- `p`: polarization field  
- `q`: memory field  
- `P`: pressure-like field  

Each subfolder corresponds to a different implementation of the **differentiation triggering field**, such as:
- individual beads (`...bd_dist`)
- continuous field (`...gauss_dist`)

---

## Repository structure
Inside each folder the structure is as follows

```text
.
├── main.py
├── operators.py
├── solver_RK4.py
├── RHS_eq.py
├── plot_frames.py
├── make_movie.sh
├── config/
│   └── base_config.json
└── outputs/
```

## Differentiation triggering field
The field that trigger the differentiation can be modeled in several ways. For all cases the field is active for a time `t_decay`. After this time this field, and `rq` are set to zero.
### Individual beads (bd_dist)
For these simulations, the differentiation field is given by individual beads which are modeled as a small Gaussian. Each bead contribution is summed. The number of beads `N_bd`, their magnitude `m_bd` and width `sig_bd` can be set in the config JSON file. The spatial distribution `dist` of the beads can be `homo`, so they are homogenously distributed across the domain or `pol`, so they are concentrated in the bottom left quarter of the domain. There is also `one`, which generates one bead at the center of the domain. 

### Gauss field (gauss_dist)
For these simulations, the differentiation field is given by a bigger continous field. There are two `types` of field, it can be Gaussian `pol` with a magnitude `rbm` and width `sigma` centered at `x0`,`y0`. Or it can be homogeneous `homo` with a value `rbm`.  

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


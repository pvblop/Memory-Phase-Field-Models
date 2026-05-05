# Memory Phase Field SImulations


Inside each folder of this repository there is code to run a 2D phase-field simulation with coupled fields:

- `psi`: differentiation/order-parameter field
- `p`: polarization field
- `q`: memory field
- `P`: pressure-like field

The main simulation script reads parameters from a JSON config file, runs the numerical solver, and saves the output in HDF5 format. The repository also includes scripts to generate frames and movies from the simulation output.
Each folder is for a different differentiation triggering field, these can be with individual beads "...bd_dist" o with a Gaussian field "...gauss_dist".



## Repository structure
Inside each folder the structure is as follows
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

## Running a simulatuion
Use:
python main.py --config config/base_config.json --out outputs/run

Parameters from the JSON config file can be override using --set:
python main.py --config config/base_config.json --out outputs/run --set domain.Nx=128

The simulation creates a folder of the form:
outputs/run/sim_data_YYYYMMDD_HHMMSS/

which contains
data.h5
config_used.json

config_used.json is a copy of the parameter metadata. The file data.h5 contains the simulation fields, grid, bead positions, and parameter metadata.

## Make movies and frames
To generate frames from a simulation folder:
python plot_frames.py --dir outputs/run/sim_data_YYYYMMDD_HHMMSS

To generate a movie directly from a simulation folder:
bash make_movie.sh outputs/run/sim_data_YYYYMMDD_HHMMSS

You also need ffmpeg to create movies and the plot_frames.py script accesible from the bash shell script.
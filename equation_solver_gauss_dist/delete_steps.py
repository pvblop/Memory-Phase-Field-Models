# reduce_h5.py
import h5py
import numpy as np
from pathlib import Path
import shutil
import sys

infile = Path(sys.argv[1])
outfile = infile.with_name("data_reduced.h5")

with h5py.File(infile, "r") as fin, h5py.File(outfile, "w") as fout:
    # copy everything except solution datasets manually
    for key in fin.keys():
        if key != "solution":
            fin.copy(key, fout)

    sol_in = fin["solution"]
    sol_out = fout.create_group("solution")

    for name, dset in sol_in.items():
        if dset.ndim >= 1:
            nt = dset.shape[0]
            keep = sorted(set([nt // 2, nt - 1]))

            sol_out.create_dataset(
                name,
                data=dset[keep],
                compression="gzip",
                compression_opts=4,
            )

            # save which original timesteps were kept
            sol_out[name].attrs["kept_indices"] = keep
        else:
            fin.copy(dset, sol_out, name=name)

print(f"saved reduced file to {outfile}")
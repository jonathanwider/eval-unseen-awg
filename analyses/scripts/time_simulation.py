import os
import timeit

import numpy as np
import yaml
from unseen_awg.simulate_trajectory import simulate_trajectory

n_runs = 10

with open("configs/paths.yaml") as file:
    paths = yaml.safe_load(file)["paths"]

path_wg = os.path.join(paths["dir_wgs"], "wg_reforecasts_5e06172f_f40e9460_1e69bda9")
probability_model = "NoRestrictions"
sigma = 2.5
seed = 0
n_days = 8100
blocksize = 30
path_out = os.path.join(paths["dir_simulations"], "traj_test.nc")

n_repeats = 10

times = np.array(
    timeit.repeat(
        lambda: simulate_trajectory(
            path_wg=path_wg,
            probability_model=probability_model,
            sigma=sigma,
            seed=seed,
            n_days=n_days,
            blocksize=blocksize,
            path_trajectory=path_out,
        ),
        number=1,
        repeat=n_repeats,
    )
)


print(f"Mean:   {times.mean():.2f} s")
print(f"Std:    {times.std():.2f} s")
print(f"Min:    {times.min():.2f} s")
print(f"Median: {np.median(times):.2f} s")

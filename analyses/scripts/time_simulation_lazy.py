import os
import timeit

import numpy as np
import yaml
from unseen_awg.probability_models import NormalProbabilityModel
from unseen_awg.time_steppers import StandardStepper
from unseen_awg.weather_generator import WeatherGenerator

n_runs = 10

with open("configs/paths.yaml") as file:
    paths = yaml.safe_load(file)["paths"]

path_wg = os.path.join(paths["dir_wgs"], "wg_reforecasts_5e06172f_f40e9460_1e69bda9")
sigma = 2.5
seed = 0
n_days = 8100
blocksize = 30
path_out = os.path.join(paths["dir_simulations"], "traj_test_lazy.nc")

n_repeats = 5

params = {
    "weather_generator.window_size": 10,
    "weather_generator.var": "geopotential_height",
    "weather_generator.similarity": "mse_similarity",
    "weather_generator.n_samples": 14,
    "weather_generator.use_precomputed_similarities": False,
    "dir_wg": os.path.join(paths["dir_wgs"], "lazy_wg_getting_timed"),
    "zarr_year_dayofyear": os.path.join(
        paths["dir_preprocessed_datasets"],
        "preprocessed_circulation_reforecasts/data_year_dayofyear_5e06172f_f40e9460.zarr",
    ),
}

wg = WeatherGenerator(params=params)


def simulate_and_store(
    blocksize, probability_model, stepper_class, n_steps, rng, path_out
):
    traj = wg.sample_trajectory(
        blocksize=blocksize,
        probability_model=probability_model,
        stepper_class=stepper_class,
        n_steps=n_steps,
        rng=rng,
        show_progressbar=True,
    )
    traj.to_netcdf(path_out)


times = np.array(
    timeit.repeat(
        lambda: simulate_and_store(
            blocksize=blocksize,
            probability_model=NormalProbabilityModel(sigma=sigma),
            stepper_class=StandardStepper,
            n_steps=int(np.ceil(n_days / blocksize)),
            rng=np.random.default_rng(seed=seed),
            path_out=path_out,
        ),
        number=1,
        repeat=n_repeats,
    )
)


print(f"Mean:   {times.mean():.2f} s")
print(f"Std:    {times.std():.2f} s")
print(f"Min:    {times.min():.2f} s")
print(f"Median: {np.median(times):.2f} s")

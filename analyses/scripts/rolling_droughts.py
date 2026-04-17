from pathlib import Path

import numpy as np
import xarray as xr
import xesmf as xe
import yaml
from icecream import ic
from metpy.units import units
from tqdm.auto import tqdm
from unseen_awg.snakemake_utils import snakemake_handler

from analyses.utils import (
    extract_datapoints_in_years,
    extract_n_years_from_trajectory,
    load_trajectories,
    stack_to_dim,
)


def eval_rolling_droughts(
    year_min: int,
    n_years: int,
    lsm_threshold: float,
    use_anomalies: bool,
    n_months_rolling: int,
    path_zarr_era5: str,
    path_era5_climatology: str,
    path_zarr_wg: str,
    paths_trajectories: list[str],
    path_lsm: str,
    path_rolling_sum_gt: str,
    path_rolling_sum_wg: str,
) -> None:
    """Compute rolling drought metrics for ground truth and generated weather data.

    This function computes rolling sums of precipitation over a specified number of
    months for both ERA5 ground truth and weather generator output. Optionally computes
    droughts as anomalies relative to climatology. Results are saved to NetCDF files.

    Parameters
    ----------
    year_min : int
        Start year of the analysis period.
    n_years : int
        Number of years to analyze.
    lsm_threshold : float
        Threshold for land-sea mask to identify land points.
    use_anomalies : bool
        If True, compute anomalies relative to climatology. If False, use raw values.
    n_months_rolling : int
        Number of months for the rolling window sum.
    path_zarr_era5 : str
        Path to Zarr store containing ERA5 precipitation data.
    path_era5_climatology : str
        Path to ERA5 climatology NetCDF file.
    path_zarr_wg : str
        Path to Zarr store containing weather generator precipitation data.
    paths_trajectories : list[str]
        List of paths to simulated time series / trajectories.
    path_lsm : str
        Path to land-sea mask NetCDF file.
    path_rolling_sum_gt : str
        Output path for ground truth rolling sum NetCDF file.
    path_rolling_sum_wg : str
        Output path for weather generator rolling sum NetCDF file.
    """
    year_max = year_min + n_years - 1

    tp = xr.open_zarr(
        path_zarr_era5,
        decode_timedelta=True,
    )["tp"]
    tp = tp.assign_coords({"valid_time": tp.init_time + tp.lead_time})
    tp = tp.where(tp.metpy.quantify() > 1 * units.millimeter, 0).metpy.dequantify()

    lsm = xr.open_dataset(path_lsm)["lsm"]
    if not (lsm.latitude.equals(tp.latitude) and lsm.longitude.equals(tp.longitude)):
        ic(
            f"LSM is at a different grid, regrid to tp grid. lat: {tp.latitude.data}, lon:{tp.longitude.data}"
        )
        regridder = xe.Regridder(lsm, tp, method="bilinear")
        lsm = regridder(lsm, keep_attrs=True)
    lsm = (lsm > lsm_threshold).squeeze()
    clim_tp = xr.open_dataset(path_era5_climatology)["tp"]
    clim_tp = clim_tp.assign_coords(dayofyear=("dayofyear", np.arange(1, 367)))

    ground_truth = (
        extract_datapoints_in_years(
            stack_to_dim(tp),
            year_max=year_max,
            year_min=year_min,
        ).swap_dims({"datapoint": "valid_time"})
    ).drop_vars("datapoint")

    assert ground_truth.dims.index("latitude") < ground_truth.dims.index("longitude")

    if use_anomalies:
        ground_truth = ground_truth - clim_tp.sel(
            dayofyear=ground_truth.valid_time.dt.dayofyear
        )

    tp_monthly = (
        ground_truth.groupby(("valid_time.year", "valid_time.month"))
        .sum()
        .stack(time=("year", "month"))
    )

    tp_rolling_sum = (
        tp_monthly.rolling(time=n_months_rolling)
        .sum()
        .isel(time=slice(n_months_rolling, None))
    )

    # now generated data:
    tp_wg = xr.open_zarr(
        path_zarr_wg,
        decode_timedelta=True,
    )["tp"]
    tp_wg = tp_wg.assign_coords({"valid_time": tp_wg.init_time + tp_wg.lead_time})
    tp_wg = tp_wg.where(
        tp_wg.metpy.quantify() > 1 * units.millimeter, 0
    ).metpy.dequantify()

    # load trajectories:
    paths_trajectories = [Path(traj).parents[0] for traj in paths_trajectories]
    trajectories = load_trajectories(paths_trajectories)

    ic("load trajectories")
    trimmed_trajs = []
    for traj in trajectories:
        trimmed_trajs.append(
            extract_n_years_from_trajectory(
                traj=traj,
                n_years=n_years,
                new_start_year=ground_truth.valid_time.dt.year.min().data,
            )
        )

    trimmed_trajs = xr.combine_by_coords(trimmed_trajs)

    tp_rolling_sum_traj = []

    for s in tqdm(trimmed_trajs.seed):
        # Select a subset of seeds
        traj_chunk = trimmed_trajs.sel(seed=s)
        # Process this chunk
        tp_chunk = (
            tp_wg.drop_vars("valid_time")
            .sel(traj_chunk)
            .rename({"out_time": "valid_time"})
        )

        if use_anomalies:
            tp_chunk = tp_chunk - clim_tp.sel(
                dayofyear=tp_chunk.valid_time.dt.dayofyear
            )

        # Perform groupby on this chunk
        tp_chunk = tp_chunk.groupby(("valid_time.year", "valid_time.month")).sum()

        # Stack and compute immediately to free memory
        tp_chunk = tp_chunk.stack(time=("year", "month")).compute()

        tp_chunk_rolling_sum = (
            tp_chunk.rolling(time=n_months_rolling)
            .sum()
            .isel(time=slice(n_months_rolling, None))
        )

        # Store the result
        tp_rolling_sum_traj.append(tp_chunk_rolling_sum.expand_dims(seed=[s.data]))
        # Explicitly delete to help with garbage collection
        del tp_chunk, traj_chunk, tp_chunk_rolling_sum

    tp_rolling_sum_traj = xr.combine_by_coords(tp_rolling_sum_traj)["tp"]

    tp_rolling_sum.reset_index("time").to_netcdf(path_rolling_sum_gt)
    tp_rolling_sum_traj.reset_index("time").to_netcdf(path_rolling_sum_wg)


@snakemake_handler
def main(snakemake) -> None:
    all_params = dict(snakemake.params.all_params)
    tracked_params = dict(snakemake.params.tracked_params)

    tracked_params["use_anomalies"] = (
        True if snakemake.wildcards.use_anomalies == "True" else False
    )
    tracked_params["n_months_rolling"] = int(snakemake.wildcards.n_months_rolling)

    all_params["use_anomalies"] = tracked_params["use_anomalies"]
    all_params["n_months_rolling"] = tracked_params["n_months_rolling"]

    with open(snakemake.output.params, "w") as f:
        yaml.dump(tracked_params, f, default_flow_style=False, sort_keys=False)

    eval_rolling_droughts(
        year_min=all_params["eval_rolling_droughts.year_min"],
        n_years=all_params["eval_rolling_droughts.n_years"],
        lsm_threshold=all_params["eval_rolling_droughts.lsm_threshold"],
        use_anomalies=all_params["use_anomalies"],
        n_months_rolling=all_params["n_months_rolling"],
        path_zarr_era5=snakemake.input.zarr_era5,
        path_era5_climatology=snakemake.input.nc_era5_climatology,
        path_zarr_wg=snakemake.input.zarr_wg,
        paths_trajectories=snakemake.input.trajectories,
        path_lsm=snakemake.config["paths"]["path_lsm"],
        path_rolling_sum_gt=snakemake.output.nc_rolling_sum_gt,
        path_rolling_sum_wg=snakemake.output.nc_rolling_sum_wg,
    )


if __name__ == "__main__":
    main(snakemake=snakemake)  # noqa: F821

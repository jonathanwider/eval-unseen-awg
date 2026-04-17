import os
from pathlib import Path

import cmcrameri as cmc  # noqa: F401
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import xarray as xr
import yaml
from icecream import ic
from unseen_awg.snakemake_utils import snakemake_handler

from analyses.utils import (
    extract_datapoints_in_months,
    extract_datapoints_in_years,
    extract_n_years_from_trajectory,
    load_trajectories,
    stack_to_dim,
    subsampled_dataarray_to_year_sample,
)


def eval_autocorrelation(
    lat_pt: float,
    lon_pt: float,
    months: list[int],
    var: str,
    year_min: int,
    n_years: int,
    subset_blocksizes: list[int],
    path_era5_climatology: str,
    path_wg_climatology: str,
    path_zarr_rechunk_era5: str,
    path_zarr_rechunk_wg: str,
    paths_trajectories: list[str],
    paths_images: list[str],
    path_autocorrelation_gt: str,
    path_autocorrelation_wg: str,
) -> None:
    """Evaluate temporal autocorrelation of generated weather for given parameters and save plots.

    This function computes the autocorrelation of anomalies of `var` for both
    ground truth (ERA5) and generated time series, comparing weather generator output for different
    `blocksize` values.

    Parameters
    ----------
    lat_pt : float
        Latitude coordinate of a sample point.
    lon_pt : float
        Longitude coordinate of a sample point.
    months : list[int]
        List of months (1-12) to analyze. Should be contiguous.
    var : str
        Name of variable  to analyze.
    year_min : int
        Start year of the analysis period.
    n_years : int
        Number of years to analyze.
    subset_blocksizes : list[int]
        Block sizes to subset for analysis.
    path_era5_climatology : str
        Path to ERA5 climatology NetCDF file.
    path_wg_climatology : str
        Path to climatology NetCDF file of the dataset that the wetaher generator uses the weather generator.
    path_zarr_rechunk_era5 : str
        Path to Zarr store containing ERA5 data chunked along the longitude dimension.
    path_zarr_rechunk_wg : str
        Path to Zarr store containing the dataset the weather generator uses.
    paths_trajectories : list[str]
        List of paths to simulated time series / trajectories.
    paths_images : list[str]
        Paths to store the generated images under.
    path_autocorrelation_gt: str
        Path to store the computed lagged autocorrelation values in for ground truth dataset.
    path_autocorrelation_wg: str
        Path to store the computed lagged autocorrelation values in for generated dataset.
    """
    months = np.array(months)
    coords_pt = {
        "latitude": lat_pt,
        "longitude": lon_pt,
    }
    year_max = year_min + n_years - 1
    assert (months[1:] - months[:-1] == 1).all()  # assert that months are consecutive

    mpl.rc_file("matplotlibrc")
    # load data:
    clim_era = xr.open_dataset(
        path_era5_climatology,
        decode_timedelta=True,
    )
    clim_era = clim_era.assign_coords(
        dayofyear=("dayofyear", clim_era.dayofyear.data + 1)
    )
    clim_wg = xr.open_dataset(
        path_wg_climatology,
        decode_timedelta=True,
    )
    if "lead_time" in clim_wg.dims:
        clim_wg = clim_wg.mean("lead_time")
    clim_wg = clim_wg.assign_coords(dayofyear=("dayofyear", clim_wg.dayofyear.data + 1))
    ds_era = xr.open_zarr(
        path_zarr_rechunk_era5,
        decode_timedelta=True,
    )
    ds_wg = xr.open_zarr(
        path_zarr_rechunk_wg,
        decode_timedelta=True,
    )

    ds_era = ds_era.assign_coords({"valid_time": ds_era.init_time + ds_era.lead_time})
    ds_wg = ds_wg.assign_coords({"valid_time": ds_wg.init_time + ds_wg.lead_time})

    ground_truth = (
        extract_datapoints_in_years(
            stack_to_dim(ds_era.sel(coords_pt, method="nearest")),
            year_max=year_max,
            year_min=year_min,
        )
        .load()
        .swap_dims({"datapoint": "valid_time"})
    ).expand_dims()

    paths_trajectories = [Path(traj).parents[0] for traj in paths_trajectories]
    trajectories = load_trajectories(paths_trajectories)  # CONTINUE HERE:

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

    data = (
        ds_wg.sel(coords_pt, method="nearest")
        .drop_vars("valid_time")
        .sel(trimmed_trajs.load())
        .load()
        .rename({"out_time": "valid_time"})
    )

    # subset to months:
    data = extract_datapoints_in_months(data, months=months)
    ground_truth = extract_datapoints_in_months(ground_truth, months)

    # get anomalies:
    data = data.assign_coords(
        dayofyear=("valid_time", data.valid_time.dt.dayofyear.data)
    )
    ic(ground_truth)
    ic(data.valid_time.dt.dayofyear)
    ground_truth = ground_truth.assign_coords(
        dayofyear=("valid_time", data.valid_time.dt.dayofyear.data)
    )

    anomalies_trajs = data[var] - clim_wg[var].sel(coords_pt, method="nearest").sel(
        dayofyear=data.dayofyear
    )
    anomalies_gt = ground_truth[var] - clim_era[var].sel(
        coords_pt, method="nearest"
    ).sel(dayofyear=ground_truth.dayofyear)

    anomalies_trajs = subsampled_dataarray_to_year_sample(anomalies_trajs)
    anomalies_gt = subsampled_dataarray_to_year_sample(anomalies_gt)

    corr_gt = xr.apply_ufunc(
        sm.tsa.acf,
        anomalies_gt,
        input_core_dims=[["i_day_included"]],
        output_core_dims=[["lag"]],
        vectorize=True,
    )

    corr_trajs = xr.apply_ufunc(
        sm.tsa.acf,
        anomalies_trajs,
        input_core_dims=[["i_day_included"]],
        output_core_dims=[["lag"]],
        vectorize=True,
    )
    corr_trajs.to_netcdf(path_autocorrelation_wg)
    corr_gt.to_netcdf(path_autocorrelation_gt)
    for i in range(len(subset_blocksizes)):
        ss_et = subset_blocksizes[: i + 1]

        _, ax = plt.subplots(1, 1)
        for i, tau in enumerate(corr_trajs.blocksize.sel(blocksize=ss_et)):
            corr_trajs.sel(blocksize=tau).mean(("seed", "year")).squeeze().plot(
                ax=ax, label=rf"$\tau =$ {tau.data}", color=f"C{i}"
            )
            ax.fill_between(
                x=corr_trajs.lag,
                y1=corr_trajs.mean("year")
                .sel(blocksize=tau)
                .quantile(0.05, dim=("seed"))
                .squeeze(),
                y2=corr_trajs.mean("year")
                .sel(blocksize=tau)
                .quantile(0.95, dim=("seed"))
                .squeeze(),
                alpha=0.2,
                color=f"C{i}",
            )

        corr_gt.mean("year").plot(ax=ax, label="ERA5", color="k")
        ax.set_ylabel("Autocorrelation")
        ax.set_xlabel("lag [days]")

        # ax.set_title(
        #     f"Leipzig cell, months: {months}, {len(corr_trajs.seed)} repetitions per elemental timestep"
        # )
        ax.set_title(r"Leipzig, autocorrelation of JJA daily $T_{2m}$")

        plt.legend()

        os.makedirs(os.path.dirname(paths_images[i]), exist_ok=True)
        plt.savefig(paths_images[i])
    ic("Saved figures")


@snakemake_handler
def main(snakemake) -> None:
    """Execute autocorrelation evaluation workflow.

    Loads parameters from Snakemake configuration, executes the autocorrelation
    evaluation, and saves tracked parameters to output file.

    Parameters
    ----------
    snakemake : snakemake.snakemake.Snakemake
        Snakemake object containing input/output paths and parameters.
    """
    all_params = dict(snakemake.params.all_params)
    tracked_params = dict(snakemake.params.tracked_params)
    with open(snakemake.output.params, "w") as f:
        yaml.dump(tracked_params, f, default_flow_style=False, sort_keys=False)

    eval_autocorrelation(
        lat_pt=all_params["eval_autocorrelation.latitude"],
        lon_pt=all_params["eval_autocorrelation.longitude"],
        months=all_params["eval_autocorrelation.months"],
        var=all_params["eval_autocorrelation.var"],
        year_min=all_params["eval_autocorrelation.year_min"],
        n_years=all_params["eval_autocorrelation.n_years"],
        subset_blocksizes=all_params["eval_autocorrelation.subset_tau"],
        path_era5_climatology=snakemake.input.nc_era5_climatology,
        path_wg_climatology=snakemake.input.nc_wg_climatology,
        path_zarr_rechunk_era5=snakemake.input.zarr_rechunk_era5,
        path_zarr_rechunk_wg=snakemake.input.zarr_rechunk_wg,
        paths_trajectories=snakemake.input.trajectories,
        paths_images=snakemake.output.images,
        path_autocorrelation_gt=snakemake.output.autocorrelation_gt,
        path_autocorrelation_wg=snakemake.output.autocorrelation_wg,
    )


if __name__ == "__main__":
    main(snakemake=snakemake)  # noqa: F821

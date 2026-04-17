from pathlib import Path

import numpy as np
import xarray as xr
import xesmf as xe
import yaml
from icecream import ic
from tqdm.auto import tqdm
from unseen_awg.snakemake_utils import snakemake_handler

from analyses.utils import (
    extract_datapoints_in_months,
    extract_datapoints_in_years,
    extract_n_years_from_trajectory,
    get_image_count_labeled_connected_components,
    get_max_cluster_area_from_binary_image,
    get_size_largest_connected_component,
    load_trajectories,
    stack_to_dim,
)


def eval_hot_day_connected_components(
    year_min: int,
    n_years: int,
    quantiles: list[float],
    months: list[int],
    use_lsm: bool,
    lsm_threshold: float,
    path_zarr_rechunk_era5: str,
    path_zarr_rechunk_wg: str,
    path_zarr_wg: str,
    paths_trajectories: list[str],
    path_lsm: str,
    path_max_fraction_area_wg: str,
    path_max_fraction_area_gt: str,
    path_quantiles_wg: str,
    path_quantiles_gt: str,
) -> None:
    """
    Evaluate connected components of hot days for ground truth and generated data.

    This function identifies spatial clusters of days exceeding local temperature quantiles
    for both ERA5 ground truth data and weather generator simulations. It computes the
    maximum cluster area (as a fraction of total domain area) for each time step and
    saves results along with quantile thresholds as NetCDF files.

    Parameters
    ----------
    year_min : int
        Start year of the analysis period.
    n_years : int
        Number of years to analyze.
    quantiles : list[float]
        List of quantile thresholds (between 0 and 1) to use for identifying hot days.
    months : list[int]
        List of months (1-12 )to analyze.
    use_lsm : bool
        If True, mask out ocean grid cells using the land-sea mask.
    lsm_threshold : float
        Threshold value for the land-sea mask. Grid cells with LSM values above this
        threshold are considered land.
    path_zarr_rechunk_era5 : str
        Path to Zarr store containing ERA5 temperature data chunked along the
        longitude dimension.
    path_zarr_rechunk_wg : str
        Path to Zarr store containing weather generator input temperature data
        chunked along the longitude dimension.
    path_zarr_wg : str
        Path to Zarr store containing full weather generator output temperature data chunked along init_time dimension.
    paths_trajectories : list[str]
        List of paths to simulated time series (trajectories) NetCDF files from the weather generator.
    path_lsm : str
        Path to land-sea mask NetCDF file.
    path_max_fraction_area_wg : str
        Output path for NetCDF file containing maximum cluster area fractions for
        generated data.
    path_max_fraction_area_gt : str
        Output path for NetCDF file containing maximum cluster area fractions for
        ground truth data.
    path_quantiles_wg : str
        Output path for NetCDF file containing temperature quantiles computed from
        weather generator data.
    path_quantiles_gt : str
        Output path for NetCDF file containing temperature quantiles computed from
        ground truth data.

    Notes
    -----
    Connected components are identified using 8-connectivity (including diagonal
    neighbors). Area weights are computed based on latitude to account for grid
    cell size variations. If `use_lsm` is True, ocean grid cells are excluded
    from the analysis.
    """
    year_max = year_min + n_years - 1
    quantiles = xr.DataArray(quantiles)
    months = np.array(months)

    # Log parameter values
    ic(f"Running with parameters: use_lsm={use_lsm}, months={months}")

    t2m = xr.open_dataset(
        path_zarr_rechunk_era5,
        decode_timedelta=True,
    )["t2m"]
    t2m = t2m.assign_coords({"valid_time": t2m.init_time + t2m.lead_time})

    lsm = xr.open_dataset(path_lsm)["lsm"]  #  stored as NetCDF4 file.
    if not (lsm.latitude.equals(t2m.latitude) and lsm.longitude.equals(t2m.longitude)):
        ic(
            f"LSM is at a different grid, regrid to t2m grid. lat: {t2m.latitude.data}, lon:{t2m.longitude.data}"
        )
        regridder = xe.Regridder(lsm, t2m, method="bilinear")
        lsm = regridder(lsm, keep_attrs=True)

    lsm = (lsm > lsm_threshold).squeeze()

    ground_truth = (
        extract_datapoints_in_years(
            stack_to_dim(t2m),
            year_max=year_max,
            year_min=year_min,
        ).swap_dims({"datapoint": "valid_time"})
    ).drop_vars("datapoint")

    ground_truth = extract_datapoints_in_months(ground_truth, months)

    assert ground_truth.dims.index("latitude") < ground_truth.dims.index("longitude")

    area_weights = np.cos(np.deg2rad(ground_truth.latitude))
    area_weights.name = "area_weights"
    area_weights = area_weights.expand_dims(
        {"longitude": ground_truth.longitude.data}, axis=-1
    )

    # normalize their sum to 1:
    area_weights = area_weights / area_weights.sum()

    quantiles_groundtruth = ground_truth.quantile(quantiles, dim="valid_time")

    exceeding_quantile = ground_truth > quantiles_groundtruth
    if use_lsm:
        exceeding_quantile = exceeding_quantile.where(lsm, other=0)

    # ic(exceeding_quantile)

    labeled_clusters, counts = xr.apply_ufunc(
        get_image_count_labeled_connected_components,
        exceeding_quantile,
        input_core_dims=[["latitude", "longitude"]],
        output_core_dims=[["latitude", "longitude"], []],
        vectorize=True,
    )

    (
        max_size_cluster_pixels,
        i_max_size_cluster_pixels,
        max_fraction_area,
        i_max_fraction_area,
    ) = xr.apply_ufunc(
        get_size_largest_connected_component,
        labeled_clusters,
        area_weights,
        counts,
        input_core_dims=[["latitude", "longitude"], ["latitude", "longitude"], []],
        output_core_dims=[[], [], [], []],
        vectorize=True,
    )

    # same for generated data:

    # load data:
    t2m_wg = xr.open_zarr(
        path_zarr_wg,
        decode_timedelta=True,
    )["t2m"]
    t2m_wg = t2m_wg.assign_coords({"valid_time": t2m_wg.init_time + t2m_wg.lead_time})

    t2m_re_rechunk = xr.open_zarr(
        path_zarr_rechunk_wg,
        decode_timedelta=True,
    )["t2m"]
    t2m_re_rechunk = t2m_re_rechunk.assign_coords(
        {"valid_time": t2m_re_rechunk.init_time + t2m_re_rechunk.lead_time}
    )

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
    trimmed_trajs = extract_datapoints_in_months(
        trimmed_trajs.rename({"out_time": "valid_time"}), months
    )

    t2m_traj_rechunk = t2m_re_rechunk.drop_vars("valid_time").sel(trimmed_trajs.load())
    t2m_traj_rechunk = extract_datapoints_in_months(t2m_traj_rechunk, months=months)

    quantiles_trajs = []
    for lon in tqdm(t2m_traj_rechunk.longitude):
        quantiles_trajs.append(
            t2m_traj_rechunk.sel(longitude=lon)
            .quantile(quantiles, dim="valid_time")
            .expand_dims({"longitude": [lon]}, axis=-1)
            .load()
        )
    quantiles_trajs = xr.combine_by_coords(quantiles_trajs)

    max_area_re = []

    for s in tqdm(trimmed_trajs.seed):
        t2m_tmp = t2m_wg.drop_vars("valid_time").sel(trimmed_trajs.sel(seed=s).load())
        t2m_tmp = extract_datapoints_in_months(t2m_tmp, months=months)
        exceeding_quantile = t2m_tmp > quantiles_trajs.sel(seed=s)
        if use_lsm:
            exceeding_quantile = exceeding_quantile.where(lsm, other=0)

        max_area_re.append(
            xr.apply_ufunc(
                get_max_cluster_area_from_binary_image,
                exceeding_quantile.load(),
                area_weights,
                input_core_dims=[
                    ["latitude", "longitude"],
                    ["latitude", "longitude"],
                ],
                output_core_dims=[[]],
                vectorize=True,
                output_dtypes=[float],
            )
            .compute()
            .expand_dims({"seed": [s.data]})
        )

    max_area_re = xr.combine_by_coords(max_area_re)
    if "valid_time" in max_area_re.coords.keys():
        max_area_re = max_area_re.drop_vars("valid_time")

    max_area_re.to_netcdf(path_max_fraction_area_wg)
    max_fraction_area.to_netcdf(path_max_fraction_area_gt)
    quantiles_trajs.to_netcdf(path_quantiles_wg)
    quantiles_groundtruth.to_netcdf(path_quantiles_gt)


@snakemake_handler
def main(snakemake):
    all_params = dict(snakemake.params.all_params)
    tracked_params = dict(snakemake.params.tracked_params)
    tracked_params["use_lsm"] = True if snakemake.wildcards.use_lsm == "True" else False
    all_params["use_lsm"] = tracked_params["use_lsm"]
    with open(snakemake.output.params, "w") as f:
        yaml.dump(tracked_params, f, default_flow_style=False, sort_keys=False)

    # Extract parameters from config
    eval_hot_day_connected_components(
        year_min=all_params["eval_hot_day_connected_components.year_min"],
        n_years=all_params["eval_hot_day_connected_components.n_years"],
        quantiles=all_params["eval_hot_day_connected_components.quantiles"],
        use_lsm=all_params["use_lsm"],
        lsm_threshold=all_params["eval_hot_day_connected_components.lsm_threshold"],
        months=all_params["eval_hot_day_connected_components.months"],
        path_zarr_rechunk_era5=snakemake.input.zarr_rechunk_era5,
        path_zarr_rechunk_wg=snakemake.input.zarr_rechunk_wg,
        path_zarr_wg=snakemake.input.zarr_wg,
        paths_trajectories=snakemake.input.trajectories,
        path_lsm=snakemake.config["paths"]["path_lsm"],
        path_max_fraction_area_gt=snakemake.output.nc_max_fraction_area_gt,
        path_max_fraction_area_wg=snakemake.output.nc_max_fraction_area_wg,
        path_quantiles_gt=snakemake.output.nc_quantiles_gt,
        path_quantiles_wg=snakemake.output.nc_quantiles_wg,
    )


if __name__ == "__main__":
    main(snakemake=snakemake)  # noqa: F821

from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from metpy.units import units
from tqdm.auto import tqdm
from unseen_awg.snakemake_utils import snakemake_handler

from analyses.utils import (
    extract_datapoints_in_months,
    extract_datapoints_in_years,
    extract_n_years_from_trajectory,
    load_trajectories,
    stack_to_dim,
)

MONTHS = {
    "DJF": np.array([12, 1, 2]),
    "MAM": np.array([3, 4, 5]),
    "JJA": np.array([6, 7, 8]),
    "SON": np.array([9, 10, 11]),
}


def eval_quantiles(
    year_min: int,
    n_years: int,
    quantiles: list[float],
    months: str,
    path_zarr_rechunk_era5: str,
    path_zarr_rechunk_wg: str,
    paths_trajectories: list[str],
    path_quantiles_gt: str,
    path_quantiles_wg: str,
    path_quantiles_yearly_gt: str,
    path_quantiles_yearly_wg: str,
) -> None:
    year_max = year_min + n_years - 1
    quantiles = xr.DataArray(quantiles)
    months = MONTHS[months]

    ds_era5 = xr.open_zarr(
        path_zarr_rechunk_era5,
        decode_timedelta=True,
    )

    ds_wg = xr.open_zarr(
        path_zarr_rechunk_wg,
        decode_timedelta=True,
    )

    ds_era5 = ds_era5.assign_coords(
        {"valid_time": ds_era5.init_time + ds_era5.lead_time}
    )
    ds_wg = ds_wg.assign_coords({"valid_time": ds_wg.init_time + ds_wg.lead_time})

    if "tp" in ds_era5.data_vars:
        ds_era5["tp"] = (
            ds_era5["tp"]
            .where(ds_era5["tp"].metpy.quantify() > 1 * units.millimeter, 0)
            .metpy.dequantify()
        )
    else:
        raise ValueError("ds_era5 doesn't contain variable 'tp'.")

    if "tp" in ds_wg.data_vars:
        ds_wg["tp"] = (
            ds_wg["tp"]
            .where(ds_wg["tp"].metpy.quantify() > 1 * units.millimeter, 0)
            .metpy.dequantify()
        )
    else:
        raise ValueError("ds_wg doesn't contain variable 'tp'.")

    xr.testing.assert_equal(ds_era5.latitude, ds_wg.latitude)
    xr.testing.assert_equal(ds_era5.longitude, ds_wg.longitude)

    ground_truth = (
        extract_datapoints_in_years(
            stack_to_dim(ds_era5),
            year_max=year_max,
            year_min=year_min,
        ).swap_dims({"datapoint": "valid_time"})
    ).expand_dims()

    ground_truth["tp"] = ground_truth["tp"].where(ground_truth["tp"] != 0)
    qs_gt_all_months = ground_truth.quantile(
        quantiles, dim="valid_time", skipna=True
    ).load()
    data_gt = extract_datapoints_in_months(ground_truth, months=months)
    qs_gt = data_gt.quantile(quantiles, dim="valid_time", skipna=True).load()

    paths_trajectories = [Path(traj).parents[0] for traj in paths_trajectories]
    trajectories = load_trajectories(paths_trajectories)
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

    qs_trajs = []
    qs_trajs_all_months = []
    for lon in tqdm(ds_wg.longitude):
        data = (
            ds_wg.sel(longitude=lon)
            .drop_vars("valid_time")
            .sel(trimmed_trajs.load())
            .rename({"out_time": "valid_time"})
        )
        # For precip only consider non-zero precipitation values when computing quantiles.
        data["tp"] = data["tp"].where(data["tp"] != 0)
        qs_trajs_all_months.append(
            data.load()
            .quantile(quantiles, dim=("valid_time", "seed"), skipna=True)
            .expand_dims({"longitude": [lon]})
        )
        data = extract_datapoints_in_months(data, months=months).load()
        data = extract_datapoints_in_months(data, months=months).load()
        data = data.quantile(quantiles, dim=("valid_time", "seed"), skipna=True)
        qs_trajs.append(data.expand_dims({"longitude": [lon]}))

    qs_trajs = xr.combine_by_coords(qs_trajs)
    qs_trajs_all_months = xr.combine_by_coords(qs_trajs_all_months)

    qs_gt.to_netcdf(path_quantiles_gt)
    qs_trajs.to_netcdf(path_quantiles_wg)

    qs_gt_all_months.to_netcdf(path_quantiles_yearly_gt)
    qs_trajs_all_months.to_netcdf(path_quantiles_yearly_wg)


@snakemake_handler
def main(snakemake):
    all_params = dict(snakemake.params.all_params)
    tracked_params = dict(snakemake.params.tracked_params)

    tracked_params["months"] = snakemake.wildcards.months
    all_params["months"] = tracked_params["months"]

    with open(snakemake.output.params, "w") as f:
        yaml.dump(tracked_params, f, default_flow_style=False, sort_keys=False)

    eval_quantiles(
        year_min=all_params["eval_quantile_maps.year_min"],
        n_years=all_params["eval_quantile_maps.n_years"],
        quantiles=all_params["eval_quantile_maps.quantiles"],
        months=all_params["months"],
        path_zarr_rechunk_era5=snakemake.input.zarr_rechunk_era5,
        path_zarr_rechunk_wg=snakemake.input.zarr_rechunk_wg,
        paths_trajectories=snakemake.input.trajectories,
        path_quantiles_gt=snakemake.output.nc_quantiles_gt,
        path_quantiles_wg=snakemake.output.nc_quantiles_wg,
        path_quantiles_yearly_gt=snakemake.output.nc_quantiles_yearly_gt,
        path_quantiles_yearly_wg=snakemake.output.nc_quantiles_yearly_wg,
    )


if __name__ == "__main__":
    main(snakemake=snakemake)  # noqa: F821

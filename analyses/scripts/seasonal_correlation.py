# imports
from pathlib import Path
from typing import Tuple  # noqa: F401

import dcor  # noqa: F401
import numpy as np
import xarray as xr
import yaml
from hyppo.ksample import Energy  # noqa: F401
from icecream import ic
from metpy.units import units
from numpy.typing import NDArray  # noqa: F401
from scipy.signal import detrend
from scipy.stats import false_discovery_control, pearsonr, spearmanr  # noqa: F401
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler  # noqa: F401
from tqdm.auto import tqdm
from unseen_awg.snakemake_utils import snakemake_handler

from analyses.utils import (
    extract_datapoints_in_months,
    extract_datapoints_in_years,
    extract_n_years_from_trajectory,
    load_trajectories,
    stack_to_dim,
)

"""
def get_energy_distance_p_value(
    t2m_gt: NDArray,
    tp_gt: NDArray,
    t2m_traj: NDArray,
    tp_traj: NDArray,
    scaler: MinMaxScaler,
) -> Tuple[NDArray, NDArray]:
    data_t = scaler.transform(np.stack([t2m_gt, tp_gt], axis=1))
    data_g = scaler.transform(np.stack([t2m_traj, tp_traj], axis=1))
    return dcor.energy_distance(data_t, data_g), Energy().test(data_g, data_t)[1]
"""


def detrend_and_correlate(t2m_yearly, tp_yearly, print_diagnostics=False):
    t2m_yearly_detrended = xr.apply_ufunc(
        detrend,
        t2m_yearly,
        input_core_dims=[["year"]],
        output_core_dims=[["year"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    tp_yearly_detrended = xr.apply_ufunc(
        detrend,
        tp_yearly,
        input_core_dims=[["year"]],
        output_core_dims=[["year"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    # Check for constant arrays
    if (
        (t2m_yearly_detrended.std(dim="year") == 0).any()
        or (tp_yearly_detrended.std(dim="year") == 0)
    ).any() and print_diagnostics:
        ic("Constant input array detected:")
        ic("t2m_yearly:")
        ic(t2m_yearly)
        ic("tp_yearly:")
        ic(tp_yearly)
        ic("t2m_yearly_detrended:")
        ic(t2m_yearly_detrended)
        ic("tp_yearly_detrended:")
        ic(tp_yearly_detrended)

    r_pearson, p_pearson = xr.apply_ufunc(
        pearsonr,
        t2m_yearly_detrended,
        tp_yearly_detrended,
        input_core_dims=[["year"], ["year"]],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float],
    )
    r_spearman, p_spearman = xr.apply_ufunc(
        spearmanr,
        t2m_yearly_detrended,
        tp_yearly_detrended,
        input_core_dims=[["year"], ["year"]],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float],
    )

    return (
        (r_pearson, p_pearson),
        (r_spearman, p_spearman),
    )


class PrescribedSigmaScaler(BaseEstimator, TransformerMixin):
    def __init__(self, sigma):
        self.sigma = sigma

    def fit(self, X, y=None):
        # No fitting necessary, as mu and sigma are prescribed
        return self

    def transform(self, X):
        return X / self.sigma

    def inverse_transform(self, X_scaled):
        return X_scaled * self.sigma


def eval_seasonal_correlation(
    year_min: int,
    n_years: int,
    months: list[int],
    path_zarr_rechunk_era5: str,
    path_zarr_rechunk_wg: str,
    paths_trajectories: list[str],
    path_tas_pr_pearsonr: str,
    path_tas_pr_pearsonp: str,
    path_tas_pr_spearmanr: str,
    path_tas_pr_spearmanp: str,
    # path_e_dist: str,
    # path_p_value_e_dist: str,
) -> None:
    """Evaluate seasonal correlation between temperature and precipitation.

    This function computes Pearson and Spearman correlations between temperature
    and precipitation for both ground truth (ERA5) and weather generator output,
    as well as energy distance metrics. Results are saved to NetCDF files with
    multiple comparison correction applied.

    Parameters
    ----------
    year_min : int
        Start year of the analysis period.
    n_years : int
        Number of years to analyze.
    months : list[int]
        List of months (1-12) to analyze.
    path_zarr_rechunk_era5 : str
        Path to Zarr store containing ERA5 data chunked along the longitude dimension.
    path_zarr_rechunk_wg : str
        Path to Zarr store containing the dataset the weather generator uses.
    paths_trajectories : list[str]
        List of paths to simulated time series / trajectories.
    path_tas_pr_pearsonr : str
        Output path for Pearson correlation coefficient NetCDF file.
    path_tas_pr_pearsonp : str
        Output path for Pearson correlation p-value NetCDF file.
    path_tas_pr_spearmanr : str
        Output path for Spearman correlation coefficient NetCDF file.
    path_tas_pr_spearmanp : str
        Output path for Spearman correlation p-value NetCDF file.
    """
    """
        path_e_dist : str
        Output path for energy distance NetCDF file.
    path_p_value_e_dist : str
        Output path for energy distance p-value NetCDF file.
    """

    year_max = year_min + n_years - 1
    months = np.array(months)

    # load data:
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

    longitudes = ds_era5.longitude

    # extract trajectories:
    ground_truth = (
        extract_datapoints_in_years(
            stack_to_dim(ds_era5),
            year_max=year_max,
            year_min=year_min,
        )
        .load()
        .swap_dims({"datapoint": "valid_time"})
    ).expand_dims(
        {
            "seed": [-1],
            "blocksize": [-1],
            "sigma": [-1],
            "WG": ["Ground Truth"],
        }
    )
    t2m_ground_truth = (
        extract_datapoints_in_months(ground_truth["t2m"], months)
        .groupby("valid_time.year")
        .mean()
    )
    tp_ground_truth = (
        extract_datapoints_in_months(ground_truth["tp"], months)
        .groupby("valid_time.year")
        .sum()
    )
    (
        (tas_pr_pearsonr_gt, tas_pr_pearsonp_gt),
        (tas_pr_spearmanr_gt, tas_pr_spearmanp_gt),
    ) = detrend_and_correlate(t2m_ground_truth, tp_ground_truth)

    # get scaler for energy distance (later)
    # sigma_t2m = (
    #     t2m_ground_truth.std(("year")).mean(("latitude", "longitude")).data.flatten()
    # )
    # sigma_tp = (
    #     tp_ground_truth.std(("year")).mean(("latitude", "longitude")).data.flatten()
    # )

    # ic("sigma_t2m: ", sigma_t2m)
    # ic("sigma_tp: ", sigma_tp)

    # scaler = PrescribedSigmaScaler(sigma=np.array([sigma_t2m, sigma_tp]).squeeze())

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
    trimmed_trajs = xr.combine_by_coords(trimmed_trajs).load()

    tas_pr_pearsonr_traj = []
    tas_pr_pearsonp_traj = []
    tas_pr_spearmanr_traj = []
    tas_pr_spearmanp_traj = []

    # e_dist_traj = []
    # p_value_e_dist_traj = []

    for lon in tqdm(longitudes):
        data = (
            ds_wg.sel(longitude=lon)
            .drop_vars("valid_time")
            .sel(trimmed_trajs.load())
            .load()
            .rename({"out_time": "valid_time"})
        )
        # have to do this in a slightly weird way because of leap years...
        data["valid_time"] = data["valid_time"] + (
            ground_truth["valid_time"][0] - data["valid_time"][0]
        )
        data = data.where(data.valid_time.dt.year <= year_max, drop=True)
        datasets_trajs = extract_datapoints_in_months(data, months).expand_dims(
            {
                "WG": ["Generated"],
            }
        )

        t2m_summer_trajs = datasets_trajs["t2m"].groupby("valid_time.year").mean()
        tp_summer_trajs = datasets_trajs["tp"].groupby("valid_time.year").sum()
        # detrend WG data
        ((r_pearson, p_pearson), (r_spearman, p_spearman)) = detrend_and_correlate(
            t2m_summer_trajs, tp_summer_trajs
        )

        """
        e_dist_single_lon, p_values_single_lon = xr.apply_ufunc(
            get_energy_distance_p_value,
            t2m_ground_truth.sel(
                longitude=lon,
                blocksize=-1,
                seed=-1,
                sigma=-1,
                WG="Ground Truth",
            ).rename({"year": "sdim"}),
            tp_ground_truth.sel(
                longitude=lon,
                blocksize=-1,
                seed=-1,
                sigma=-1,
                WG="Ground Truth",
            ).rename({"year": "sdim"}),
            t2m_summer_trajs.stack(sdim2=("seed", "year")),
            tp_summer_trajs.stack(sdim2=("seed", "year")),
            kwargs={"scaler": scaler},
            input_core_dims=[["sdim"], ["sdim"], ["sdim2"], ["sdim2"]],
            output_core_dims=[[], []],
            vectorize=True,
        )

        e_dist_traj.append(e_dist_single_lon.expand_dims({"longitude": [lon]}))
        p_value_e_dist_traj.append(
            p_values_single_lon.expand_dims({"longitude": [lon]})
        )
        """

        tas_pr_pearsonr_traj.append(r_pearson.expand_dims({"longitude": [lon]}))
        tas_pr_pearsonp_traj.append(p_pearson.expand_dims({"longitude": [lon]}))
        tas_pr_spearmanr_traj.append(r_spearman.expand_dims({"longitude": [lon]}))
        tas_pr_spearmanp_traj.append(p_spearman.expand_dims({"longitude": [lon]}))

    tas_pr_pearsonr_traj = xr.combine_by_coords(tas_pr_pearsonr_traj).to_dataarray()
    tas_pr_pearsonp_traj = xr.combine_by_coords(tas_pr_pearsonp_traj).to_dataarray()
    tas_pr_spearmanr_traj = xr.combine_by_coords(tas_pr_spearmanr_traj).to_dataarray()
    tas_pr_spearmanp_traj = xr.combine_by_coords(tas_pr_spearmanp_traj).to_dataarray()

    # e_dist_traj = xr.combine_by_coords(e_dist_traj)
    # p_value_e_dist_traj = xr.combine_by_coords(p_value_e_dist_traj)

    # correct for multiple comparisons:
    # p_value_e_dist_traj = xr.apply_ufunc(
    #     false_discovery_control, p_value_e_dist_traj, kwargs={"method": "by"}
    # )

    ic(pearsonr)
    ic(tas_pr_pearsonr_traj)
    ic(tas_pr_pearsonr_gt)

    tas_pr_pearsonr = xr.concat(
        [
            tas_pr_pearsonr_traj,
            tas_pr_pearsonr_gt,
        ],
        dim="WG",
        join="outer",
    )
    tas_pr_pearsonp = xr.concat(
        [tas_pr_pearsonp_traj, tas_pr_pearsonp_gt], dim="WG", join="outer"
    )

    tas_pr_spearmanr = xr.concat(
        [tas_pr_spearmanr_traj, tas_pr_spearmanr_gt], dim="WG", join="outer"
    )
    tas_pr_spearmanp = xr.concat(
        [tas_pr_spearmanp_traj, tas_pr_spearmanp_gt], dim="WG", join="outer"
    )

    # ic(tas_pr_pearsonr.dims)
    tas_pr_pearsonr = tas_pr_pearsonr.drop_vars("datapoint")
    tas_pr_pearsonp = tas_pr_pearsonp.drop_vars("datapoint")
    tas_pr_spearmanr = tas_pr_spearmanr.drop_vars("datapoint")
    tas_pr_spearmanp = tas_pr_spearmanp.drop_vars("datapoint")

    # e_dist_traj = e_dist_traj.drop_vars("datapoint")
    # p_value_e_dist_traj = p_value_e_dist_traj.drop_vars("datapoint")

    # store results:
    tas_pr_pearsonr.to_netcdf(path_tas_pr_pearsonr)
    tas_pr_pearsonp.to_netcdf(path_tas_pr_pearsonp)
    tas_pr_spearmanr.to_netcdf(path_tas_pr_spearmanr)
    tas_pr_spearmanp.to_netcdf(path_tas_pr_spearmanp)
    # e_dist_traj.to_netcdf(path_e_dist)
    # p_value_e_dist_traj.to_netcdf(path_p_value_e_dist)


@snakemake_handler
def main(snakemake) -> None:
    """Execute seasonal correlation evaluation workflow.

    Loads parameters from Snakemake configuration, executes the seasonal correlation
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

    eval_seasonal_correlation(
        year_min=all_params["eval_seasonal_correlation.year_min"],
        n_years=all_params["eval_seasonal_correlation.n_years"],
        months=all_params["eval_seasonal_correlation.months"],
        path_zarr_rechunk_era5=snakemake.input.zarr_rechunk_era5,
        path_zarr_rechunk_wg=snakemake.input.zarr_rechunk_wg,
        paths_trajectories=snakemake.input.trajectories,
        path_tas_pr_pearsonr=snakemake.output.nc_tas_pr_pearsonr,
        path_tas_pr_pearsonp=snakemake.output.nc_tas_pr_pearsonp,
        path_tas_pr_spearmanr=snakemake.output.nc_tas_pr_spearmanr,
        path_tas_pr_spearmanp=snakemake.output.nc_tas_pr_spearmanp,
        # path_e_dist=snakemake.output.nc_e_dist,
        # path_p_value_e_dist=snakemake.output.nc_p_value_e_dist,
    )


if __name__ == "__main__":
    main(snakemake=snakemake)  # noqa: F821

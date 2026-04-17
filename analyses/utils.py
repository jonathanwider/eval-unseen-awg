"""Utilities for analyzing weather generator trajectories and datasets.

This module provides functions for loading trajectory datasets, extracting
temporal subsets, and analyzing connected components in binary images.
"""

import os
from typing import Any

import numpy as np
import skimage.measure
import xarray as xr
import yaml
from numpy.typing import NDArray


def load_datasets_and_configs(
    paths_trajectories: list[str],
) -> tuple[list[dict[str, Any]], list[xr.Dataset]]:
    """Load trajectory datasets and their corresponding configuration files.

    Parameters
    ----------
    paths_trajectories : list[str]
        List of paths to directories containing trajectory and parameter files.

    Returns
    -------
    tuple[list[dict[str, Any]], list[xr.Dataset]]
        A tuple containing:

        - List of configuration dictionaries loaded from params.yaml files.
        - List of xarray Datasets with dimensions expanded to make trajectory
          unique (seed, probability_model, sigma, blocksize).

    Notes
    -----
    Each trajectory is expanded with seed, probability model, sigma, and
    blocksize dimensions from the configuration file. Errors in loading
    individual trajectories are printed but do not halt processing.

    Examples
    --------
    >>> configs, trajectories = load_datasets_and_configs(
    ...     ["/path/to/trajectory1", "/path/to/trajectory2"]
    ... )
    """
    trajectories: list[xr.Dataset] = []
    configs: list[dict[str, Any]] = []

    for path in paths_trajectories:
        traj_path = os.path.join(path, "trajectory.nc")
        try:
            config_path = os.path.join(path, "params.yaml")
            with open(config_path) as file:
                config = yaml.safe_load(file)
                configs.append(config)

            trajectory = xr.open_dataset(traj_path, decode_timedelta=True).expand_dims(
                {
                    "seed": [config["seed"]],
                    "probability_model": [
                        config["simulate_trajectory.probability_model"]
                    ],
                    "sigma": [config["sigma"]],
                    "blocksize": [config["blocksize"]],
                }
            )
            trajectories.append(trajectory)

        except Exception as e:
            raise RuntimeError(f"Error loading {traj_path}: {e!s}") from e

    return configs, trajectories


def load_trajectories(paths_trajectories: list[str]) -> list[xr.Dataset]:
    """Load trajectory datasets from directories.

    Parameters
    ----------
    paths_trajectories : list[str]
        List of paths to directories containing trajectory and parameter files.

    Returns
    -------
    list[xr.Dataset]
        List of xarray Datasets with dimensions expanded to make trajectory
        unique (seed, probability_model, sigma, blocksize).

    Notes
    -----
    Each trajectory is expanded with seed, probability model, sigma, and
    blocksize dimensions from the configuration file. Errors in loading
    individual trajectories are printed but do not halt processing.

    Examples
    --------
    >>> trajectories = load_trajectories(
    ...     ["/path/to/trajectory1", "/path/to/trajectory2"]
    ... )
    """
    trajectories: list[xr.Dataset] = []

    for path in paths_trajectories:
        traj_path = os.path.join(path, "trajectory.nc")
        try:
            config_path = os.path.join(path, "params.yaml")
            with open(config_path) as file:
                config = yaml.safe_load(file)
            trajectories.append(
                xr.open_dataset(traj_path, decode_timedelta=True).expand_dims(
                    {
                        "seed": [config["seed"]],
                        "probability_model": [
                            config["simulate_trajectory.probability_model"]
                        ],
                        "sigma": [config["sigma"]],
                        "blocksize": [config["blocksize"]],
                    }
                )
            )
        except Exception as e:
            raise RuntimeError(f"Error loading {traj_path}: {e!s}") from e
    return trajectories


def extract_n_years_from_trajectory(
    traj: xr.Dataset, n_years: int, new_start_year: int
) -> xr.Dataset:
    """Extract a contiguous n-year period from a trajectory dataset.

    Parameters
    ----------
    traj : xr.Dataset
        Input trajectory dataset with an 'out_time' coordinate.
    n_years : int
        Number of years to extract.
    new_start_year : int
        The year to use as the start of the extracted period.

    Returns
    -------
    xr.Dataset
        Trajectory dataset trimmed to n_years starting from new_start_year,
        with time coordinates adjusted accordingly.

    Raises
    ------
    AssertionError
        If the trimmed trajectory does not contain enough time steps for the
        requested n_years period.

    Notes
    -----
    The function removes the first year of data (to avoid edge effects),
    then extracts n_years of data and shifts the time coordinates to start
    at new_start_year.

    Examples
    --------
    >>> extracted = extract_n_years_from_trajectory(
    ...     traj, n_years=10, new_start_year=2000
    ... )
    """
    all_years = np.unique(traj.out_time.dt.year)

    # to assure that the year starts on January first.
    trimmed_traj = extract_datapoints_in_years(
        traj.rename({"out_time": "valid_time"}),
        year_min=np.amin(all_years) + 1,
        year_max=np.inf,
    ).rename({"valid_time": "out_time"})

    expected_days = (
        np.datetime64(f"{new_start_year + n_years - 1}-12-31")
        - np.datetime64(f"{new_start_year}-01-01")
    ) / np.timedelta64(1, "D")

    if len(trimmed_traj.out_time) < expected_days:
        raise ValueError("Not enough time steps in trimmed time series.")

    trimmed_traj["out_time"] = trimmed_traj["out_time"] + (
        np.datetime64(f"{new_start_year}-01-01", "ns") - trimmed_traj["out_time"][0]
    )
    return trimmed_traj.where(
        trimmed_traj.out_time.dt.year <= new_start_year + n_years - 1, drop=True
    )


def get_image_count_labeled_connected_components(
    image: NDArray[np.bool_ | np.integer[Any]],
) -> tuple[NDArray[np.integer[Any]], int]:
    """Label connected components in a binary image and return count.

    Parameters
    ----------
    image : NDArray
        Binary image array where True/non-zero values represent foreground.

    Returns
    -------
    tuple[NDArray, int]
        A tuple containing:

        - Labeled image where each connected component has a unique integer label.
        - Total count of connected components found.

    Examples
    --------
    >>> labeled, count = get_image_count_labeled_connected_components(binary_image)
    """
    labeled_image, count = skimage.measure.label(image, return_num=True)
    return labeled_image, count


def get_image_labeled_largest_connected_component(
    image: NDArray[np.bool_ | np.integer[Any]],
    area_weights: NDArray[np.floating[Any]] | None = None,
) -> NDArray[np.bool_]:
    """Extract the largest connected component from a binary image.

    Parameters
    ----------
    image : NDArray
        Binary image array where True/non-zero values represent foreground.
    area_weights : NDArray, optional
        Array of weights (e.g., grid cell areas) with same shape as image.
        If provided, the largest component is determined by weighted area
        rather than pixel count.

    Returns
    -------
    NDArray
        Binary image containing only the largest connected component.

    Examples
    --------
    >>> largest_component = get_image_labeled_largest_connected_component(binary_image)
    >>> largest_component = get_image_labeled_largest_connected_component(
    ...     binary_image, weights
    ... )
    """
    labeled_image = skimage.measure.label(image, return_num=False)

    unique_labels, counts = np.unique(labeled_image, return_counts=True)
    nonzero_unique_labels = unique_labels[unique_labels >= 1]
    nonzero_counts = counts[unique_labels >= 1]
    
    # If no area weights provided, use pixel count to determine largest component
    if area_weights is None:
        l_max = nonzero_unique_labels[nonzero_counts.argmax()]
    else:
        # Find component with largest weighted area
        max_weighted_area = 0
        l_max = -1
        for i in nonzero_unique_labels:
            weighted_area = np.sum(area_weights[labeled_image == i])
            if weighted_area > max_weighted_area:
                max_weighted_area = weighted_area
                l_max = i
    
    res = labeled_image == l_max
    return res


def get_size_largest_connected_component(
    image: NDArray[np.integer[Any]],
    area_weights: NDArray[np.floating[Any]],
    n_components: int,
) -> tuple[int, int, float, int]:
    """Find the largest connected component by pixel count and area.

    Parameters
    ----------
    image : NDArray
        Labeled image where each connected component has a unique integer label.
    area_weights : NDArray
        Array of weights (e.g., grid cell areas) with same shape as image.
    n_components : int
        Total number of connected components in the image.

    Returns
    -------
    tuple[int, int, float, int]
        A tuple containing:

        - Maximum number of pixels in any component.
        - Label of component with most pixels.
        - Maximum weighted area (sum of area_weights) in any component.
        - Label of component with largest weighted area.

    Notes
    -----
    The component with the most pixels may not be the one with the largest
    area, as different grid may contribute different areas.

    Examples
    --------
    >>> max_pixels, label_pixels, max_area, label_area = (
    ...     get_size_largest_connected_component(labeled_image, weights, n_comp)
    ... )
    """
    max_fraction_area = 0
    i_max_fraction_area = -1
    max_size_cluster_pixels = 0
    i_max_size_cluster_pixels = -1
    for i in range(1, n_components + 1):
        n_pixels = np.sum(image == i)
        fraction_area = np.sum(area_weights[image == i])

        # Note: the cluster with most pixels isn't necessarily the one
        # covering the largest area.
        if n_pixels > max_size_cluster_pixels:
            max_size_cluster_pixels = n_pixels
            i_max_size_cluster_pixels = i
        if fraction_area > max_fraction_area:
            max_fraction_area = fraction_area
            i_max_fraction_area = i
    return (
        max_size_cluster_pixels,
        i_max_size_cluster_pixels,
        max_fraction_area,
        i_max_fraction_area,
    )


def get_max_cluster_area_from_binary_image(
    image: NDArray[np.bool_ | np.integer[Any]],
    area_weights: NDArray[np.floating[Any]],
) -> float:
    """Compute the maximum weighted area of connected components in a binary image.

    Parameters
    ----------
    image : NDArray
        Binary image array where True/non-zero values represent foreground.
    area_weights : NDArray
        Array of weights (e.g., grid cell areas) with same shape as image.

    Returns
    -------
    float
        The maximum weighted area (sum of area_weights) among all connected
        components.

    Examples
    --------
    >>> max_area = get_max_cluster_area_from_binary_image(binary_image, weights)
    """
    labeled_image, count = get_image_count_labeled_connected_components(image=image)
    _, _, max_fraction_area, _ = get_size_largest_connected_component(
        labeled_image, area_weights=area_weights, n_components=count
    )
    return max_fraction_area


def stack_to_dim(
    data: xr.Dataset | xr.DataArray, dim: str = "datapoint"
) -> xr.Dataset | xr.DataArray:
    """Stack a Dataset or DataArray into a single dimension if not already stacked.

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray
        The input Dataset or DataArray.
    dim : str, optional
        The name of the new dimension, by default "datapoint".

    Returns
    -------
    xr.Dataset | xr.DataArray
        The stacked Dataset or DataArray.
    """
    if dim in data.dims:
        return data
    return data.stack(datapoint=("ensemble_member", "init_time", "lead_time"))


def extract_datapoints_in_months(
    data: xr.Dataset | xr.DataArray, months: NDArray[np.integer[Any]]
) -> xr.Dataset | xr.DataArray:
    """Extract datapoints from a Dataset or DataArray that fall within specific months.

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray
        The input dataset or data array.
    months : NDArray[np.integer[Any]]
        An array of month numbers (1-12) to extract.

    Returns
    -------
    xr.Dataset | xr.DataArray
        The extracted dataset or data array.
    """
    return data.where(data.valid_time.dt.month.isin(months), drop=True)


def extract_datapoints_in_years(
    data: xr.Dataset | xr.DataArray, year_min: int, year_max: int
) -> xr.Dataset | xr.DataArray:
    """Extract datapoints from Dataset or DataArray that fall within a year range.

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray
        The input dataset or data array.
    year_min : int
        The minimum year to extract (inclusive).
    year_max : int
        The maximum year to extract (inclusive).

    Returns
    -------
    xr.Dataset | xr.DataArray
        The extracted dataset or data array.
    """
    data = data.where(
        np.logical_and(
            data.valid_time.dt.year >= year_min,
            data.valid_time.dt.year <= year_max,
        ),
        drop=True,
    )
    return data


def create_empty_dataset_with_coords(
    ds: xr.Dataset,
    exclude_dims: list[str] | None = None,
    exclude_coords: list[str] | None = None,
) -> xr.Dataset:
    """Create a new empty dataset sharing dimensions and coordinates with original.

    Creates a new empty dataset that shares dimensions and coordinates with the
    original except those specified in exclude_dims and exclude_coords.

    Parameters
    ----------
    ds : xr.Dataset
        The original dataset.
    exclude_dims : list[str] | None, optional
        Dimensions to exclude from the new Dataset.
    exclude_coords : list[str] | None, optional
        Coordinates to exclude from the new Dataset.

    Returns
    -------
    xr.Dataset
        A new empty dataset with the specified dimensions and coordinates.
    """
    # Initialize lists if None
    exclude_dims = exclude_dims or []
    exclude_coords = exclude_coords or []

    # Start with an empty dataset
    new_ds = xr.Dataset()

    # Add dimensions and coordinates from original dataset, excluding those specified
    for dim_name, dim_size in ds.sizes.items():
        if dim_name not in exclude_dims:
            # If this dimension has a coordinate with the same name
            if dim_name in ds.coords:
                if dim_name not in exclude_coords:
                    new_ds[dim_name] = ds[dim_name]
            else:
                # Create dimension without coordinate
                new_ds = new_ds.expand_dims({dim_name: dim_size})

    # Add non-dimension coordinates
    for coord_name, coord_var in ds.coords.items():
        if coord_name not in exclude_coords and coord_name not in new_ds:
            # Check if all the dimensions of this coordinate are in the new dataset
            if all(dim not in exclude_dims for dim in coord_var.dims):
                # Explicitly add as a coordinate to avoid creating a data variable
                new_ds = new_ds.assign_coords({coord_name: coord_var})

    return new_ds


def subsampled_dataarray_to_year_sample(da: xr.DataArray) -> xr.DataArray:
    """Convert a subsampled DataArray to year-sample format.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray with valid_time dimension.

    Returns
    -------
    xr.DataArray
        DataArray reshaped with year and i_day_included dimensions.

    Raises
    ------
    AssertionError
        If not all years have the same number of data points.
    """
    gby = da.valid_time.groupby("valid_time.year")
    # This can fail if February is included
    if not (gby.count() == gby.count()[0]).all():
        raise ValueError("All years must have the same number of data points")

    years = xr.DataArray(list(gby.groups.keys()), dims="year")
    days = xr.DataArray(np.arange(gby.count().max()), dims="i_day_included")

    ds = create_empty_dataset_with_coords(da, exclude_dims="valid_time")
    ds = ds.expand_dims({"year": years, "i_day_included": days})
    shape = tuple(ds.sizes[dim] for dim in ds.dims)
    nan_data = np.full(shape, np.nan)
    ds["var"] = xr.DataArray(data=nan_data, dims=list(ds.dims), coords=ds.coords)
    for year in ds.year:
        ds["var"].loc[dict(year=year)] = (
            da.where(da.valid_time.dt.year == year, drop=True)
            .assign_coords(valid_time=days.data)
            .rename({"valid_time": days.dims[0]})
        )
    return ds["var"]


def select_months_and_resample_to_n_day_means(
    single_year: xr.Dataset | xr.DataArray,
    months: NDArray[np.integer[Any]],
    n_days: int,
    use_strided_resample: bool = True,
) -> xr.Dataset | xr.DataArray:
    """Select months and resample to n-day means.

    Parameters
    ----------
    single_year : xr.Dataset | xr.DataArray
        Input data for a single year.
    months : NDArray[np.integer[Any]]
        Array of month numbers (1-12) to select.
    n_days : int
        Number of days for rolling mean window.
    use_strided_resample : bool, optional
        If True, use strided resampling (every n_days). If False, use all points.

    Returns
    -------
    xr.Dataset | xr.DataArray
        Resampled data with n-day rolling means.
    """
    if use_strided_resample:
        return (
            extract_datapoints_in_months(single_year, months)
            .rolling(valid_time=n_days)
            .mean()
            .isel(valid_time=slice(n_days - 1, None, n_days))
        )
    else:
        return (
            extract_datapoints_in_months(single_year, months)
            .rolling(valid_time=n_days)
            .mean()
            .isel(valid_time=slice(n_days - 1, None, 1))
        )

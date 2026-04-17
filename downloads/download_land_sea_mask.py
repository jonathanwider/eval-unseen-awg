import cdsapi
import yaml

# For downloading ERA5 land-sea mask at given resolution.

with open("../../configs/paths/default_paths.yaml") as file:
    paths = yaml.safe_load(file)["paths"]

c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": ["reanalysis"],
        "variable": ["land_sea_mask"],
        "year": ["1989"],
        "month": ["02"],
        "day": ["11"],
        "time": ["00:00"],
        "data_format": "netcdf",
        "area": [72, -25, 25, 46],
        "grid": [0.4, 0.4],
    },
    paths.path_lsm,
)

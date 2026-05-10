"""
Regrid model data onto a target grid (WRF geo_em or lat/lon bbox).

Uses xESMF (bilinear) when available; falls back to scipy griddata.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def _load_target_grid(config) -> xr.Dataset:
    if config.domain.wrf_geo_file is not None:
        with xr.open_dataset(config.domain.wrf_geo_file) as ds:
            lat_var = "XLAT_M" if "XLAT_M" in ds else "XLAT"
            lon_var = "XLONG_M" if "XLONG_M" in ds else "XLONG"
            lat = ds[lat_var].values.squeeze()
            lon = ds[lon_var].values.squeeze()
        return xr.Dataset({"lat": (("y", "x"), lat), "lon": (("y", "x"), lon)})
    else:
        d = config.domain
        from utils.geo import EARTH_RADIUS_KM
        import math
        dlat = d.dx_km / 111.0
        dlon = d.dx_km / (111.0 * math.cos(math.radians((d.lat_min + d.lat_max) / 2)))
        lat1d = np.arange(d.lat_min, d.lat_max, dlat)
        lon1d = np.arange(d.lon_min, d.lon_max, dlon)
        lon2d, lat2d = np.meshgrid(lon1d, lat1d)
        return xr.Dataset({"lat": (("y", "x"), lat2d), "lon": (("y", "x"), lon2d)})


def _scipy_regrid(da: xr.DataArray, ds_out: xr.Dataset) -> np.ndarray:
    from scipy.interpolate import griddata
    src_lat = da["lat"].values.ravel()
    src_lon = da["lon"].values.ravel()
    src_vals = da.values.ravel()
    mask = np.isfinite(src_vals)
    return griddata(
        (src_lat[mask], src_lon[mask]),
        src_vals[mask],
        (ds_out["lat"].values, ds_out["lon"].values),
        method="linear",
    )


def regrid(ds_model: xr.Dataset, config, variable: str) -> xr.Dataset:
    """
    Regrid ds_model onto the target grid defined by config.domain.
    Returns a new Dataset with dims (time, y, x) and 2-D lat/lon coords.
    """
    weights_path = config.regrid_dir / "weights_bilinear.nc"
    ds_out = _load_target_grid(config)

    use_xesmf = False
    try:
        import xesmf as xe
        use_xesmf = True
    except ImportError:
        logger.warning("xESMF not available — falling back to scipy griddata")

    regridder = None
    time_datasets = []

    for t in ds_model.time.values:
        da = ds_model[variable].sel(time=t)

        if use_xesmf:
            if regridder is None:
                ds_src_xe = xr.Dataset({
                    "lat": da["lat"] if "lat" in da.coords else ds_model["lat"],
                    "lon": da["lon"] if "lon" in da.coords else ds_model["lon"],
                })
                reuse = weights_path.exists()
                regridder = xe.Regridder(
                    ds_src_xe, ds_out, method="bilinear",
                    extrap_method="nearest_s2d",
                    reuse_weights=reuse,
                    weights=str(weights_path) if reuse else None,
                    filename=str(weights_path),
                )
            try:
                vals = regridder(da).values
            except Exception:
                vals = _scipy_regrid(da, ds_out)
        else:
            vals = _scipy_regrid(da, ds_out)

        ny, nx = ds_out["lat"].shape
        time_datasets.append(xr.Dataset(
            {variable: (("time", "y", "x"), vals[np.newaxis])},
            coords={"time": [pd.Timestamp(t)],
                    "lat": ds_out["lat"],
                    "lon": ds_out["lon"]},
        ))

    combined = xr.concat(time_datasets, dim="time").sortby("time")
    combined[variable].attrs.update(ds_model[variable].attrs)
    logger.info(f"Regridded {len(time_datasets)} time steps → {ds_out['lat'].shape}")
    return combined

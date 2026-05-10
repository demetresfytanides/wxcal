"""
Export bias-corrected dataset to NetCDF and/or Zarr.

NetCDF is written with CF-1.8 conventions and zlib compression.
Zarr is chunked along time for efficient ML training reads.
"""
from __future__ import annotations

import logging
from pathlib import Path

import xarray as xr

logger = logging.getLogger(__name__)

_ENCODING_DEFAULTS = {"zlib": True, "complevel": 4, "dtype": "float32"}


def _cf_attrs(ds: xr.Dataset, config) -> xr.Dataset:
    ds = ds.copy()
    ds.attrs.update({
        "Conventions":    "CF-1.8",
        "title":          "wxCal bias-corrected weather model output",
        "source":         str(config.model_source),
        "obs_source":     str(config.obs_source),
        "date_start":     str(config.date_start),
        "date_end":       str(config.date_end),
        "bias_method":    "IDW (Inverse Distance Weighting)",
        "created_by":     "wxCal — Weather Calibration Agent",
    })
    if "lat" in ds.coords:
        ds["lat"].attrs.update({"units": "degrees_north", "long_name": "latitude",
                                 "standard_name": "latitude"})
    if "lon" in ds.coords:
        ds["lon"].attrs.update({"units": "degrees_east", "long_name": "longitude",
                                 "standard_name": "longitude"})
    return ds


def export_netcdf(ds: xr.Dataset, config, tag: str = "corrected") -> Path:
    path = config.output_dir / f"wxcal_{config.model_variable}_{config.date_tag}_{tag}.nc"
    ds = _cf_attrs(ds, config)
    encoding = {v: _ENCODING_DEFAULTS.copy() for v in ds.data_vars}
    ds.to_netcdf(path, encoding=encoding)
    logger.info(f"NetCDF saved → {path}")
    return path


def export_zarr(ds: xr.Dataset, config, tag: str = "corrected") -> Path:
    path = config.output_dir / f"wxcal_{config.model_variable}_{config.date_tag}_{tag}.zarr"
    ds = _cf_attrs(ds, config)
    nt = ds.sizes.get("time", 1)
    ny = ds.sizes.get("y", ds.sizes.get("lat", 1))
    nx = ds.sizes.get("x", ds.sizes.get("lon", 1))
    chunks = {"time": min(24, nt), "y": min(256, ny), "x": min(256, nx)}
    ds.chunk(chunks).to_zarr(str(path), mode="w")
    logger.info(f"Zarr saved → {path}")
    return path


def export(ds: xr.Dataset, config, tag: str = "corrected") -> list[Path]:
    paths = []
    if config.output_format in ("netcdf", "both"):
        paths.append(export_netcdf(ds, config, tag))
    if config.output_format in ("zarr", "both"):
        paths.append(export_zarr(ds, config, tag))
    return paths

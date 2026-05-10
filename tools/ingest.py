"""
Model data ingestion — auto-detects format and returns a normalised xarray Dataset.

Supported sources
-----------------
  "hrrr"        — NOAA public S3 (downloads GRIB2, decodes via cfgrib)
  "era5"        — local ERA5 NetCDF or GRIB file
  Path(...)     — any local NetCDF / GRIB2 / Zarr file (format auto-detected)

Output contract
---------------
All loaders return an xr.Dataset with:
  - variable named after config.model_variable  ("precipitation", "temperature", …)
  - dims: ("time", "lat", "lon") where lat/lon are 1-D or 2-D coordinate arrays
  - time in UTC, dtype datetime64
  - units attribute on the variable
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# Map logical variable names → per-format short names
_HRRR_VAR_MAP   = {"precipitation": "tp",  "temperature": "t2m",
                    "wind_u": "u10", "wind_v": "v10", "pressure": "sp"}
_ERA5_VAR_MAP   = {"precipitation": "tp",  "temperature": "t2m",
                    "wind_u": "u10", "wind_v": "v10", "pressure": "sp"}
_HRRR_UNITS     = {"precipitation": "mm",  "temperature": "K",
                    "wind_u": "m/s", "wind_v": "m/s", "pressure": "Pa"}


# ─────────────────────────────────────────────────────────── HRRR S3 download

def _hrrr_idx_url(date: "date", hour: int) -> tuple[str, str]:
    ymd = date.strftime("%Y%m%d")
    base = (f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
            f"/hrrr.{ymd}/conus/hrrr.t{hour:02d}z.wrfsfcf01.grib2")
    return base, base + ".idx"


def _hrrr_sfc_fields(variable: str) -> list[str]:
    short = _HRRR_VAR_MAP.get(variable, variable)
    field_map = {
        "tp":  [":APCP:surface:"],
        "t2m": [":TMP:2 m above ground:"],
        "u10": [":UGRD:10 m above ground:"],
        "v10": [":VGRD:10 m above ground:"],
        "sp":  [":PRES:surface:"],
    }
    return field_map.get(short, [])


def _download_hrrr_hour(url: str, idx_url: str, fields: list[str],
                         out_path: Path) -> bool:
    import requests
    if out_path.exists():
        return True
    try:
        idx_r = requests.get(idx_url, timeout=30)
        idx_r.raise_for_status()
        byte_ranges = []
        lines = idx_r.text.splitlines()
        for i, line in enumerate(lines):
            if any(f in line for f in fields):
                start = int(line.split(":")[1])
                end = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else ""
                byte_ranges.append((start, end))
        if not byte_ranges:
            return False
        chunks = []
        for start, end in byte_ranges:
            hdr = {"Range": f"bytes={start}-{end}"} if end != "" else {"Range": f"bytes={start}-"}
            r = requests.get(url, headers=hdr, timeout=120)
            r.raise_for_status()
            chunks.append(r.content)
        out_path.write_bytes(b"".join(chunks))
        return True
    except Exception as exc:
        logger.warning(f"HRRR download failed for {url}: {exc}")
        return False


def load_hrrr(config) -> xr.Dataset:
    """Download and decode HRRR data for the configured date range and variable."""
    import concurrent.futures
    from datetime import timedelta

    variable  = config.model_variable
    short_var = _HRRR_VAR_MAP.get(variable, variable)
    fields    = _hrrr_sfc_fields(variable)
    hours     = config.model_hours or list(range(24))
    raw_dir   = config.raw_dir / "hrrr"
    raw_dir.mkdir(parents=True, exist_ok=True)

    from utils.accumulation import hrrr_fetch_start
    tasks = []
    # When obs_utc_cutoff > 0 the accumulation window for date_start begins on
    # date_start - 1, so we must download that day's HRRR as well.
    fetch_start = hrrr_fetch_start(config.date_start,
                                   getattr(config, "obs_utc_cutoff", 0))
    d = fetch_start
    while d <= config.date_end:
        for h in hours:
            url, idx_url = _hrrr_idx_url(d, h)
            out = raw_dir / f"hrrr_{variable}_{d:%Y%m%d}{h:02d}.grib2"
            tasks.append((url, idx_url, fields, out, d, h))
        from datetime import timedelta
        d = d + timedelta(days=1)

    logger.info(f"Downloading {len(tasks)} HRRR files (variable={variable})…")
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_download_workers) as ex:
        futures = {ex.submit(_download_hrrr_hour, *t[:4]): t for t in tasks}
        for f in concurrent.futures.as_completed(futures):
            t = futures[f]
            if not f.result():
                logger.warning(f"Skipped {t[4]} {t[5]:02d}z")

    grib_files = sorted(raw_dir.glob(f"hrrr_{variable}_*.grib2"))
    if not grib_files:
        raise RuntimeError("No HRRR files downloaded")

    datasets = []
    for grib_path in grib_files:
        try:
            ds = _read_hrrr_grib2(grib_path, short_var)
            if ds is not None:
                datasets.append(ds)
        except Exception as exc:
            logger.warning(f"Could not read {grib_path.name}: {exc}")

    if not datasets:
        raise RuntimeError("No HRRR files could be decoded")

    combined = xr.concat(datasets, dim="time").sortby("time")
    combined[variable].attrs["units"] = _HRRR_UNITS.get(variable, "unknown")
    combined[variable].attrs["source"] = "HRRR"
    return combined


def _read_hrrr_grib2(grib_path: Path, short_var: str) -> xr.Dataset | None:
    filter_groups = [
        {"typeOfLevel": "surface"},
        {"typeOfLevel": "heightAboveGround", "level": 2},
        {"typeOfLevel": "heightAboveGround", "level": 10},
    ]
    for filters in filter_groups:
        try:
            ds = xr.open_dataset(str(grib_path), engine="cfgrib",
                                  filter_by_keys=filters)
        except Exception:
            continue
        if short_var not in ds:
            continue

        da = ds[short_var]
        rename = {k: v for k, v in [("latitude", "lat"), ("longitude", "lon")] if k in da.coords}
        if rename:
            da = da.rename(rename)
        lon = da["lon"].values.copy()
        da = da.assign_coords(lon=(da["lon"].dims, np.where(lon > 180, lon - 360, lon)))
        drop = [c for c in ("surface", "heightAboveGround", "step", "time", "valid_time")
                if c in da.coords and da.coords[c].ndim == 0]
        if drop:
            da = da.drop_vars(drop)

        stem = grib_path.stem
        try:
            ts = stem.split("_")[-1]
            valid_time = pd.Timestamp(year=int(ts[:4]), month=int(ts[4:6]),
                                      day=int(ts[6:8]), hour=int(ts[8:10]))
        except Exception:
            return None

        # Rename variable to logical name
        var_name = grib_path.stem.split("_")[1]  # e.g. "precipitation"
        return xr.Dataset(
            {var_name: da.expand_dims("time").assign_coords(time=[valid_time])})

    return None


# ──────────────────────────────────────────────────────── generic local file

def load_local(path: Path, variable: str) -> xr.Dataset:
    """Load a local NetCDF, GRIB2, or Zarr file."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".nc", ".nc4", ".netcdf"):
        ds = xr.open_dataset(path)
    elif suffix in (".grib", ".grib2", ".grb", ".grb2"):
        ds = xr.open_dataset(str(path), engine="cfgrib")
    elif path.is_dir() or suffix == ".zarr":
        ds = xr.open_zarr(str(path))
    else:
        raise ValueError(f"Unrecognised file format: {path}")

    # Try to find a matching variable
    candidates = [v for v in ds.data_vars
                  if variable.lower() in v.lower()
                  or _ERA5_VAR_MAP.get(variable, "").lower() in v.lower()]
    if not candidates:
        raise ValueError(
            f"Variable '{variable}' not found in {path}. "
            f"Available: {list(ds.data_vars)}"
        )
    var = candidates[0]
    if var != variable:
        ds = ds.rename({var: variable})

    # Normalise coordinate names
    rename = {}
    for alt, canonical in [("latitude", "lat"), ("longitude", "lon"),
                            ("valid_time", "time")]:
        if alt in ds.coords:
            rename[alt] = canonical
    if rename:
        ds = ds.rename(rename)

    # Normalise longitudes
    if "lon" in ds.coords:
        lon = ds["lon"].values.copy()
        if lon.max() > 180:
            ds = ds.assign_coords(lon=np.where(lon > 180, lon - 360, lon))

    return ds


# ─────────────────────────────────────────────────────────────── dispatcher

def load_model_data(config) -> xr.Dataset:
    """Route to the correct loader based on config.model_source."""
    src = config.model_source
    if src == "hrrr":
        return load_hrrr(config)
    elif src == "era5" or (isinstance(src, Path) and src.exists()):
        path = src if isinstance(src, Path) else Path(src)
        return load_local(path, config.model_variable)
    else:
        raise ValueError(f"Unknown model_source: {src}")

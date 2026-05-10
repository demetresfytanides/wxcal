"""
Observation loading — returns a normalised pandas DataFrame.

Sources
-------
  "acis"    — NOAA ACIS multi-network API (auto-bbox from domain)
  Path(...) — local CSV with columns: station_id, lat, lon, date, value

Output contract
---------------
DataFrame columns: station_id, lat, lon, date (datetime.date), value (float)
Units: same as the model variable (mm for precipitation, K for temperature, …)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from utils.geo import domain_bbox_from_wrf

logger = logging.getLogger(__name__)

ACIS_URL = "http://data.rcc-acis.org/MultiStnData"

# ACIS element codes per variable
_ACIS_ELEMENTS = {
    "precipitation": [{"name": "pcpn", "interval": "dly", "units": "in"}],
    "temperature":   [{"name": "avgt", "interval": "dly", "units": "degF"}],
}

_UNIT_CONVERT = {
    "precipitation": lambda v: v * 25.4,      # inches → mm
    "temperature":   lambda v: (v - 32) * 5 / 9 + 273.15,  # degF → K
}


def _domain_bbox(config) -> dict[str, float]:
    if config.domain.wrf_geo_file is not None:
        bb = domain_bbox_from_wrf(config.domain.wrf_geo_file)
    else:
        bb = {"south": config.domain.lat_min, "north": config.domain.lat_max,
              "west":  config.domain.lon_min, "east":  config.domain.lon_max}
    # add a small pad so edge stations are captured
    pad = 0.5
    return {k: v + (pad if k in ("north", "east") else -pad) for k, v in bb.items()}


def load_acis(config) -> pd.DataFrame:
    variable = config.model_variable
    elements = _ACIS_ELEMENTS.get(variable)
    if elements is None:
        raise ValueError(f"No ACIS element mapping for variable '{variable}'")

    convert = _UNIT_CONVERT.get(variable, lambda v: v)
    bbox = _domain_bbox(config)

    payload = {
        "bbox":    f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}",
        "sdate":   config.date_start.strftime("%Y-%m-%d"),
        "edate":   config.date_end.strftime("%Y-%m-%d"),
        "elems":   elements,
        "meta":    "name,ll",
    }
    logger.info(f"Fetching ACIS observations for {config.date_start} → {config.date_end}…")
    r = requests.post(ACIS_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()

    rows = []
    for stn in data.get("data", []):
        try:
            lon, lat = stn["meta"]["ll"]
            sid = stn["meta"].get("name", str(lon) + "_" + str(lat))
        except (KeyError, TypeError):
            continue

        for i, day_val in enumerate(stn.get("data", [])):
            date = pd.Timestamp(config.date_start) + pd.Timedelta(days=i)
            val = day_val[0] if isinstance(day_val, list) else day_val
            if val in ("M", "", None, "S"):
                val = np.nan
            elif val == "T":
                val = 0.0
            else:
                try:
                    val = float(val)
                    val = convert(val)
                except (ValueError, TypeError):
                    val = np.nan
            rows.append({"station_id": sid, "lat": lat, "lon": lon,
                         "date": date.date(), "value": val})

    df = pd.DataFrame(rows)
    logger.info(f"  {len(df):,} obs-days from {df['station_id'].nunique():,} stations")
    return df


def load_csv_obs(path: Path, variable: str) -> pd.DataFrame:
    """Load a user-supplied CSV. Must have: station_id, lat, lon, date, value."""
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    required = {"station_id", "lat", "lon", "date", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    logger.info(f"Loaded {len(df):,} rows from {path}")
    return df


def load_observations(config) -> pd.DataFrame:
    src = config.obs_source
    if src == "acis":
        return load_acis(config)
    path = Path(src)
    if path.exists() and path.suffix.lower() == ".csv":
        return load_csv_obs(path, config.model_variable)
    raise ValueError(f"Unknown obs_source: {src}")

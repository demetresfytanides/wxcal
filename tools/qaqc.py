"""
QA/QC tool functions.

These are called BY the QA/QC agent — the LLM decides which to call and with
what thresholds based on data summaries it inspects first.

All functions return (modified_data, log_entry) so the agent can accumulate
a transparent decision log.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


# ──────────────────────────────────────────── inspection (read-only, for LLM)

def summarise_grid(ds: xr.Dataset, variable: str) -> dict[str, Any]:
    """Return distribution statistics the agent uses to decide QA/QC strategy."""
    vals = ds[variable].values
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        return {"n": 0, "warning": "all values NaN or missing"}
    pcts = np.percentile(finite, [1, 5, 25, 50, 75, 95, 99])
    return {
        "n_total":        int(vals.size),
        "n_finite":       int(len(finite)),
        "n_nan":          int(np.isnan(vals).sum()),
        "n_negative":     int((finite < 0).sum()),
        "mean":           float(np.mean(finite)),
        "std":            float(np.std(finite)),
        "min":            float(np.min(finite)),
        "p01":            float(pcts[0]),
        "p05":            float(pcts[1]),
        "p25":            float(pcts[2]),
        "p50":            float(pcts[3]),
        "p75":            float(pcts[4]),
        "p95":            float(pcts[5]),
        "p99":            float(pcts[6]),
        "max":            float(np.max(finite)),
        "zero_fraction":  float((finite == 0).mean()),
        "units":          ds[variable].attrs.get("units", "unknown"),
    }


def summarise_obs(df: pd.DataFrame) -> dict[str, Any]:
    """Return observation distribution statistics."""
    vals = df["value"].dropna()
    if len(vals) == 0:
        return {"n": 0, "warning": "no valid observations"}
    pcts = np.percentile(vals, [1, 5, 25, 50, 75, 95, 99])
    return {
        "n_stations":     int(df["station_id"].nunique()),
        "n_records":      int(len(df)),
        "n_valid":        int(len(vals)),
        "n_missing":      int(df["value"].isna().sum()),
        "n_negative":     int((vals < 0).sum()),
        "mean":           float(vals.mean()),
        "std":            float(vals.std()),
        "min":            float(vals.min()),
        "p01":            float(pcts[0]),
        "p05":            float(pcts[1]),
        "p25":            float(pcts[2]),
        "p50":            float(pcts[3]),
        "p75":            float(pcts[4]),
        "p95":            float(pcts[5]),
        "p99":            float(pcts[6]),
        "max":            float(vals.max()),
        "dates_covered":  int(df["date"].nunique()),
    }


# ──────────────────────────────────────────── model grid QA/QC actions

def clip_negative_grid(ds: xr.Dataset, variable: str) -> tuple[xr.Dataset, dict]:
    before = int((ds[variable].values < 0).sum())
    ds = ds.copy()
    ds[variable] = ds[variable].clip(min=0)
    return ds, {"action": "clip_negative_grid", "n_clipped": before,
                "variable": variable}


def flag_extreme_grid(ds: xr.Dataset, variable: str,
                       max_value: float) -> tuple[xr.Dataset, dict]:
    vals = ds[variable].values.copy()
    mask = vals > max_value
    n_flagged = int(mask.sum())
    vals[mask] = np.nan
    ds = ds.copy()
    ds[variable] = xr.DataArray(vals, dims=ds[variable].dims,
                                  coords=ds[variable].coords,
                                  attrs=ds[variable].attrs)
    return ds, {"action": "flag_extreme_grid", "threshold": max_value,
                "n_flagged": n_flagged, "variable": variable}


def zero_trace_grid(ds: xr.Dataset, variable: str,
                     min_threshold: float) -> tuple[xr.Dataset, dict]:
    vals = ds[variable].values.copy()
    mask = (vals > 0) & (vals < min_threshold)
    n_zeroed = int(mask.sum())
    vals[mask] = 0.0
    ds = ds.copy()
    ds[variable] = xr.DataArray(vals, dims=ds[variable].dims,
                                  coords=ds[variable].coords,
                                  attrs=ds[variable].attrs)
    return ds, {"action": "zero_trace_grid", "threshold": min_threshold,
                "n_zeroed": n_zeroed, "variable": variable}


def no_change_grid(ds: xr.Dataset, reason: str) -> tuple[xr.Dataset, dict]:
    return ds, {"action": "no_change_grid", "reason": reason}


# ──────────────────────────────────────────── observation QA/QC actions

def clip_negative_obs(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n = int((df["value"] < 0).sum())
    df = df.copy()
    df.loc[df["value"] < 0, "value"] = 0.0
    return df, {"action": "clip_negative_obs", "n_clipped": n}


def flag_extreme_obs(df: pd.DataFrame, max_value: float) -> tuple[pd.DataFrame, dict]:
    mask = df["value"] > max_value
    n = int(mask.sum())
    df = df.copy()
    df.loc[mask, "value"] = np.nan
    return df, {"action": "flag_extreme_obs", "threshold": max_value, "n_flagged": n}


def drop_sparse_stations(df: pd.DataFrame,
                          min_days: int) -> tuple[pd.DataFrame, dict]:
    counts = df.groupby("station_id")["date"].nunique()
    keep = counts[counts >= min_days].index
    before = df["station_id"].nunique()
    df = df[df["station_id"].isin(keep)].copy()
    return df, {"action": "drop_sparse_stations", "min_days": min_days,
                "removed": int(before - df["station_id"].nunique())}


def no_change_obs(df: pd.DataFrame, reason: str) -> tuple[pd.DataFrame, dict]:
    return df, {"action": "no_change_obs", "reason": reason}


# ──────────────────────────────────────────── log serialisation

def format_log(entries: list[dict]) -> str:
    return json.dumps(entries, indent=2, default=str)

"""
Daily accumulation window helpers.

CoCoRaHS/ACIS observers report at ~7 am local time (≈ 12:00 UTC for Central,
11:00 UTC for Eastern in summer).  The label "2024-06-01" therefore represents
the 24-hour period that ENDS at obs_utc_cutoff on 2024-06-01, i.e.:

    window: [2024-05-31 12:00 UTC, 2024-06-01 12:00 UTC)

When obs_utc_cutoff = 0 the function falls back to a plain calendar-day sum
(backward-compatible behaviour).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import xarray as xr


def day_accumulation(ds: xr.Dataset, variable: str,
                     day: date, obs_utc_cutoff: int) -> np.ndarray:
    """
    Return the 2-D (ny, nx) daily accumulation for `day` using the
    observation accumulation window defined by obs_utc_cutoff.

    Parameters
    ----------
    ds              : Dataset with a 'time' dimension in UTC.
    variable        : Variable name to sum.
    day             : The reported observation date.
    obs_utc_cutoff  : UTC hour at which the 24-h obs accumulation ends.
                      12 = CoCoRaHS standard (7 am CDT).
                      0  = plain calendar day (00:00–23:00 UTC), backward compat.
    """
    if obs_utc_cutoff == 0:
        mask = ds["time"].dt.date == day
    else:
        t_end   = pd.Timestamp(day) + pd.Timedelta(hours=obs_utc_cutoff)
        t_start = t_end - pd.Timedelta(hours=24)
        mask    = (ds["time"] >= t_start) & (ds["time"] < t_end)

    selected = ds[variable].sel(time=mask)
    if selected.sizes["time"] == 0:
        # No data in window — return zeros with correct spatial shape
        spatial_dims = [d for d in ds[variable].dims if d != "time"]
        shape = tuple(ds.sizes[d] for d in spatial_dims)
        return np.zeros(shape, dtype=np.float32)

    return selected.sum("time").values


def hrrr_fetch_start(date_start: date, obs_utc_cutoff: int) -> date:
    """
    Return the earliest date that must be downloaded so that the first
    obs accumulation window is fully covered.

    For obs_utc_cutoff > 0 the window for date_start begins on date_start - 1,
    so we need HRRR data from that day.
    """
    if obs_utc_cutoff > 0:
        return date_start - timedelta(days=1)
    return date_start

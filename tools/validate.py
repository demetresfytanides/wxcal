"""
Validation metrics — called by the correction agent to decide whether to
accept a correction or retry with different parameters.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from utils.accumulation import day_accumulation


def _grid_to_stns(data_2d, grid_lat, grid_lon, stn_lat, stn_lon):
    """Nearest-neighbour interpolation via cKDTree — much faster than griddata
    for large grids (avoids O(n log n) Delaunay triangulation of 1M points)."""
    from scipy.spatial import cKDTree
    from utils.geo import latlon_to_xyz

    tree = cKDTree(latlon_to_xyz(grid_lat.ravel(), grid_lon.ravel()))
    _, idx = tree.query(latlon_to_xyz(stn_lat, stn_lon), k=1, workers=-1)
    return data_2d.ravel()[idx].astype(float)


def compute_metrics(ds_before: xr.Dataset, ds_after: xr.Dataset,
                     df_obs: pd.DataFrame, variable: str,
                     obs_utc_cutoff: int = 0) -> dict[str, Any]:
    """
    Compute cross-validated error metrics at station locations
    for both pre- and post-correction fields.

    Returns a dict the correction agent uses to decide accept/retry.
    """
    grid_lat = ds_before["lat"].values
    grid_lon = ds_before["lon"].values

    df_obs = df_obs.copy()
    df_obs["date"] = pd.to_datetime(df_obs["date"]).dt.date

    all_obs, all_pre, all_post = [], [], []

    days = sorted({pd.Timestamp(t).date() for t in ds_before.time.values})
    for day in days:
        pre_daily  = day_accumulation(ds_before, variable, day, obs_utc_cutoff)
        post_daily = day_accumulation(ds_after,  variable, day, obs_utc_cutoff)

        obs_day = df_obs[df_obs["date"] == day].dropna(subset=["value"])
        if obs_day.empty:
            continue

        stn_lat = obs_day["lat"].values
        stn_lon = obs_day["lon"].values
        obs_mm  = obs_day["value"].values

        pre_at  = _grid_to_stns(pre_daily,  grid_lat, grid_lon, stn_lat, stn_lon)
        post_at = _grid_to_stns(post_daily, grid_lat, grid_lon, stn_lat, stn_lon)
        mask    = np.isfinite(obs_mm) & np.isfinite(pre_at) & np.isfinite(post_at)
        all_obs.append(obs_mm[mask]);  all_pre.append(pre_at[mask])
        all_post.append(post_at[mask])

    if not all_obs:
        return {"error": "no overlapping station/grid pairs found"}

    obs  = np.concatenate(all_obs)
    pre  = np.concatenate(all_pre)
    post = np.concatenate(all_post)

    def _stats(model):
        mask = np.isfinite(obs) & np.isfinite(model)
        if mask.sum() < 2:
            return {}
        o, m = obs[mask], model[mask]

        bias = float(np.mean(m - o))
        mae  = float(np.mean(np.abs(m - o)))
        rmse = float(np.sqrt(np.mean((m - o) ** 2)))
        corr = float(np.corrcoef(o, m)[0, 1]) if o.std() > 0 and m.std() > 0 else float("nan")

        # Kling-Gupta Efficiency: combines correlation, variability ratio, bias ratio
        if o.std() > 0 and m.std() > 0 and o.mean() > 0:
            r     = np.corrcoef(o, m)[0, 1]
            alpha = float(m.std()  / o.std())
            beta  = float(m.mean() / o.mean())
            kge   = float(1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))
        else:
            kge = float("nan")

        # Categorical wet/dry metrics (threshold 0.1 mm)
        thresh  = 0.1
        obs_wet = o > thresh
        mdl_wet = m > thresh
        hits    = int(np.sum( obs_wet &  mdl_wet))
        misses  = int(np.sum( obs_wet & ~mdl_wet))
        fa      = int(np.sum(~obs_wet &  mdl_wet))
        pod        = float(hits / (hits + misses)) if (hits + misses) > 0 else float("nan")
        far        = float(fa   / (hits + fa))     if (hits + fa)     > 0 else float("nan")
        freq_bias  = float((hits + fa) / (hits + misses)) if (hits + misses) > 0 else float("nan")

        # Upper-tail ratio
        o95 = float(np.percentile(o, 95))
        m95 = float(np.percentile(m, 95))
        p95_ratio = float(m95 / o95) if o95 > 0 else float("nan")

        return {
            "bias": bias, "mae": mae, "rmse": rmse, "corr": corr,
            "kge": kge,
            "pod": pod, "far": far, "freq_bias": freq_bias,
            "p95_ratio": p95_ratio,
            "n": int(mask.sum()),
        }

    s_pre  = _stats(pre)
    s_post = _stats(post)

    improvement = {}
    for k in ("bias", "rmse"):
        if k in s_pre and k in s_post:
            improvement[f"{k}_improvement_pct"] = float(
                (abs(s_pre[k]) - abs(s_post[k])) / (abs(s_pre[k]) + 1e-9) * 100
            )
    if "corr" in s_pre and "corr" in s_post:
        improvement["corr_delta"] = float(s_post["corr"] - s_pre["corr"])

    return {
        "pre_correction":  s_pre,
        "post_correction": s_post,
        "improvement":     improvement,
        "n_pairs":         int(len(obs)),
        "accept_recommendation": (
            improvement.get("rmse_improvement_pct", 0) > -5  # allow up to 5% degradation
        ),
    }

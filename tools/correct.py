"""
Vectorised IDW bias correction engine.

Called by the correction agent with LLM-chosen parameters.
Returns corrected dataset + validation metrics so the agent can
decide whether to accept or retry.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from utils.accumulation import day_accumulation
from utils.geo import EARTH_RADIUS_KM, km_to_chord, latlon_to_xyz

logger = logging.getLogger(__name__)

_CHUNK_ROWS = 64


def _interp_to_stations(grid_data: np.ndarray, grid_lat: np.ndarray,
                          grid_lon: np.ndarray, stn_lat: np.ndarray,
                          stn_lon: np.ndarray) -> np.ndarray:
    tree = cKDTree(latlon_to_xyz(grid_lat.ravel(), grid_lon.ravel()))
    _, idx = tree.query(latlon_to_xyz(stn_lat, stn_lon), k=1, workers=-1)
    return grid_data.ravel()[idx].astype(float)


def _build_ratio_field(
    model_daily: np.ndarray, model_at_stns: np.ndarray,
    obs_vals: np.ndarray, stn_lat: np.ndarray, stn_lon: np.ndarray,
    grid_lat: np.ndarray, grid_lon: np.ndarray,
    idw_power: float, max_dist_km: float,
    min_stations: int, max_ratio: float, min_thresh: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised chunk IDW — returns (ratio_field, additive_field), shape (ny, nx)."""
    ny, nx = model_daily.shape
    ratio_flat    = np.ones(ny * nx, dtype=np.float32)
    additive_flat = np.zeros(ny * nx, dtype=np.float32)

    valid = np.isfinite(obs_vals) & np.isfinite(model_at_stns) & (obs_vals >= 0)
    if not valid.any():
        return ratio_flat.reshape(ny, nx), additive_flat.reshape(ny, nx)

    obs_v = obs_vals[valid];  mdl_v = model_at_stns[valid]
    s_lat = stn_lat[valid];   s_lon = stn_lon[valid]
    n = len(s_lat)

    model_pos = mdl_v > min_thresh
    ratios = np.where(model_pos, obs_v / np.where(model_pos, mdl_v, 1.0), np.nan)
    ratios = np.clip(ratios, 0.0, max_ratio)
    additive_stn = np.where(~model_pos & (obs_v > min_thresh), obs_v / 24.0, 0.0)

    tree      = cKDTree(latlon_to_xyz(s_lat, s_lon))
    max_chord = km_to_chord(max_dist_km)
    k         = min(n, 100)

    g_lat = grid_lat.ravel();  g_lon = grid_lon.ravel()

    for r0 in range(0, ny, _CHUNK_ROWS):
        r1  = min(r0 + _CHUNK_ROWS, ny)
        sl  = slice(r0 * nx, r1 * nx)
        xyz = latlon_to_xyz(g_lat[sl], g_lon[sl])

        cd, ni = tree.query(xyz, k=k, distance_upper_bound=max_chord, workers=-1)
        if k == 1:
            cd = cd[:, np.newaxis];  ni = ni[:, np.newaxis]

        in_r = np.isfinite(cd)
        d_km = np.where(in_r,
                        2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(cd * 0.5, 0, 1)),
                        1.0)
        d_km = np.maximum(d_km, 1e-3)
        w = np.where(in_r, 1.0 / d_km ** idw_power, 0.0)
        si = np.where(in_r, ni, 0)

        # multiplicative
        rn = ratios[si]
        mv = in_r & np.isfinite(rn)
        mw = np.where(mv, w, 0.0);  mws = mw.sum(1);  nm = mv.sum(1)
        has_m = (mws > 0) & (nm >= min_stations)
        rf = ratio_flat[sl]
        rf[has_m] = ((mw * np.where(mv, rn, 0.0)).sum(1)[has_m] / mws[has_m]).astype(np.float32)

        # additive
        an = additive_stn[si]
        av = in_r & (an > 0)
        aw = np.where(av, w, 0.0);  aws = aw.sum(1)
        has_a = aws > 0
        af = additive_flat[sl]
        af[has_a] = ((aw * an).sum(1)[has_a] / aws[has_a]).astype(np.float32)

    return ratio_flat.reshape(ny, nx), additive_flat.reshape(ny, nx)


def _window_mask(times: pd.DatetimeIndex, day, obs_utc_cutoff: int) -> np.ndarray:
    """Boolean mask for the hours belonging to this observation date's accumulation window."""
    if obs_utc_cutoff == 0:
        return (times.date == day)
    t_end   = pd.Timestamp(day) + pd.Timedelta(hours=obs_utc_cutoff)
    t_start = t_end - pd.Timedelta(hours=24)
    return ((times >= t_start) & (times < t_end)).values


def apply_correction(
    ds_model: xr.Dataset,
    df_obs: pd.DataFrame,
    variable: str,
    idw_power: float,
    radius_km: float,
    min_stations: int = 3,
    max_ratio: float = 10.0,
    min_thresh: float = 0.1,
    obs_utc_cutoff: int = 0,
) -> tuple[xr.Dataset, list[dict]]:
    """
    Apply IDW bias correction window-by-window.

    Each observation date's accumulation window [D-1 cutoff, D cutoff) is
    corrected using only the model hours in that window, so ratios and the
    corrected hours are always consistent. This avoids the bug where ratios
    computed from a 12:00 UTC window get applied to calendar-day hours that
    fall outside that window (e.g. afternoon convective hours on June 1 that
    were not part of the May31→June1 accumulation but still got scaled).
    """
    grid_lat = ds_model["lat"].values
    grid_lon = ds_model["lon"].values
    df_obs   = df_obs.copy()
    df_obs["date"] = pd.to_datetime(df_obs["date"]).dt.date
    times    = pd.DatetimeIndex(ds_model["time"].values)

    # Work on a mutable copy of the full data array
    corrected_data = ds_model[variable].values.copy()   # (n_time, ny, nx)
    reports = []

    obs_dates = sorted(df_obs["date"].unique())
    for day in obs_dates:
        mask = _window_mask(times, day, obs_utc_cutoff)
        if not mask.any():
            continue

        window_data = corrected_data[mask]              # (n_window_hours, ny, nx)
        window_sum  = window_data.sum(axis=0)           # (ny, nx) daily accumulation

        obs_day = df_obs[df_obs["date"] == day].dropna(subset=["value"])
        rep     = {"date": str(day), "n_stations": len(obs_day),
                   "idw_power": idw_power, "radius_km": radius_km}

        if obs_day.empty or len(obs_day) < min_stations:
            rep["correction"] = "skipped — insufficient observations"
            reports.append(rep)
            continue

        stn_lat = obs_day["lat"].values
        stn_lon = obs_day["lon"].values
        obs_mm  = obs_day["value"].values

        mdl_at_stns = _interp_to_stations(window_sum, grid_lat, grid_lon, stn_lat, stn_lon)
        ratio, additive = _build_ratio_field(
            window_sum, mdl_at_stns, obs_mm, stn_lat, stn_lon,
            grid_lat, grid_lon, idw_power, radius_km,
            min_stations, max_ratio, min_thresh,
        )

        # Apply to exactly the window hours — no leakage to other hours
        zero_mask = window_data < min_thresh
        corrected_window = np.where(zero_mask,
                                    window_data + additive[np.newaxis],
                                    window_data * ratio[np.newaxis])
        corrected_window = np.clip(corrected_window, 0.0, None)
        corrected_data[mask] = corrected_window

        rep.update({
            "model_daily_mean_before": float(np.nanmean(window_sum)),
            "model_daily_mean_after":  float(np.nanmean(corrected_window.sum(axis=0))),
            "obs_daily_mean":          float(np.nanmean(obs_mm)),
            "ratio_mean":              float(np.nanmean(ratio)),
            "correction":              "applied",
        })
        reports.append(rep)
        logger.info(
            f"{day}: {len(obs_day)} stn, "
            f"mean {rep['model_daily_mean_before']:.2f} → "
            f"{rep['model_daily_mean_after']:.2f}"
        )

    ds_out = ds_model.copy()
    ds_out[variable] = xr.DataArray(
        corrected_data, dims=ds_model[variable].dims,
        coords=ds_model[variable].coords,
        attrs={**ds_model[variable].attrs, "bias_corrected": "IDW"})
    return ds_out, reports


def apply_quantile_mapping(
    ds_model: xr.Dataset,
    df_obs: pd.DataFrame,
    variable: str,
    obs_utc_cutoff: int = 0,
    n_quantiles: int = 100,
) -> tuple[xr.Dataset, list[dict]]:
    """
    Global quantile mapping: maps model distribution to observed distribution
    using station data, applied uniformly to all grid points.
    More robust than IDW for convective precipitation because it corrects
    the distribution shape rather than a spatially-varying ratio field.
    """
    grid_lat = ds_model["lat"].values
    grid_lon = ds_model["lon"].values
    df_obs = df_obs.copy()
    df_obs["date"] = pd.to_datetime(df_obs["date"]).dt.date

    tree = cKDTree(latlon_to_xyz(grid_lat.ravel(), grid_lon.ravel()))
    all_model, all_obs = [], []

    days = sorted({pd.Timestamp(t).date() for t in ds_model.time.values})
    for day in days:
        model_day = day_accumulation(ds_model, variable, day, obs_utc_cutoff)
        obs_day   = df_obs[df_obs["date"] == day].dropna(subset=["value"])
        if obs_day.empty:
            continue
        _, idx = tree.query(
            latlon_to_xyz(obs_day["lat"].values, obs_day["lon"].values),
            k=1, workers=-1)
        all_model.append(model_day.ravel()[idx].astype(float))
        all_obs.append(obs_day["value"].values.astype(float))

    if not all_model:
        return ds_model, [{"method": "quantile_mapping", "status": "skipped — no paired data"}]

    model_vals = np.concatenate(all_model)
    obs_vals   = np.concatenate(all_obs)
    mask = (np.isfinite(model_vals) & np.isfinite(obs_vals)
            & (obs_vals >= 0) & (model_vals >= 0))
    model_vals = model_vals[mask]
    obs_vals   = obs_vals[mask]

    if len(model_vals) < 10:
        return ds_model, [{"method": "quantile_mapping", "status": "skipped — too few pairs"}]

    quantiles = np.linspace(0, 100, n_quantiles + 1)
    model_q   = np.percentile(model_vals, quantiles)
    obs_q     = np.percentile(obs_vals,   quantiles)

    # Apply transfer function only within each observation window (same window
    # consistency fix as IDW — avoids applying a ratio calibrated for one 24h
    # window to hours outside that window).
    times          = pd.DatetimeIndex(ds_model["time"].values)
    corrected_data = ds_model[variable].values.copy()
    obs_dates      = sorted(df_obs["date"].unique())
    for day in obs_dates:
        mask = _window_mask(times, day, obs_utc_cutoff)
        if not mask.any():
            continue
        corrected_data[mask] = np.interp(
            corrected_data[mask], model_q, obs_q).clip(0).astype(np.float32)

    ds_out = ds_model.copy()
    ds_out[variable] = xr.DataArray(
        corrected_data, dims=ds_model[variable].dims,
        coords=ds_model[variable].coords,
        attrs={**ds_model[variable].attrs, "bias_corrected": "quantile_mapping"})

    return ds_out, [{
        "method":       "quantile_mapping",
        "n_paired":     int(mask.sum()),
        "n_quantiles":  n_quantiles,
        "model_median": float(np.median(model_vals)),
        "obs_median":   float(np.median(obs_vals)),
        "model_p95":    float(np.percentile(model_vals, 95)),
        "obs_p95":      float(np.percentile(obs_vals,   95)),
        "status":       "applied",
    }]


def diagnose_regime(
    ds_model: xr.Dataset,
    df_obs: pd.DataFrame,
    variable: str,
    obs_utc_cutoff: int = 0,
) -> dict:
    """
    Classify precipitation regime as convective or stratiform.

    Convective indicators:
      - High spatial coefficient of variation (CV > 2.0): precipitation is
        concentrated in small cells, not spread uniformly across the domain.
      - Low obs-model spatial correlation (r < 0.3): the model places storms
        in different locations than observed (phase error), so a spatially-
        varying ratio field like IDW would amplify rather than reduce errors.

    Stratiform indicators: low CV, high spatial correlation — IDW works well.
    """
    grid_lat = ds_model["lat"].values
    grid_lon = ds_model["lon"].values
    df_obs   = df_obs.copy()
    df_obs["date"] = pd.to_datetime(df_obs["date"]).dt.date

    tree = cKDTree(latlon_to_xyz(grid_lat.ravel(), grid_lon.ravel()))
    days = sorted({pd.Timestamp(t).date() for t in ds_model.time.values})

    cv_values, corr_values, wet_fracs = [], [], []

    for day in days:
        model_day = day_accumulation(ds_model, variable, day, obs_utc_cutoff)
        obs_day   = df_obs[df_obs["date"] == day].dropna(subset=["value"])

        flat = model_day.ravel()
        flat = flat[np.isfinite(flat) & (flat >= 0)]
        mean = flat.mean()
        cv_values.append(float(flat.std() / mean) if mean > 0.1 else 0.0)
        wet_fracs.append(float((flat > 0.1).mean()))

        if len(obs_day) >= 5:
            _, idx = tree.query(
                latlon_to_xyz(obs_day["lat"].values, obs_day["lon"].values),
                k=1, workers=-1)
            mdl_at_stns = model_day.ravel()[idx].astype(float)
            obs_vals    = obs_day["value"].values.astype(float)
            valid = np.isfinite(mdl_at_stns) & np.isfinite(obs_vals)
            if (valid.sum() >= 5
                    and obs_vals[valid].std() > 0
                    and mdl_at_stns[valid].std() > 0):
                corr_values.append(
                    float(np.corrcoef(obs_vals[valid], mdl_at_stns[valid])[0, 1]))

    mean_cv   = float(np.mean(cv_values))   if cv_values   else 0.0
    mean_corr = float(np.mean(corr_values)) if corr_values else float("nan")
    mean_wet  = float(np.mean(wet_fracs))   if wet_fracs   else 0.0

    is_convective = mean_cv > 2.0 or (np.isfinite(mean_corr) and mean_corr < 0.3)

    return {
        "regime":            "convective" if is_convective else "stratiform",
        "mean_cv":           round(mean_cv, 3),
        "mean_spatial_corr": round(mean_corr, 3) if np.isfinite(mean_corr) else None,
        "mean_wet_fraction": round(mean_wet, 3),
        "n_days":            len(days),
        "recommendation": (
            "Use quantile mapping first — IDW is unreliable for convective "
            "precipitation due to high spatial variability and/or low obs-model "
            "spatial correlation (phase errors)."
            if is_convective else
            "IDW is appropriate — spatially coherent bias field detected."
        ),
    }

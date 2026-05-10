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
    Apply IDW bias correction day-by-day.
    Returns (corrected_dataset, per_day_reports).
    """
    grid_lat = ds_model["lat"].values
    grid_lon = ds_model["lon"].values
    df_obs = df_obs.copy()
    df_obs["date"] = pd.to_datetime(df_obs["date"]).dt.date

    corrected_days = []
    reports = []

    days = sorted({pd.Timestamp(t).date() for t in ds_model.time.values})
    for day in days:
        day_mask  = ds_model["time"].dt.date == day
        ds_day    = ds_model.sel(time=day_mask)
        data_3d   = ds_day[variable].values          # (n_hours, ny, nx)
        daily_sum = day_accumulation(ds_model, variable, day, obs_utc_cutoff)

        obs_day   = df_obs[df_obs["date"] == day].dropna(subset=["value"])
        rep       = {"date": str(day), "n_stations": len(obs_day),
                     "idw_power": idw_power, "radius_km": radius_km}

        if obs_day.empty or len(obs_day) < min_stations:
            rep["correction"] = "skipped — insufficient observations"
            corrected_days.append(ds_day)
            reports.append(rep)
            continue

        stn_lat = obs_day["lat"].values
        stn_lon = obs_day["lon"].values
        obs_mm  = obs_day["value"].values

        mdl_at_stns = _interp_to_stations(daily_sum, grid_lat, grid_lon, stn_lat, stn_lon)
        ratio, additive = _build_ratio_field(
            daily_sum, mdl_at_stns, obs_mm, stn_lat, stn_lon,
            grid_lat, grid_lon, idw_power, radius_km,
            min_stations, max_ratio, min_thresh,
        )

        zero_mask = data_3d < min_thresh
        corrected = np.where(zero_mask, data_3d + additive[np.newaxis],
                             data_3d * ratio[np.newaxis])
        corrected = np.clip(corrected, 0.0, None)

        rep.update({
            "model_daily_mean_before": float(np.nanmean(daily_sum)),
            "model_daily_mean_after":  float(np.nanmean(corrected.sum(axis=0))),
            "obs_daily_mean":          float(np.nanmean(obs_mm)),
            "ratio_mean":              float(np.nanmean(ratio)),
            "correction":              "applied",
        })
        reports.append(rep)

        ds_c = ds_day.copy()
        ds_c[variable] = xr.DataArray(corrected, dims=ds_day[variable].dims,
                                       coords=ds_day[variable].coords,
                                       attrs={**ds_day[variable].attrs,
                                              "bias_corrected": "IDW"})
        corrected_days.append(ds_c)
        logger.info(
            f"{day}: {len(obs_day)} stn, "
            f"mean {rep['model_daily_mean_before']:.2f} → "
            f"{rep['model_daily_mean_after']:.2f}"
        )

    return xr.concat(corrected_days, dim="time"), reports


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

    data      = ds_model[variable].values.copy()   # (n_time, ny, nx)
    corrected = np.interp(data, model_q, obs_q).clip(0).astype(np.float32)

    ds_out = ds_model.copy()
    ds_out[variable] = xr.DataArray(
        corrected, dims=ds_model[variable].dims,
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

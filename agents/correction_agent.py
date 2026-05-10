"""
Bias Correction Agent — uses an LLM to choose IDW parameters and strategy,
validates the result, and retries with adjusted parameters if needed.

Full ReAct loop: choose → apply → validate → (accept OR adjust + retry).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd
import xarray as xr
from motus.agent import ReActAgent
from motus.tools import tool

from utils.client import make_client

from tools import correct, validate

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """
You are an expert in statistical bias correction of numerical weather model output.

You have two correction methods available:

IDW (Inverse Distance Weighting)
  Builds a spatially-varying multiplicative ratio field from station observations
  and applies it grid-point by grid-point.
  Works when: the obs-model signal is spatially coherent -- the model and
  observations agree on where precipitation occurs and the bias varies gradually
  across the domain. This signal is consistent with widespread frontal
  precipitation, well-organised rain shields, or any event where the model
  captures the large-scale spatial pattern correctly.
  Fails when: the signal is spatially variable or location-uncertain -- high CV
  means precipitation is concentrated in small cells so nearby station ratios are
  unrepresentative, and low spatial correlation means the ratio field is noise.
  This signal is consistent with convective cells, embedded convection, MCS
  cores, or spatial phase errors where the model places the storm in the wrong
  location at the grid scale.

Quantile Mapping (QM)
  Maps the model precipitation distribution to match the observed distribution.
  Works when: the signal is spatially variable or location-uncertain -- corrects
  the distribution shape without requiring spatial coherence, robust to phase
  errors and localised extremes.
  Limitation: does not correct the spatial placement of precipitation features.

Your workflow:
1. Call diagnose_regime. It returns a signal characterisation (spatially_coherent
   or spatially_variable) along with the raw diagnostics (CV, spatial correlation,
   wet fraction). Read both -- the label is a guide, not a hard rule. The
   thresholds that produce the label (CV > 2, r < 0.3) are indicative, so use
   your judgement when values are near the boundary.
2. If spatially_coherent:
   a. Call tune_idw_parameters — it finds the best (p, radius_km) via leave-one-out
      cross-validation at station locations. Use the returned best_p and
      best_radius_km directly; do not guess or override unless the LOO-RMSE is
      suspiciously high (e.g. > 5× the pre-correction RMSE).
   b. Call apply_correction with those parameters.
   c. If RMSE still degrades > 5%, the signal is not recoverable by IDW regardless
      of parameters — try QM.
3. If spatially_variable:
   a. Try QM first (apply_qm_correction).
   b. If QM degrades RMSE > 5%, call tune_idw_parameters then apply_correction
      as a fallback.
4. You have up to {max_retries} total attempts across both methods.
5. Accept the best result, or reject all corrections if none improve performance.
6. Call finish_correction with your final decision and full reasoning.

ALWAYS call narrate() before each major step.
""".strip()


class CorrectionAgent:

    def __init__(self, config):
        self.config = config
        self.ds_model: xr.Dataset | None = None
        self.ds_corrected: xr.Dataset | None = None
        self.df_obs: pd.DataFrame | None = None
        self.variable: str = ""
        self.correction_log: list[dict] = []
        self.attempt: int = 0
        self._accepted = False
        self._agent = self._build_agent()

    def _build_agent(self) -> ReActAgent:
        cfg = self.config

        def _log(msg: str):
            logger.info(f"  [correction] {msg}")

        @tool
        def get_station_density() -> str:
            """
            Return station density and domain size to help choose IDW radius.
            """
            import numpy as np
            from utils.geo import domain_bbox_from_wrf, EARTH_RADIUS_KM
            import math

            if self.df_obs is None or self.df_obs.empty:
                return json.dumps({"n_stations": 0, "density_per_10000km2": 0})

            if cfg.domain.wrf_geo_file:
                bb = domain_bbox_from_wrf(cfg.domain.wrf_geo_file)
            else:
                bb = {"south": cfg.domain.lat_min, "north": cfg.domain.lat_max,
                      "west": cfg.domain.lon_min,  "east":  cfg.domain.lon_max}

            dlat = bb["north"] - bb["south"]
            mid_lat = (bb["north"] + bb["south"]) / 2
            dlon = (bb["east"] - bb["west"]) * math.cos(math.radians(mid_lat))
            area_km2 = dlat * 111.0 * dlon * 111.0

            n_stns = self.df_obs["station_id"].nunique()
            density = n_stns / area_km2 * 10_000

            result = {
                "n_stations":            n_stns,
                "domain_area_km2":       round(area_km2),
                "density_per_10000km2":  round(density, 2),
                "mean_lat":              round(mid_lat, 2),
            }
            _log(f"Station density: {n_stns} stations over {round(area_km2):,} km²"
                 f" = {round(density, 2)} per 10,000 km²")
            return json.dumps(result, indent=2)

        @tool
        def get_bias_overview() -> str:
            """Return domain-mean and station-level bias metrics before any correction."""
            _log("Computing bias metrics at station locations…")
            if self.ds_model is None or self.df_obs is None:
                return "No data loaded."
            metrics = validate.compute_metrics(
                self.ds_model, self.ds_model, self.df_obs, self.variable,
                obs_utc_cutoff=getattr(cfg, "obs_utc_cutoff", 0))
            pre = metrics.get("pre_correction", {})
            _log(f"  Bias={pre.get('bias', float('nan')):.3f} mm"
                 f"  RMSE={pre.get('rmse', float('nan')):.3f} mm"
                 f"  r={pre.get('corr', float('nan')):.3f}"
                 f"  n={pre.get('n', 0)} pairs")
            return json.dumps(metrics, indent=2, default=str)

        @tool
        def apply_correction(idw_power: float, radius_km: float) -> str:
            """
            Apply IDW bias correction with the given parameters.
            Returns validation metrics so you can decide to accept or retry.
            """
            self.attempt += 1
            if self.attempt > cfg.max_correction_retries:
                return f"Maximum retries ({cfg.max_correction_retries}) reached."

            _log(f"Attempt {self.attempt}: applying IDW correction"
                 f" (p={idw_power}, radius={radius_km} km)…")
            ds_corr, day_reports = correct.apply_correction(
                ds_model=self.ds_model,
                df_obs=self.df_obs,
                variable=self.variable,
                idw_power=idw_power,
                radius_km=radius_km,
                min_stations=cfg.require_min_stations,
                obs_utc_cutoff=getattr(cfg, "obs_utc_cutoff", 0),
            )
            self.ds_corrected = ds_corr

            metrics = validate.compute_metrics(
                self.ds_model, ds_corr, self.df_obs, self.variable,
                obs_utc_cutoff=getattr(cfg, "obs_utc_cutoff", 0))
            post = metrics.get("post_correction", {})
            imp  = metrics.get("improvement", {})
            _log(f"  Result → Bias={post.get('bias', float('nan')):.3f} mm"
                 f"  RMSE={post.get('rmse', float('nan')):.3f} mm"
                 f"  r={post.get('corr', float('nan')):.3f}"
                 f"  RMSE improvement={imp.get('rmse_improvement_pct', float('nan')):.1f}%")

            self.correction_log.append({
                "attempt":     self.attempt,
                "method":      "idw",
                "idw_power":   idw_power,
                "radius_km":   radius_km,
                "day_reports": day_reports,
                "metrics":     metrics,
            })
            return json.dumps(metrics, indent=2, default=str)

        @tool
        def finish_correction(accepted: bool, reasoning: str) -> str:
            """
            Accept or reject the last applied correction.
            accepted: True to keep, False to discard (use uncorrected data).
            reasoning: plain-language explanation of the decision.
            """
            self._accepted = accepted
            self.correction_log.append({
                "action":    "finish",
                "accepted":  accepted,
                "reasoning": reasoning,
            })
            status = "ACCEPTED" if accepted else "REJECTED"
            _log(f"Correction {status} — {reasoning}")
            return f"Correction {status.lower()}. {reasoning}"

        @tool
        def narrate(message: str) -> str:
            """Share your reasoning at any point."""
            _log(message)
            return "ok"

        @tool
        def diagnose_regime() -> str:
            """
            Classify the precipitation regime as convective or stratiform.
            Returns spatial variability metrics and a method recommendation.
            Call this first before choosing a correction method.
            """
            _log("Diagnosing precipitation regime…")
            result = correct.diagnose_regime(
                self.ds_model, self.df_obs, self.variable,
                obs_utc_cutoff=getattr(cfg, "obs_utc_cutoff", 0))
            _log(f"  Regime: {result['regime']}  CV={result['mean_cv']:.2f}"
                 f"  spatial_r={result['mean_spatial_corr']}")
            return json.dumps(result, indent=2)

        @tool
        def apply_qm_correction() -> str:
            """
            Apply global quantile mapping correction.
            Maps the model precipitation distribution to match the observed
            distribution at station locations. No spatial parameters needed.
            Returns validation metrics so you can decide to accept or retry.
            """
            self.attempt += 1
            if self.attempt > cfg.max_correction_retries:
                return f"Maximum retries ({cfg.max_correction_retries}) reached."

            _log(f"Attempt {self.attempt}: applying quantile mapping…")
            ds_corr, reports = correct.apply_quantile_mapping(
                ds_model=self.ds_model,
                df_obs=self.df_obs,
                variable=self.variable,
                obs_utc_cutoff=getattr(cfg, "obs_utc_cutoff", 0),
            )
            self.ds_corrected = ds_corr

            metrics = validate.compute_metrics(
                self.ds_model, ds_corr, self.df_obs, self.variable,
                obs_utc_cutoff=getattr(cfg, "obs_utc_cutoff", 0))
            post = metrics.get("post_correction", {})
            imp  = metrics.get("improvement", {})
            _log(f"  Result → Bias={post.get('bias', float('nan')):.3f} mm"
                 f"  RMSE={post.get('rmse', float('nan')):.3f} mm"
                 f"  r={post.get('corr', float('nan')):.3f}"
                 f"  RMSE improvement={imp.get('rmse_improvement_pct', float('nan')):.1f}%")

            self.correction_log.append({
                "attempt":      self.attempt,
                "method":       "quantile_mapping",
                "qm_reports":   reports,
                "metrics":      metrics,
            })
            return json.dumps(metrics, indent=2, default=str)

        @tool
        def tune_idw_parameters() -> str:
            """
            Find the best IDW (p, radius_km) via leave-one-out cross-validation
            at station locations across all days in the period.
            Tries 25 combinations in parallel (p ∈ {1.0,1.5,2.0,2.5,3.0},
            radius ∈ {50,75,100,150,200} km). Call this before apply_correction
            so you use data-driven parameters instead of heuristic guesses.
            Returns best_p, best_radius_km, and full CV scores.
            """
            import itertools
            import math
            from concurrent.futures import ThreadPoolExecutor, as_completed

            import numpy as np
            from scipy.spatial import cKDTree

            from utils.accumulation import day_accumulation
            from utils.geo import EARTH_RADIUS_KM, km_to_chord, latlon_to_xyz

            _log("Tuning IDW parameters via leave-one-out CV…")

            if self.ds_model is None or self.df_obs is None:
                return "No data loaded."

            df = self.df_obs.copy()
            df["date"] = pd.to_datetime(df["date"]).dt.date
            grid_lat = self.ds_model["lat"].values
            grid_lon = self.ds_model["lon"].values
            min_thresh = 0.1
            max_ratio  = 10.0
            obs_cutoff = getattr(cfg, "obs_utc_cutoff", 0)

            # Collect station-day ratios (obs/model) where model is wet
            all_lats, all_lons, all_ratios = [], [], []
            days = sorted({pd.Timestamp(t).date() for t in self.ds_model.time.values})
            grid_tree = cKDTree(latlon_to_xyz(grid_lat.ravel(), grid_lon.ravel()))

            for day in days:
                model_day = day_accumulation(self.ds_model, self.variable, day, obs_cutoff)
                obs_day   = df[df["date"] == day].dropna(subset=["value"])
                if obs_day.empty:
                    continue
                _, idx = grid_tree.query(
                    latlon_to_xyz(obs_day["lat"].values, obs_day["lon"].values),
                    k=1, workers=-1)
                mdl_at = model_day.ravel()[idx].astype(float)
                obs_mm = obs_day["value"].values.astype(float)
                for i in range(len(mdl_at)):
                    if (mdl_at[i] > min_thresh and np.isfinite(obs_mm[i])
                            and obs_mm[i] >= 0):
                        ratio = min(obs_mm[i] / mdl_at[i], max_ratio)
                        all_lats.append(obs_day["lat"].values[i])
                        all_lons.append(obs_day["lon"].values[i])
                        all_ratios.append(ratio)

            n = len(all_lats)
            if n < 5:
                return json.dumps({"error": "too few wet station-day pairs for CV",
                                   "n": n, "best_p": 2.0, "best_radius_km": 100})

            lats   = np.array(all_lats)
            lons   = np.array(all_lons)
            ratios = np.array(all_ratios)
            xyz    = latlon_to_xyz(lats, lons)
            stn_tree = cKDTree(xyz)

            p_values      = [1.0, 1.5, 2.0, 2.5, 3.0]
            radius_values = [50, 75, 100, 150, 200]
            k_query       = min(n, 51)  # +1 so we can drop the self-match

            # Pre-query once for the largest radius to reuse distances
            max_chord = km_to_chord(max(radius_values))
            cd_all, ni_all = stn_tree.query(xyz, k=k_query,
                                            distance_upper_bound=max_chord,
                                            workers=-1)
            if cd_all.ndim == 1:
                cd_all = cd_all[:, np.newaxis]
                ni_all = ni_all[:, np.newaxis]

            def _loo_rmse(p: float, radius_km: float) -> float:
                chord = km_to_chord(radius_km)
                errors = []
                for i in range(n):
                    row_cd = cd_all[i]
                    row_ni = ni_all[i]
                    # exclude self and points beyond this radius
                    mask = (row_ni != i) & (row_cd <= chord) & np.isfinite(row_cd)
                    if not mask.any():
                        continue
                    d_km = (2.0 * EARTH_RADIUS_KM
                            * np.arcsin(np.clip(row_cd[mask] * 0.5, 0.0, 1.0)))
                    d_km = np.maximum(d_km, 1e-3)
                    w    = 1.0 / d_km ** p
                    pred = float(np.dot(w, ratios[row_ni[mask]]) / w.sum())
                    errors.append(pred - ratios[i])
                return float(np.sqrt(np.mean(np.array(errors) ** 2))) if errors else math.nan

            results = []
            with ThreadPoolExecutor() as ex:
                futures = {ex.submit(_loo_rmse, p, r): (p, r)
                           for p, r in itertools.product(p_values, radius_values)}
                for fut in as_completed(futures):
                    p, r = futures[fut]
                    rmse = fut.result()
                    results.append({"p": p, "radius_km": r, "loo_rmse": round(rmse, 4)})

            results.sort(key=lambda x: x["loo_rmse"] if math.isfinite(x["loo_rmse"]) else 1e9)
            best = results[0]
            _log(f"  Best: p={best['p']}, radius={best['radius_km']} km,"
                 f" LOO-RMSE={best['loo_rmse']:.4f}  (n={n} station-day pairs)")

            return json.dumps({
                "best_p":         best["p"],
                "best_radius_km": best["radius_km"],
                "best_loo_rmse":  best["loo_rmse"],
                "n_station_days": n,
                "top_5":          results[:5],
            }, indent=2)

        tools = [narrate, get_station_density, get_bias_overview,
                 diagnose_regime, tune_idw_parameters,
                 apply_correction, apply_qm_correction,
                 finish_correction]

        client, model_name = make_client(cfg.llm_model)
        return ReActAgent(
            client=client,
            model_name=model_name,
            system_prompt=_SYSTEM_PROMPT.format(
                max_retries=cfg.max_correction_retries),
            tools=tools,
            max_steps=30,
        )

    def run(self, ds_model: xr.Dataset, df_obs: pd.DataFrame,
            variable: str) -> tuple[xr.Dataset, list[dict]]:
        """
        Run the correction loop.
        Returns (corrected_or_original_dataset, correction_log).
        """
        self.ds_model     = ds_model
        self.df_obs       = df_obs
        self.variable     = variable
        self.correction_log = []
        self.attempt      = 0
        self._accepted    = False
        self.ds_corrected = ds_model  # default: no correction

        prompt = (
            f"Apply bias correction to {variable} model output.\n"
            f"Date range: {self.config.date_start} to {self.config.date_end}\n"
            f"Model source: {self.config.model_source}\n\n"
            f"Step 1: call diagnose_regime to classify the precipitation regime.\n"
            f"Step 2: call get_station_density and get_bias_overview.\n"
            f"Step 3: choose the appropriate method (QM or IDW) per your instructions.\n"
            f"Step 4: apply, validate, retry if needed, then finish_correction."
        )

        import asyncio
        asyncio.get_event_loop().run_until_complete(self._agent(prompt))

        if self._accepted and self.ds_corrected is not None:
            return self.ds_corrected, self.correction_log
        else:
            logger.warning("Correction agent rejected the correction — returning uncorrected data")
            return ds_model, self.correction_log

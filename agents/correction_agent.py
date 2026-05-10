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
  Builds a spatially-varying ratio field from station observations.
  Best for: stratiform precipitation or temperature with smooth, coherent bias.
  Fails for: convective precipitation — high spatial variability and phase errors
  cause the ratio field to be noise rather than signal, degrading performance.

Quantile Mapping (QM)
  Maps the model distribution to match the observed distribution globally.
  Best for: convective precipitation — corrects the distribution shape without
  assuming spatial coherence. Robust to phase errors.
  Limitation: does not correct spatial placement of precipitation features.

Your workflow:
1. Call diagnose_regime to classify the precipitation regime.
2. If STRATIFORM: try IDW first. If IDW degrades RMSE > 5%, try QM.
3. If CONVECTIVE: try QM first. If QM degrades RMSE > 5%, try IDW.
4. You have up to {max_retries} total attempts across both methods.
5. Accept the best result, or reject all corrections if none improve performance.
6. Call finish_correction with your final decision and full reasoning.

ALWAYS call narrate() before each major step.

IDW parameter guidance:
- Dense network (>10 stn/10,000 km²): radius 50 km, p = 2
- Sparse network (<3 stn/10,000 km²): radius 100 km, p = 1.5
- High spatial variability: lower radius, higher p
- Smooth large-scale bias:  larger radius, lower p
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

        tools = [narrate, get_station_density, get_bias_overview,
                 diagnose_regime, apply_correction, apply_qm_correction,
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

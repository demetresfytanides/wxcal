"""
QA/QC Agent — uses an LLM to inspect model and observation data and decide
what QA/QC steps to apply, with full reasoning logged for the report.

The agent receives data summaries (never raw arrays) and calls tool functions
to apply its decisions. It explicitly logs WHY each decision was made.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import xarray as xr
import pandas as pd
from motus.agent import ReActAgent
from motus.tools import tool

from utils.client import make_client

from tools import qaqc

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """
You are an expert meteorological data quality control analyst with 20+ years of
experience reviewing NWP model output and rain-gauge observations.

Your job:
1. Inspect the data summaries you are given (statistics, percentiles, counts).
2. Decide which QA/QC actions to apply — or explicitly decide that the data is
   already clean and no action is needed. Not every dataset requires changes.
3. Call the available tools to apply your decisions.
4. For EVERY decision, explain your reasoning clearly and specifically
   (e.g. "p99 = 320 mm/day exceeds the physical maximum for the region, flagging
   values > 300 mm/day as extreme").

Rules:
- ALWAYS call narrate() before and after each decision to explain your thinking.
- Base decisions on the statistics provided, not assumed defaults.
- If data looks clean, call no_change_* and state why.
- Prefer conservative flagging — when in doubt, keep the data.
- After applying all actions, call finish_qaqc with a concise summary.
""".strip()


class QAQCAgent:
    """
    Wraps a ReActAgent with data-aware QA/QC tools.
    State (ds, df) is held in the instance so tools can close over it.
    """

    def __init__(self, config):
        self.config = config
        self.ds: xr.Dataset | None = None
        self.df: pd.DataFrame | None = None
        self.decision_log: list[dict] = []
        self._done = False
        self._agent = self._build_agent()

    def _build_agent(self) -> ReActAgent:
        cfg = self.config

        # ── tool definitions (close over self) ──────────────────────────────

        def _log(msg: str):
            logger.info(f"  [QA/QC] {msg}")

        @tool
        def inspect_model_data(variable: str) -> str:
            """Return distribution statistics for the model grid variable."""
            _log(f"Inspecting model data for variable '{variable}'…")
            stats = qaqc.summarise_grid(self.ds, variable)
            _log(f"  mean={stats['mean']:.3f}  p99={stats['p99']:.3f}"
                 f"  max={stats['max']:.3f}  negatives={stats['n_negative']}"
                 f"  units={stats['units']}")
            return json.dumps(stats, indent=2)

        @tool
        def inspect_observations() -> str:
            """Return distribution statistics for the observation dataset."""
            _log("Inspecting observation dataset…")
            stats = qaqc.summarise_obs(self.df)
            _log(f"  stations={stats['n_stations']}  records={stats['n_records']}"
                 f"  mean={stats['mean']:.3f}  p99={stats['p99']:.3f}"
                 f"  max={stats['max']:.3f}  missing={stats['n_missing']}")
            return json.dumps(stats, indent=2)

        @tool
        def clip_negative_model(variable: str) -> str:
            """Clip negative model values to zero (e.g. interpolation artefacts)."""
            self.ds, log = qaqc.clip_negative_grid(self.ds, variable)
            self.decision_log.append(log)
            msg = f"Clipped {log['n_clipped']} negative model values to 0."
            _log(msg); return msg

        @tool
        def flag_extreme_model(variable: str, max_value: float) -> str:
            """Flag model values above max_value as NaN (unrealistic extremes)."""
            self.ds, log = qaqc.flag_extreme_grid(self.ds, variable, max_value)
            self.decision_log.append(log)
            msg = f"Flagged {log['n_flagged']} model values > {max_value} as NaN."
            _log(msg); return msg

        @tool
        def zero_trace_model(variable: str, min_threshold: float) -> str:
            """Set model values between 0 and min_threshold to exactly 0 (trace)."""
            self.ds, log = qaqc.zero_trace_grid(self.ds, variable, min_threshold)
            self.decision_log.append(log)
            msg = f"Zeroed {log['n_zeroed']} trace model values < {min_threshold}."
            _log(msg); return msg

        @tool
        def no_change_model(variable: str, reason: str) -> str:
            """Explicitly record that no QA/QC change was applied to the model data."""
            self.ds, log = qaqc.no_change_grid(self.ds, reason)
            self.decision_log.append(log)
            msg = f"Model data unchanged — {reason}"
            _log(msg); return msg

        @tool
        def clip_negative_obs() -> str:
            """Clip negative observation values to zero."""
            self.df, log = qaqc.clip_negative_obs(self.df)
            self.decision_log.append(log)
            msg = f"Clipped {log['n_clipped']} negative obs values to 0."
            _log(msg); return msg

        @tool
        def flag_extreme_obs(max_value: float) -> str:
            """Flag observation values above max_value as missing."""
            self.df, log = qaqc.flag_extreme_obs(self.df, max_value)
            self.decision_log.append(log)
            msg = f"Flagged {log['n_flagged']} obs values > {max_value} as NaN."
            _log(msg); return msg

        @tool
        def drop_sparse_stations(min_days: int) -> str:
            """Remove stations that reported on fewer than min_days days."""
            self.df, log = qaqc.drop_sparse_stations(self.df, min_days)
            self.decision_log.append(log)
            msg = f"Dropped {log['removed']} stations with fewer than {min_days} reporting days."
            _log(msg); return msg

        @tool
        def no_change_obs(reason: str) -> str:
            """Explicitly record that no QA/QC change was applied to observations."""
            self.df, log = qaqc.no_change_obs(self.df, reason)
            self.decision_log.append(log)
            msg = f"Observations unchanged — {reason}"
            _log(msg); return msg

        @tool
        def finish_qaqc(summary: str) -> str:
            """Call this when QA/QC is complete with a plain-language summary."""
            self.decision_log.append({"action": "finish", "summary": summary})
            self._done = True
            _log(f"Done — {summary}")
            return "QA/QC complete."

        @tool
        def narrate(message: str) -> str:
            """Share your reasoning at any point."""
            _log(message)
            return "ok"

        tools = [
            narrate,
            inspect_model_data, inspect_observations,
            clip_negative_model, flag_extreme_model, zero_trace_model, no_change_model,
            clip_negative_obs, flag_extreme_obs, drop_sparse_stations, no_change_obs,
            finish_qaqc,
        ]

        client, model_name = make_client(cfg.llm_model)
        return ReActAgent(
            client=client,
            model_name=model_name,
            system_prompt=_SYSTEM_PROMPT,
            tools=tools,
            max_steps=20,
        )

    def run(self, ds: xr.Dataset, df: pd.DataFrame,
            variable: str) -> tuple[xr.Dataset, pd.DataFrame, list[dict]]:
        """Run QA/QC and return (clean_ds, clean_df, decision_log)."""
        self.ds = ds
        self.df = df
        self.decision_log = []
        self._done = False

        prompt = (
            f"Perform QA/QC on the model and observation data.\n"
            f"Variable: {variable}\n"
            f"Date range: {self.config.date_start} to {self.config.date_end}\n"
            f"Model source: {self.config.model_source}\n"
            f"Obs source: {self.config.obs_source}\n\n"
            f"Start by inspecting both datasets, then apply appropriate QA/QC actions. "
            f"Finish with finish_qaqc."
        )

        import asyncio
        asyncio.get_event_loop().run_until_complete(self._agent(prompt))

        return self.ds, self.df, self.decision_log

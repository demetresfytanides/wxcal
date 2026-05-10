"""
wxCal — Motus Cloud serve entry point.

Wraps the batch wxCal pipeline in a ReActAgent that:
  - Accepts natural-language or structured parameter messages
  - Runs the full pipeline (ingest → regrid → QA/QC → correction → export → report)
  - Returns a summary with output paths

Deploy:
    motus deploy --name wxcal wxcal_serve:wxcal_agent

Interact (local):
    uv run python wxcal_serve.py
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from motus.agent import ReActAgent
from motus.tools import tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are wxCal, a weather calibration assistant. You bias-correct numerical weather \
prediction (NWP) model output against surface observations using IDW or Quantile Mapping, \
then produce a transparent PDF report documenting every decision.

When the user supplies pipeline parameters, call run_wxcal_pipeline with those values.
Always confirm what you are about to run before calling the tool, and report results
clearly — including the paths of the corrected NetCDF file and the PDF report.

Required parameters
-------------------
start        YYYY-MM-DD start date
end          YYYY-MM-DD end date
Domain (one of):
  lat_min / lat_max / lon_min / lon_max   bounding box in decimal degrees
  geo_file                                 path to WRF geo_em.d01.nc
variable     precipitation | temperature | wind_u | wind_v  (default: precipitation)

Optional parameters
-------------------
model        hrrr | era5 | /path/to/file.nc       (default: hrrr)
obs          acis | /path/to/observations.csv      (default: acis)
dx_km        grid spacing km when using bbox       (default: 3.0)
obs_cutoff   UTC hour obs accumulation ends        (default: 12 — CoCoRaHS 7 am CDT)
output_format  netcdf | zarr | both                (default: netcdf)
llm_model    LLM model string for internal agents  (default: anthropic/claude-sonnet-4-6)
max_retries  correction retry limit                (default: 3)
"""


@tool
def run_wxcal_pipeline(
    start: str,
    end: str,
    lat_min: float | None = None,
    lat_max: float | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
    dx_km: float = 3.0,
    geo_file: str | None = None,
    variable: str = "precipitation",
    model: str = "hrrr",
    obs: str = "acis",
    obs_cutoff: int = 12,
    output_format: str = "netcdf",
    llm_model: str = "anthropic/claude-sonnet-4-6",
    max_retries: int = 3,
) -> dict:
    """
    Run the full wxCal bias-correction pipeline and return output paths.

    Parameters
    ----------
    start        : Start date YYYY-MM-DD
    end          : End date   YYYY-MM-DD
    lat_min      : Bounding-box south edge (degrees). Required unless geo_file is given.
    lat_max      : Bounding-box north edge (degrees).
    lon_min      : Bounding-box west  edge (degrees).
    lon_max      : Bounding-box east  edge (degrees).
    dx_km        : Grid spacing km when using bbox (default 3.0).
    geo_file     : Path to WRF geo_em.d01.nc (alternative to bbox).
    variable     : Variable to correct: precipitation | temperature | wind_u | wind_v
    model        : Model source: hrrr | era5 | /path/to/file.nc
    obs          : Observation source: acis | /path/to/observations.csv
    obs_cutoff   : UTC hour obs 24-h accumulation ends (12 = CoCoRaHS standard).
    output_format: Output format: netcdf | zarr | both
    llm_model    : LLM model string for the three internal agents.
    max_retries  : Maximum correction retries for the bias correction agent.
    """
    from config import WxCalConfig, DomainConfig
    from orchestrator import run as _run_pipeline

    if geo_file:
        domain = DomainConfig(wrf_geo_file=Path(geo_file))
    elif all(v is not None for v in [lat_min, lat_max, lon_min, lon_max]):
        domain = DomainConfig(
            lat_min=lat_min, lat_max=lat_max,
            lon_min=lon_min, lon_max=lon_max,
            dx_km=dx_km,
        )
    else:
        return {
            "error": "Provide either geo_file or all four of lat_min / lat_max / lon_min / lon_max."
        }

    model_src: str | Path = model
    if model not in ("hrrr", "era5") and Path(model).exists():
        model_src = Path(model)

    obs_src: str | Path = obs
    if obs != "acis" and Path(obs).exists():
        obs_src = Path(obs)

    config = WxCalConfig(
        date_start=date.fromisoformat(start),
        date_end=date.fromisoformat(end),
        domain=domain,
        model_source=model_src,
        model_variable=variable,
        obs_source=obs_src,
        obs_utc_cutoff=obs_cutoff,
        output_format=output_format,
        llm_model=llm_model,
        max_correction_retries=max_retries,
    )

    output_paths = _run_pipeline(config)

    corrected_nc = next((str(p) for p in output_paths if p.suffix == ".nc"), None)
    pdf_report   = next((str(p) for p in output_paths if p.suffix == ".pdf"), None)

    return {
        "status":       "complete",
        "outputs":      [str(p) for p in output_paths],
        "corrected_nc": corrected_nc,
        "pdf_report":   pdf_report,
    }


def _make_agent() -> ReActAgent:
    from utils.client import make_client
    llm_model = os.getenv("WXCAL_LLM_MODEL", "anthropic/claude-sonnet-4-6")
    client, model_name = make_client(llm_model)
    return ReActAgent(
        client=client,
        model_name=model_name,
        system_prompt=SYSTEM_PROMPT,
        tools=[run_wxcal_pipeline],
        max_steps=5,
    )


wxcal_agent = _make_agent()


if __name__ == "__main__":
    async def main():
        print("wxCal — Weather Calibration Agent")
        print("Describe your pipeline run (or type 'quit' to exit).\n")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue
            response = await wxcal_agent(user_input)
            print(f"\nwxCal: {response}\n")

    asyncio.run(main())

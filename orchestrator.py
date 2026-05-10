"""
wxCal — Weather Calibration Agent
==================================
Orchestrates the full pipeline:

  load model → regrid → load obs
        ↓
  QA/QC Agent (LLM inspects data, decides thresholds, logs reasoning)
        ↓
  Bias Correction Agent (LLM chooses IDW params, validates, retries if needed)
        ↓
  Export (NetCDF + optional Zarr)
        ↓
  Report Agent (LLM writes transparent narrative PDF)

Usage:
    python orchestrator.py \\
        --model  hrrr \\
        --obs    acis \\
        --geo    /path/to/geo_em.d01.nc \\
        --start  2024-06-01 \\
        --end    2024-06-03 \\
        --var    precipitation

    # Bounding-box alternative to --geo:
        --bbox 41.0 43.5 -89.0 -86.0 --dx 3

    # Local file instead of HRRR:
        --model /path/to/model.nc

    # Zarr output:
        --format zarr

    # Use a different model:
        --llm anthropic/claude-opus-4-7
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY (OpenRouter) from .env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wxCal")


def run(config) -> list[Path]:
    from tools.ingest       import load_model_data
    from tools.observations import load_observations
    from tools.regrid       import regrid
    from tools.export       import export
    from agents.qaqc_agent       import QAQCAgent
    from agents.correction_agent import CorrectionAgent
    from agents.report_agent     import ReportAgent

    variable = config.model_variable

    # ── 1. Load ──────────────────────────────────────────────────────────────
    logger.info(f"Loading model data ({config.model_source}, {variable})…")
    ds_raw = load_model_data(config)

    logger.info("Loading observations…")
    df_obs_raw = load_observations(config)

    # ── 2. Regrid ────────────────────────────────────────────────────────────
    logger.info("Regridding model data onto target grid…")
    ds_grid = regrid(ds_raw, config, variable)

    # ── 3. QA/QC (agentic) ───────────────────────────────────────────────────
    logger.info("Running QA/QC agent…")
    qaqc = QAQCAgent(config)
    ds_clean, df_clean, qaqc_log = qaqc.run(ds_grid, df_obs_raw, variable)
    logger.info(f"  QA/QC complete: {len(qaqc_log)} decisions logged")

    # ── 4. Bias correction (agentic, with ReAct retry loop) ──────────────────
    logger.info("Running bias correction agent…")
    correction = CorrectionAgent(config)
    ds_corrected, correction_log = correction.run(ds_clean, df_clean, variable)
    logger.info(f"  Correction complete: {len(correction_log)} entries logged")

    # ── 5. Export ────────────────────────────────────────────────────────────
    logger.info(f"Exporting ({config.output_format})…")
    output_paths = export(ds_corrected, config, tag="corrected")

    # ── 6. Report (agentic) ──────────────────────────────────────────────────
    logger.info("Running report agent…")
    reporter = ReportAgent(config)
    pdf_path = reporter.run(
        ds_before=ds_clean,
        ds_after=ds_corrected,
        df_obs=df_clean,
        variable=variable,
        qaqc_log=qaqc_log,
        correction_log=correction_log,
        output_paths=output_paths,
    )
    output_paths.append(pdf_path)

    logger.info("Pipeline complete.")
    for p in output_paths:
        logger.info(f"  → {p}")

    return output_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="wxCal — Weather Calibration Agent")

    # Time range
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end",   required=True, help="End date   YYYY-MM-DD")

    # Domain
    dom = p.add_mutually_exclusive_group(required=True)
    dom.add_argument("--geo",  help="Path to WRF geo_em.d01.nc")
    dom.add_argument("--bbox", nargs=4, type=float,
                     metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                     help="Bounding box (degrees)")
    p.add_argument("--dx", type=float, default=3.0,
                   help="Grid spacing in km when using --bbox (default 3)")

    # Sources
    p.add_argument("--model", default="hrrr",
                   help="Model source: 'hrrr', 'era5', or path to local file")
    p.add_argument("--obs",   default="acis",
                   help="Obs source: 'acis' or path to CSV")
    p.add_argument("--var",   default="precipitation",
                   help="Variable: precipitation | temperature | wind_u | wind_v")
    p.add_argument("--hours", nargs="+", type=int, default=None,
                   help="Subset of UTC hours to process (default: all)")
    p.add_argument("--obs-cutoff", type=int, default=12,
                   help="UTC hour at which obs 24-h accumulation ends "
                        "(12 = CoCoRaHS 7 am CDT standard; 0 = calendar day)")

    # Output
    p.add_argument("--outdir", default="data",   help="Base output directory")
    p.add_argument("--format", default="netcdf",
                   choices=["netcdf", "zarr", "both"], help="Output format")

    # Agent
    p.add_argument("--llm", default="anthropic/claude-sonnet-4-6",
                   help="LLM model string (OpenRouter format, e.g. anthropic/claude-sonnet-4-6)")
    p.add_argument("--retries", type=int, default=3,
                   help="Max correction retries for the bias correction agent")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel download threads")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    from config import WxCalConfig, DomainConfig

    if args.geo:
        domain = DomainConfig(wrf_geo_file=Path(args.geo))
    else:
        lat_min, lat_max, lon_min, lon_max = args.bbox
        domain = DomainConfig(lat_min=lat_min, lat_max=lat_max,
                              lon_min=lon_min, lon_max=lon_max,
                              dx_km=args.dx)

    model_src = args.model
    if model_src not in ("hrrr", "era5") and Path(model_src).exists():
        model_src = Path(model_src)

    obs_src = args.obs
    if obs_src != "acis" and Path(obs_src).exists():
        obs_src = Path(obs_src)

    config = WxCalConfig(
        date_start=date.fromisoformat(args.start),
        date_end=date.fromisoformat(args.end),
        domain=domain,
        model_source=model_src,
        model_variable=args.var,
        model_hours=args.hours,
        obs_source=obs_src,
        obs_utc_cutoff=args.obs_cutoff,
        base_dir=Path(args.outdir),
        output_format=args.format,
        llm_model=args.llm,
        max_correction_retries=args.retries,
        max_download_workers=args.workers,
    )

    try:
        outputs = run(config)
        print("\nOutputs:")
        for p in outputs:
            print(f"  {p}")
        sys.exit(0)
    except Exception as exc:
        logger.exception(f"Pipeline failed: {exc}")
        sys.exit(1)

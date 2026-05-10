<img src="assets/logo.png" alt="wxCal logo" width="400"/>

# wxCal — Weather Calibration Agent

**wxCal** is an agentic pipeline that bias-corrects numerical weather prediction (NWP) and AI-based model output against surface observations, then produces a transparent, publication-ready PDF report documenting every decision made.

> **Motus use case** — wxCal is built entirely on [LithosAI Motus](https://motus.lithosai.com), demonstrating how Motus `ReActAgent` loops can replace hand-coded correction logic with autonomous agents that reason, act, validate, and self-correct.

---

## Workflow

![wxCal pipeline](assets/workflow.png)

The pipeline runs six stages in sequence:

| Stage | What happens |
|---|---|
| **Ingest** | Downloads HRRR hourly GRIB2 from the NOAA public S3 bucket (or reads a local file) |
| **Regrid** | Reprojects onto the target grid — either a WRF `geo_em` domain or a lat/lon bounding box |
| **Observations** | Fetches daily station data from the NOAA ACIS multi-network API or a local CSV |
| **QA/QC Agent** | LLM inspects data statistics, decides thresholds, flags/clips/drops, logs every decision |
| **Correction Agent** | LLM diagnoses precipitation regime, selects and applies IDW or Quantile Mapping, validates, retries or rejects |
| **Report Agent** | LLM writes a full narrative; Python generates comparison maps and scatter plots; XeLaTeX compiles a PDF |

---

## The Motus Agent Layer

wxCal replaces traditional hard-coded correction scripts with three autonomous agents, each built on a **Motus `ReActAgent`** loop:

```
Reason → Call tool → Observe result → Decide next step → repeat
```

Each agent receives statistical summaries (never raw arrays), calls Python functions via the `@tool` decorator, and logs every decision for the report.

### QA/QC Agent

Inspects model grid statistics and observation distributions. Decides whether to clip negatives, flag extremes, zero sub-instrument trace values, or drop sparse stations — with explicit written reasoning for every choice. If the data is already clean, it records that too.

### Bias Correction Agent

1. **Diagnoses the precipitation regime** — computes spatial coefficient of variation (CV) and obs–model spatial correlation per day to classify as *convective* or *stratiform*
2. **Selects a method** — IDW for stratiform (spatially coherent bias), Quantile Mapping for convective (distribution correction robust to phase errors)
3. **Applies, validates, and retries** — if RMSE degrades > 5%, switches to the alternative method; rejects all corrections if neither improves the baseline
4. **Reports honestly** — if correction is harmful, returns uncorrected data and explains why

### Report Agent

Receives all decision logs, writes four narrative sections (executive summary, QA/QC narrative, correction narrative, conclusions) with specific numbers cited throughout. Python generates spatial comparison maps (before / after / difference) and obs-vs-model scatter plots. XeLaTeX compiles the final PDF.

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/your-org/wxcal.git
cd wxcal
uv sync

# Configure API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Run over a bounding box
python orchestrator.py \
    --model hrrr \
    --obs   acis \
    --bbox  41.0 43.5 -89.0 -86.0 \
    --start 2024-06-01 \
    --end   2024-06-03 \
    --var   precipitation

# Run using a WRF geo_em domain file
python orchestrator.py \
    --model hrrr \
    --obs   acis \
    --geo   /path/to/geo_em.d01.nc \
    --start 2024-06-01 \
    --end   2024-06-03 \
    --var   precipitation
```

Outputs land in `data/output/`:
- `wxcal_precipitation_YYYYMMDD_YYYYMMDD_corrected.nc` — bias-corrected NetCDF
- `wxcal_report_YYYYMMDD_YYYYMMDD.pdf` — full transparency report

---

## CLI Reference

```
python orchestrator.py [options]

Time range:
  --start YYYY-MM-DD        Start date (required)
  --end   YYYY-MM-DD        End date (required)

Domain (choose one):
  --geo   PATH              WRF geo_em.d01.nc file
  --bbox  LAT_MIN LAT_MAX LON_MIN LON_MAX
  --dx    KM                Grid spacing when using --bbox (default: 3 km)

Sources:
  --model hrrr|era5|PATH    Model source (default: hrrr)
  --obs   acis|PATH         Observation source (default: acis)
  --var   precipitation|temperature|wind_u|wind_v

Accumulation:
  --obs-cutoff HOUR         UTC hour obs accumulation ends (default: 12 = CoCoRaHS 7am CDT)

Output:
  --outdir  PATH            Base output directory (default: data/)
  --format  netcdf|zarr|both

Agent:
  --llm     MODEL           LLM model string (default: claude-sonnet-4-6)
  --retries N               Max correction retries (default: 3)
  --workers N               Parallel download threads (default: 4)
```

---

## Environment

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Direct Anthropic API (local runs) |
| `MOTUS_CLOUD=1` | Route LLM calls through Motus Cloud / OpenRouter |
| `OPENAI_API_KEY` | OpenRouter key (when `MOTUS_CLOUD=1`) |

On Motus Cloud, `OPENAI_API_KEY` and `OPENAI_BASE_URL` are injected automatically — no configuration needed.

---

## Correction Methods

### IDW — Inverse Distance Weighting

Builds a spatially-varying multiplicative ratio field from station observations and applies it to every model grid point. Best for stratiform precipitation and temperature where the bias is spatially coherent.

### Quantile Mapping

Maps the model precipitation distribution to the observed distribution by constructing a piecewise-linear quantile transfer function from all station-day pairs. Applied uniformly across the grid. Best for convective precipitation where phase errors make spatially-varying corrections unreliable.

### Regime Diagnosis

Before correction, the agent computes:
- **CV** (spatial coefficient of variation of daily model precipitation)
- **r** (Pearson correlation between station observations and nearest-neighbour model values)

| CV > 2 or r < 0.3 | → Convective → Quantile Mapping first |
|---|---|
| Otherwise | → Stratiform → IDW first |

If the first method degrades RMSE by more than 5%, the agent falls back to the other. If neither improves the baseline, the correction is rejected and uncorrected data is returned.

---

## Observation Accumulation Window

CoCoRaHS and ACIS observers read gauges at approximately 7 am local time — 12:00 UTC in Central Daylight Time. An observation labelled date *D* represents the 24-hour window `[D-1 12:00 UTC, D 12:00 UTC)`. wxCal sums model hourly fields over that same window (`--obs-cutoff 12`) to avoid the 12-hour phase offset that would otherwise degrade spatial correlation for convective events.

---

## Stack

| Component | Role |
|---|---|
| [LithosAI Motus](https://motus.lithosai.com) | Agent orchestration (`ReActAgent`, `@tool`, cloud deploy) |
| [Anthropic Claude](https://anthropic.com) | LLM backbone for all three agents |
| [xarray](https://xarray.pydata.org) | Gridded model data (NetCDF / Zarr) |
| [scipy](https://scipy.org) | `cKDTree` nearest-neighbour, quantile functions |
| [NOAA HRRR S3](https://registry.opendata.aws/noaa-hrrr-pds/) | Public model archive |
| [NOAA ACIS API](https://www.rcc-acis.org) | Multi-network station observations |
| [XeLaTeX](https://tug.org/xetex/) | PDF report compilation |
| [matplotlib](https://matplotlib.org) | Spatial maps and scatter plots |

---

## Project Structure

```
wxcal/
├── orchestrator.py          # Pipeline entry point
├── config.py                # WxCalConfig dataclass
├── agents/
│   ├── qaqc_agent.py        # QA/QC ReActAgent
│   ├── correction_agent.py  # Bias correction ReActAgent
│   └── report_agent.py      # Report ReActAgent + LaTeX builder
├── tools/
│   ├── ingest.py            # HRRR / ERA5 / local file loader
│   ├── observations.py      # ACIS API + CSV loader
│   ├── regrid.py            # xESMF / scipy reprojection
│   ├── correct.py           # IDW + Quantile Mapping engines
│   ├── validate.py          # Bias / RMSE / correlation metrics
│   ├── qaqc.py              # QA/QC operations
│   └── export.py            # NetCDF / Zarr writer
├── utils/
│   ├── client.py            # Motus client selector (Anthropic vs OpenRouter)
│   ├── accumulation.py      # Obs accumulation window helper
│   └── geo.py               # Coordinate utilities
└── assets/
    ├── logo.png
    └── workflow.png
```

---

## License

MIT

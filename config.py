"""
wxCal configuration.

Domain can be specified two ways:
  1. WRF geo_em file  (wrf_geo_file=Path(...))
  2. Bounding box + resolution  (bbox + dx_km)

Model source can be:
  - "hrrr"       — NOAA public S3 bucket
  - "era5"       — local ERA5 NetCDF/GRIB file (path required)
  - Path(...)    — any local file (format auto-detected)

Observation source can be:
  - "acis"       — NOAA ACIS multi-network API
  - Path(...)    — local CSV with columns: station_id, lat, lon, date, value
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal


@dataclass
class DomainConfig:
    """Target grid specification — one of wrf_geo_file OR bbox must be given."""
    wrf_geo_file: Path | None = None

    # Bounding-box alternative (degrees)
    lat_min: float | None = None
    lat_max: float | None = None
    lon_min: float | None = None
    lon_max: float | None = None
    dx_km: float = 3.0       # target grid spacing

    def validate(self):
        if self.wrf_geo_file is None and any(
            v is None for v in [self.lat_min, self.lat_max, self.lon_min, self.lon_max]
        ):
            raise ValueError("Provide either wrf_geo_file or all of lat_min/lat_max/lon_min/lon_max")


@dataclass
class WxCalConfig:
    # ── Time range ───────────────────────────────────────────────────────────
    date_start: date
    date_end:   date

    # ── Domain ───────────────────────────────────────────────────────────────
    domain: DomainConfig

    # ── Model source ─────────────────────────────────────────────────────────
    model_source: str | Path = "hrrr"
    model_variable: str = "precipitation"  # "precipitation" | "temperature" | "wind"
    model_hours: list[int] | None = None   # subset of UTC hours; None = all available

    # ── Observation source ───────────────────────────────────────────────────
    obs_source: str | Path = "acis"

    # UTC hour at which the 24-h obs accumulation period ends on the reported
    # date.  CoCoRaHS observers read at ~7 am local time; for Central Daylight
    # Time that is 12:00 UTC.  Set to 0 for a plain calendar-day sum.
    obs_utc_cutoff: int = 12

    # ── Output ───────────────────────────────────────────────────────────────
    base_dir: Path = Path("data")
    output_format: Literal["netcdf", "zarr", "both"] = "netcdf"

    # ── Agent behaviour ──────────────────────────────────────────────────────
    llm_model: str = "anthropic/claude-sonnet-4-6"
    max_correction_retries: int = 3          # ReAct retry limit for bias correction
    require_min_stations: int = 5            # below this count agent skips correction

    # ── Defaults the agent may override ─────────────────────────────────────
    idw_power_default: float = 2.0
    idw_radius_km_default: float = 75.0
    max_download_workers: int = 4

    def __post_init__(self):
        self.domain.validate()
        self.base_dir = Path(self.base_dir)
        if isinstance(self.model_source, str) and self.model_source not in ("hrrr", "era5"):
            self.model_source = Path(self.model_source)
        if isinstance(self.obs_source, str) and self.obs_source != "acis":
            self.obs_source = Path(self.obs_source)

    # ── Derived directories ──────────────────────────────────────────────────
    @property
    def date_tag(self) -> str:
        return f"{self.date_start:%Y%m%d}_{self.date_end:%Y%m%d}"

    @property
    def raw_dir(self) -> Path:
        d = self.base_dir / "raw"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def regrid_dir(self) -> Path:
        d = self.base_dir / "regridded"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def obs_dir(self) -> Path:
        d = self.base_dir / "observations"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def corrected_dir(self) -> Path:
        d = self.base_dir / "corrected"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def output_dir(self) -> Path:
        d = self.base_dir / "output"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def log_dir(self) -> Path:
        d = self.base_dir / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d

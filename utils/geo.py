"""Geographic utilities shared across tools."""
from __future__ import annotations
import numpy as np

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def latlon_to_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    return np.column_stack([
        np.cos(lat_r) * np.cos(lon_r),
        np.cos(lat_r) * np.sin(lon_r),
        np.sin(lat_r),
    ])


def km_to_chord(dist_km: float) -> float:
    return 2 * np.sin(dist_km / (2 * EARTH_RADIUS_KM))


def domain_bbox_from_wrf(geo_file) -> dict[str, float]:
    import xarray as xr
    with xr.open_dataset(geo_file) as ds:
        lat_var = "XLAT_M" if "XLAT_M" in ds else "XLAT"
        lon_var = "XLONG_M" if "XLONG_M" in ds else "XLONG"
        lat = ds[lat_var].values.squeeze()
        lon = ds[lon_var].values.squeeze()
    return {"south": float(lat.min()), "north": float(lat.max()),
            "west":  float(lon.min()), "east":  float(lon.max())}

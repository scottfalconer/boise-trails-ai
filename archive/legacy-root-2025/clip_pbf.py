#!/usr/bin/env python
"""Clip an OSM PBF file to the Boise trails region."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import geopandas as gpd


def clip_pbf(
    pbf_path: str,
    trails_path: str,
    out_path: str = "osm_boise_clipped.pbf",
    buffer_km: float = 3.0,
) -> Path:
    """Clip ``pbf_path`` to a bounding box around ``trails_path``.

    Requires the ``osmium" command line tool to be installed.
    The ``trails_path`` GeoJSON is used to compute the area of interest.
    ``buffer_km`` expands the bounding box in all directions.
    """
    gdf = gpd.read_file(trails_path, engine="fiona")
    minx, miny, maxx, maxy = gdf.total_bounds
    deg = buffer_km / 111.32
    minx -= deg
    miny -= deg
    maxx += deg
    maxy += deg
    bbox = f"{minx},{miny},{maxx},{maxy}"

    out_path = Path(out_path).resolve()
    cmd = ["osmium", "extract", "-b", bbox, pbf_path, "-o", str(out_path)]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "osmium command not found; install osmium-tool"
        ) from exc

    size_mb = out_path.stat().st_size / 1e6
    print(f"Clipped OSM saved: {size_mb:.1f} MB -> {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbf", required=True, help="Input OSM PBF file")
    parser.add_argument(
        "--trails", required=True, help="Trail GeoJSON defining area"
    )
    parser.add_argument(
        "--out", default="osm_boise_clipped.pbf", help="Output PBF path"
    )
    parser.add_argument(
        "--buffer_km", type=float, default=3.0, help="Buffer distance in km"
    )
    args = parser.parse_args(argv)

    clip_pbf(args.pbf, args.trails, args.out, args.buffer_km)


if __name__ == "__main__":
    main()

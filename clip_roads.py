#!/usr/bin/env python
"""Clip Idaho OSM data to a small Boise area."""

import argparse
import geopandas as gpd
import numpy as np
from shapely.ops import unary_union
from pyrosm import OSM


def buffered_bbox(geojson_path: str, buffer_km: float = 3.0):
    """Return lon/lat bounds buffered by `buffer_km` from features in `geojson_path`."""
    gdf = gpd.read_file(geojson_path)
    bounds = unary_union(gdf.geometry).bounds
    minx, miny, maxx, maxy = bounds
    avg_lat = (miny + maxy) / 2
    km_per_deg_lon = 111.32 * abs(np.cos(np.radians(avg_lat)))
    km_per_deg_lat = 111.32
    dx = buffer_km / km_per_deg_lon
    dy = buffer_km / km_per_deg_lat
    return (minx - dx, miny - dy, maxx + dx, maxy + dy)


def clip_roads(pbf_path: str, bbox, out_path: str) -> None:
    """Clip the OSM PBF to `bbox` and write as GeoJSON."""
    osm = OSM(pbf_path, bounding_box=bbox)
    roads = osm.get_network(network_type="driving")
    roads.to_file(out_path, driver="GeoJSON")
    print(f"Saved {len(roads):,} road segments \u2192 {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Clip Idaho-latest.osm.pbf to Boise/Eagle/Foothills road network"
    )
    parser.add_argument("--pbf", required=True, help="Path to idaho-latest.osm.pbf")
    parser.add_argument("--trails", required=True, help="Path to Boise trails GeoJSON")
    parser.add_argument("--out", default="boise_roads.geojson", help="Output GeoJSON")
    parser.add_argument(
        "--buffer_km", type=float, default=3.0, help="Buffer beyond trail bbox (km)"
    )
    args = parser.parse_args(argv)

    bbox = buffered_bbox(args.trails, buffer_km=args.buffer_km)
    print("Bounding box:", bbox)
    clip_roads(args.pbf, bbox, args.out)


if __name__ == "__main__":
    main()

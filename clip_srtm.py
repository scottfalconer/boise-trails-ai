#!/usr/bin/env python
"""Download and clip SRTM elevation data for the Boise Foothills."""

import argparse
from pathlib import Path

import geopandas as gpd
from shapely.ops import unary_union
import elevation
import rasterio
from rasterio.mask import mask


def clip_srtm(
    trails_path: str, out_path: str = "srtm_boise_clipped.tif", buffer_km: float = 3.0
) -> Path:
    """Download SRTM tiles touching the buffered trails and clip them."""
    # use Fiona engine for better driver support on some systems
    gdf = gpd.read_file(trails_path, engine="fiona")
    minx, miny, maxx, maxy = gdf.total_bounds
    # Rough degrees per km at mid-latitude
    deg = buffer_km / 111.32
    minx -= deg
    miny -= deg
    maxx += deg
    maxy += deg
    from shapely.geometry import box

    bbox = (minx, miny, maxx, maxy)
    area = box(minx, miny, maxx, maxy)

    out_path = Path(out_path).resolve()
    elevation.clip(bounds=(minx, miny, maxx, maxy), output=str(out_path))

    with rasterio.open(out_path) as src:
        out_img, transform = mask(src, [area.__geo_interface__], crop=True)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "transform": transform,
                "height": out_img.shape[1],
                "width": out_img.shape[2],
                "compress": "lzw",
            }
        )

    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(out_img)

    size_mb = out_path.stat().st_size / 1e6
    print(f"Final DEM saved: {size_mb:.1f} MB -> {out_path}")
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trails", required=True, help="Boise trails GeoJSON")
    parser.add_argument(
        "--out", default="srtm_boise_clipped.tif", help="Output GeoTIFF path"
    )
    parser.add_argument(
        "--buffer_km", type=float, default=3.0, help="Buffer beyond trail bbox"
    )
    args = parser.parse_args(argv)

    clip_srtm(args.trails, args.out, args.buffer_km)


if __name__ == "__main__":
    main()

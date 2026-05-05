# USGS DEM

Pulled: 2026-05-04

Source:

- USGS 1/3 Arc Second DEM GeoTIFF via TNM/S3.
- `https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/historical/n44w116/USGS_13_n44w116_20260407.tif`
- `https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/historical/n44w117/USGS_13_n44w117_20260407.tif`

Files:

- `usgs_13arcsec_boise_planning_bbox.tif` - clipped DEM for the 2026 planning bbox, ignored by git.
- `dem_summary.json` - source URLs, bounds, output dimensions, CRS, and resolution.

Output summary:

- CRS: EPSG:4269
- Bounds: `[-116.432708, 43.475706, -115.907060, 43.852187]`
- Size: 5,677 x 4,066 pixels
- Resolution: approximately 1/3 arc-second
- Elevation range in clipped raster: 761.06 m to 2,311.25 m

Use this for route effort estimates and derived elevation gain. It does not by itself make an official challenge elevation total; that still needs a documented derivation against the official segment geometry.

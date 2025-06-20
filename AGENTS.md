# Repository Guidelines

This project uses two trail data files under `data/traildata/`:

* `trail.json` – the default segment file. It contains **all official Boise Trails Challenge segments**. Most planning scripts and tests rely on this path by default.
* `Boise_Parks_Trails_Open_Data.geojson` – the full Boise parks trail network. This data is helpful for locating connector segments and other reference paths but is **not** required to track official challenge progress.

Keep these roles consistent in any future code or documentation changes so that `trail.json` remains the canonical list of official segments and the open data file continues to serve as supplemental network information.

## Testing

When modifying dataset handling or anything that parses these files, run the test suite to ensure functionality remains correct:

```bash
pytest -q
```


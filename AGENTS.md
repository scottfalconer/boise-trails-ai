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

## Challenge Objectives and Key Metrics

When planning routes for a challenge, the primary goal is to achieve 100% completion of all specified unique segments.

### Target Challenge Statistics (Example - Boise Challenge)
- **Total Target Distance:** ~169.35 miles
- **Total Target Climb:** ~36,000 ft
- **Total Unique Segments:** 247
- **Total Unique Trails:** 100

### Planning Efficiency
While ensuring all segments are covered, the planner should aim to:
- Stay as close as possible to the target distance and elevation, without going significantly under.
- Minimize unnecessary redundant mileage and elevation gain.

### Key Evaluation Metrics
The following metrics are important for evaluating the quality of a generated challenge plan (and are included in the HTML/CSV reports):

- **Progress (Distance/Elevation) %:** Percentage of the target new official distance/elevation covered.
  - `Progress (Distance) % = (Total New Official Trail Distance / Challenge Target Distance) * 100`
  - `Progress (Elevation) % = (Total New Official Trail Elevation Gain / Challenge Target Elevation) * 100`
- **% Over Target (Distance/Elevation):** How much the total on-foot distance/elevation gain exceeds the target.
  - `% Over Target Distance = ((Total On-Foot Distance / Challenge Target Distance) - 1) * 100`
  - `% Over Target Elevation = ((Total Elevation Gain / Challenge Target Elevation) - 1) * 100`
- **Efficiency Score (Distance/Elevation):** Ratio of target to actuals, indicating how efficiently the target was met.
  - `Efficiency Score (Distance) = (Challenge Target Distance / Total On-Foot Distance) * 100`
  - `Efficiency Score (Elevation) = (Challenge Target Elevation / Total Elevation Gain) * 100`
- **Detailed Distance Breakdown:** The plan provides totals for:
  - New and Redundant Official Challenge Trail Distance
  - New and Redundant Connector Trail Distance
  - On-Foot Road Distance
  - Total On-Foot Distance
  - Total Drive Distance & Time

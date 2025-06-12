# Boise Trails Challenge AI Assistant

*An automated route planner to help me plan my route for trail segments in the Boise Trails Challenge.*

The [Boise Trails Challenge](https://boisetrailschallenge.com) is a month-long event where participants attempt to cover **every official trail segment** in the Boise area within a defined period. 

Last year I made it about 75% of the way through the challenge, while running more than the amount of miles in the challenge, ended up with a bunch of hard to get to final segments. This year I wanted to be a bit more strategic. Yes I could have probably spent two hours mapping out a route, but why do that when I could spend two days building this.

## Overview

**boise-trails-ai** takes the list of trail segments required for the challenge and computes a day-by-day plan that ensures 100% coverage of the trails. You specify the challenge start and end dates along with your available time per day, and the planner outputs daily loop routes that fit within your time budget. It is geared toward trail runners and hikers participating in the Boise Trails Challenge, but its parameters (pace, allowed time, etc.) can be adjusted for differentlevels or bikers.

**Key features include:**

* **100% trail coverage** – Every required trail segment is included so nothing is missed. The planner focuses on covering all segments you haven’t completed yet.
* **Efficient routing** – Trail segments are grouped by geographic proximity (using a clustering algorithm) to create logical daily routes. Large clusters are split as needed (via spatial K-Means) to keep each day's route manageable, and redundant out-and-back mileage is minimized.
* **Smart trailhead selection** – Each daily route starts and ends at a convenient point. Known trailheads are preferred; if a cluster lacks a nearby official trailhead, the planner selects a roadside access point near the cluster to start/finish the loop.
* **Route optimization** – Within each cluster of segments, the planner builds a loop or out-and-back path that covers all segments. It first connects segments in a reasonable order (greedily) and then applies a 2-opt optimization to shorten the path and eliminate unnecessary backtracking before returning to the start.
* **Daily scheduling** – Planned routes are scheduled across the challenge timeframe with an eye toward reducing driving between days. Geographically isolated groups of segments are prioritized early in the schedule so that remote areas are completed first, avoiding a situation where only hard-to-reach segments remain at the end.
* **Time and effort estimates** – For each day, the planner calculates total distance, elevation gain, and an estimated moving time based on your provided pace and an adjustment for climb (e.g. adding extra seconds per 100 ft of elevation gain). It also estimates driving time from a home base to the trailhead and between trail clusters if applicable, giving you a realistic sense of the total time commitment per day.
* **Multiple output formats** – The tool produces a CSV summary of the plan, individual GPX track files for each day’s route (plus an optional combined GPX of all routes), and an HTML overview report with interactive maps and elevation profiles for each day.

## How It Works

1. **Clustering by Area:** The planner starts by grouping the trail segments into clusters based on location. This ensures each day’s route covers segments that are near each other. A spatial clustering algorithm (K-Means) is used to split the trail network into sensible daily chunks if necessary, so that routes stay within a reasonable length.
2. **Choosing Start Points:** For each cluster of segments, the planner picks a start/end point that will serve as the day’s trailhead. It prefers official trailheads from a provided list. If no known trailhead is nearby, it will choose a point near the nearest road to access the cluster.
3. **Routing Within a Cluster:** Given a cluster of segments and a start point, the planner computes a route that visits every trail segment in that cluster. It typically creates a loop (or an out-and-back if looping isn’t possible) that returns to the start. The initial order of segment traversal is determined greedily (connecting the closest next segment), then a 2-opt algorithm refines the route to cut out extra distance and backtracking.
4. **Applying Time Constraints:** The length of each route is checked against your available time for that day. The planner uses your base pace (minutes per mile) and an additional per-climb penalty (seconds per 100 ft of elevation gain) to estimate how long you’ll need to run/hike the route. If a projected route is too long to fit in the allotted time, the planner may split the cluster further or flag that day as needing adjustment.
5. **Scheduling the Days:** Once routes are generated for all clusters, the planner assigns each route to a calendar date within your specified challenge period. It sequences the routes to minimize driving distances between consecutive days. Isolated or outlying trail clusters are scheduled earlier in the challenge so you can tackle the remote areas first. The output schedule provides a day-by-day plan from the start date to the end date.
6. **Generating Outputs:** Finally, **boise-trails-ai** writes out the plan results. This includes a CSV file summarizing all days of the challenge, GPX files for each day’s route (for use in GPS devices or apps), and an HTML report for visualization. These outputs contain detailed stats for each day (distance, elevation, time estimates, etc.), and they highlight any days that cannot be completed within the given time constraints so you can adjust your parameters and rerun the planner if needed.

## Installation

Before using the planner, set up the Python environment and dependencies:

1. **Clone the repository** (or download the source).
2. **Install dependencies**. It's recommended to use a Python virtual environment. For example:

   ```bash
   python -m venv .venv  
   source .venv/bin/activate  
   pip install -r requirements.txt  
   ```

   This will install all required libraries as listed in `requirements.txt` (which is generated from `requirements.toml`).

   Alternatively, you can run the provided setup script to create the environment and install dependencies in one step:

   ```bash
   bash run/setup.sh
   ```

## Download Data Assets

Certain data files are required for the planner but are not included in the repository. Run the helper script to fetch these external assets:

```bash
bash run/get_data.sh
```

This currently downloads the latest OpenStreetMap data for Idaho (`idaho-latest.osm.pbf` from Geofabrik) and places it under `data/osm/`. This OSM file is used for road segments (if you allow road connectors in your routes).

**Trail data:** Ensure you have the Boise trail segments data file (`data/traildata/trail.json`) available. This file defines all the trail segments for the challenge and is needed for planning and for processing GPX files. (The repository includes this file, or you can obtain it from the Boise Trails Challenge organizers.)

## Running Tests

If you want to run the test suite (for development purposes), install the runtime dependencies and `pytest`:

```bash
pip install -r requirements.txt
pip install pytest
pytest -q
```

This will execute the unit tests to verify that everything is working properly.

## GPX to CSV Utility

If you have recorded GPS tracks from previous activities or past Boise Trails Challenges, you can use the GPX-to-CSV utility to consolidate your completed segments into a performance log. This is useful for updating the planner with segments you’ve already done.

To convert a folder of GPX files into a consolidated `segment_perf.csv` file, run:

```bash
python -m trail_route_ai.gpx_to_csv --year 2024 --verbose
```

This will scan all GPX activity files in `data/gpx/2024/` and append any completed trail segments to `data/segment_perf.csv`. (By default it processes the current year if `--year` is not provided.) Make sure your GPX files are organized under `data/gpx/<YEAR>/` and that the trail definitions file (`data/traildata/trail.json`) is present so the script can match GPS tracks to trail segment IDs.

Use the `--rebuild` flag if you want to regenerate the entries for that year from scratch (it will delete any existing rows for that year in the CSV before processing). The resulting `segment_perf.csv` will list each segment you completed and can be used by the planner to avoid planning those segments again.

**Arguments for `gpx_to_csv`:**

* `--year YEAR` – Year of GPX files to process (defaults to the current year).
* `--rebuild` – Overwrite any existing entries for the given year instead of appending.
* `--verbose` – Print progress information during processing.

## Using the Challenge Planner

Once your environment is set up and data is in place, you can generate your challenge plan. The planner is run via a command-line interface. At minimum you should specify the start date and end date of the challenge period. You can also provide your typical pace and daily time budget (or use defaults).

For example, to plan a challenge for July 2024 with a 4-hour daily running window and a 10 min/mile base pace (with 30 seconds added per 100 ft of climb), you could run:

```bash
python -m trail_route_ai.challenge_planner --start-date 2024-07-01 --end-date 2024-07-31 \
    --time 4h --pace 10 --grade 30 --year 2024 \
    --dem data/srtm_boise_clipped.tif
```

In this example, the `--dem` option is pointed to a digital elevation model file so the planner can calculate elevation gain for each segment (improving the accuracy of time estimates and enabling elevation profile output). The `--year 2024` flag tells the planner to ignore any segment completions not from 2024, so it plans all segments for the current year’s challenge by default.

After running the planner, check the output directory for results. By default, the planner will produce:

* A summary CSV file (named `challenge_plan.csv`) listing each planned day of the challenge and the segments covered.
* A series of GPX files under the `gpx/` directory (one GPX track for each day’s route).
* An HTML report (named `challenge_plan.html`) providing an interactive overview of all days, including maps and elevation profiles.

## Outputs

**Challenge Plan CSV (`challenge_plan.csv`):** This CSV is the master summary of your plan. Each row corresponds to one day. Key columns include the date, the list of segment IDs or trail names planned that day (the “plan” description), total distance, total elevation gain, estimated moving time, and any driving time or notes. The CSV also breaks down how much of that day’s distance is “unique” (new trail miles) vs. “repeated” (redundant trail miles). If any day’s route cannot be completed in the allotted time (based on your settings), the planner will mark that day with an **"Unable to complete"** note in the plan description so you can adjust your parameters or schedule and try again.

**Daily GPX Files:** For each planned day, a GPX file is generated (e.g. `day_1.gpx`, `day_2.gpx`, etc., or stored in a specified GPX directory). Each GPX track contains the route you need to follow for that day, which you can load onto a GPS watch or mapping application for navigation. The GPX files include waypoints and segmented tracks to highlight any road connector sections (by default, road portions are labeled and waypoints mark where you transition on or off a road).

**Combined GPX:** Optionally, the planner can output a single GPX file that merges all daily routes. This can be useful for an overview of the entire challenge or for loading everything at once into certain mapping tools. (If you want this, you can manually combine the per-day GPX files or use a GPX merging tool, as the planner focuses on per-day files.)

**HTML Overview (`challenge_plan.html`):** Alongside the CSV, the planner creates an HTML file that gives a day-by-day visual summary of the plan. Open this file in a web browser to see an interactive map and elevation profile for each day’s route. This allows you to quickly visualize where each route goes and how tough the elevation gains are, without importing each GPX manually into a map. The HTML report makes it easy to verify that the routes make sense and to spot any days that look especially challenging.

## Optional Features and Customization

The challenge planner has several options to accommodate different preferences and scenarios:

* **Allowing road links between trails:** By default the planner will try to keep you on trails, but you can enable short road connectors to significantly shorten a route. Use the `--roads` option to provide a road network (for example, `--roads data/osm/idaho-latest.osm.pbf` which is the OpenStreetMap file downloaded earlier). When road connectors are enabled, the planner may occasionally link trail segments via nearby roads if it results in a much shorter route. You can control this behavior with `--max-road` (the maximum road distance allowed, default 1.0 mile) and `--road-threshold` (how much faster a road route must be, as a fraction, to justify taking it over an all-trail route; default 0.1, meaning the road option must be >10% faster than staying on trails). Road sections in the output GPX will be clearly marked so you know when you need to travel on a road.
* **Variable daily time budgets:** If your available time differs on certain days (for example, you can run longer on weekends), you can provide a JSON file with per-day hours using `--daily-hours-file`. In this file, specify a mapping from dates to hours available on those dates. Any date not listed will use the default daily time (from `--time`). For instance, you might give yourself 4 hours on most days but only 1.5 hours on a busy day like 2024-07-05. The planner will then plan a shorter route on that day. If a `config/daily_hours.json` file exists, the planner will automatically use it without the need to specify `--daily-hours-file` on the command line.
* **Incorporating completed segments:** The planner can account for trails you have already completed so it doesn’t schedule them again. If you used the GPX-to-CSV utility or otherwise updated `data/segment_perf.csv` with past segment completions, the planner will consider those segments "done" and exclude them from the new plan. You can also explicitly list segments to include via the `--remaining` option (which accepts a comma-separated list of segment IDs or a path to a file containing remaining segment IDs). By default, the planner filters out any segments marked as completed in the current year (or in the year specified by `--year`).
* **Custom output locations:** By default, outputs are saved to the current directory (`challenge_plan.csv`, `challenge_plan.html`, and a `gpx/` folder). You can customize these with `--output` to specify the CSV/HTML filename and `--gpx-dir` to specify a directory for the GPX files. For example, you might use `--output plans/my_plan.csv --gpx-dir plans/gpx` to organize results in a `plans/` folder.
* **Starting location for drive estimates:** The planner estimates driving time from a “home” location to the start of each route (and between routes if multiple are planned in one day). By default, it assumes home is Camel’s Back Park in Boise (coordinates 43.635278° N, -116.205° W). You can set your own home base using `--home-lat` and `--home-lon` to make drive time calculations more accurate for your situation.
* **Configuration file for defaults:** Instead of typing many options each time, you can use a configuration file. If a file named `planner_config.yaml` or `planner_config.json` is present in the `config/` directory, the planner will automatically load it and use those values as defaults. You can also specify a custom config file path via `--config path/to/file.yaml`. This is a convenient way to save your typical settings (dates, pace, etc.). For example, a YAML config might look like:

  ```yaml
  start_date: "2024-07-01"
  end_date: "2024-07-31"
  time: "4h"
  pace: 10
  grade: 30
  gpx_dir: "plans/gpx"
  output: "plans/challenge.csv"
  daily_hours_file: "config/daily_hours.json"
  ```

  With this file in place, you could run the planner with just `python -m trail_route_ai.challenge_planner` and it would use all the values from the config by default. (You can still override any specific option via the command line if needed.) Similarly, if you have a `config/daily_hours.json` file for variable daily hours, it will be picked up automatically without specifying `--daily-hours-file`.

> *Note:* The planner keeps an internal log of segment completion status in `config/segment_tracking.json`. This file is updated each time you run the planner or the GPX-to-CSV utility. If it doesn’t exist on first run, it will be created with all segments marked incomplete. You normally don’t need to edit this file manually – it’s used by the planner to track progress.

## Full Command-Line Reference

Below is a full list of command-line flags available for the challenge planner script (`trail_route_ai.challenge_planner`). (All of these can also be set via a config file as described above, using the same option names.)

* `--config PATH` – Path to a YAML/JSON config file with default options.
* `--start-date YYYY-MM-DD` – Challenge start date.
* `--end-date YYYY-MM-DD` – Challenge end date.
* `--time TIME` – Daily time budget (e.g. `4h` for 4 hours, or a number of minutes).
* `--daily-hours-file PATH` – JSON file mapping specific dates to available hours.
* `--pace FLOAT` – Base running pace in minutes per mile.
* `--grade FLOAT` – Additional seconds per 100 ft of elevation gain (to account for climbing effort; default 0).
* `--segments PATH` – Path to the trail segment definitions file (defaults to `data/traildata/trail.json`).
* `--dem PATH` – Path to a digital elevation model (GeoTIFF) for computing elevation gain (optional but recommended for accurate stats).
* `--roads PATH` – Path to a road network file (GeoJSON or OSM PBF) to enable road connectors (optional).
* `--trailheads PATH` – Path to a trailheads file (JSON or CSV) if you have custom trailhead locations to consider (optional).
* `--home-lat FLOAT` – Home latitude (for drive time calculations).
* `--home-lon FLOAT` – Home longitude.
* `--max-road FLOAT` – Maximum distance (in miles) for any single road connector (default is 1.0 mile).
* `--road-threshold FLOAT` – Fractional speed advantage required for using a road connector (default 0.1, meaning the road route must be >10% faster than staying on trails).
* `--road-pace FLOAT` – Pace on roads in minutes per mile (default is 18, slower than trail running pace to account for walking or cautious travel on roads).
* `--perf PATH` – Path to a CSV of past segment completions (e.g. the `segment_perf.csv` generated by GPX-to-CSV).
* `--year INT` – Filter the completions to consider (e.g. `--year 2024` ignores completions from other years in `segment_perf.csv`). Defaults to the current year.
* `--remaining LIST` – Comma-separated list of segment IDs (or a filename containing segment IDs) to force as the set of remaining segments. If provided, the planner will only plan these segments (overriding the automatic detection of remaining segments).
* `--output PATH` – Output CSV file path for the summary (also determines the name of the HTML report). Default is `challenge_plan.csv` in the current directory.
* `--gpx-dir DIR` – Directory to save the daily GPX files (default is a `gpx/` subdirectory).
* `--no-mark-road-transitions` – If set, do *not* add special markers in GPX files for road-to-trail transitions. (By default, road segments are highlighted with waypoints and separate track segments in the GPX output.)
* `--average-driving-speed-mph FLOAT` – Assumed driving speed in miles per hour for calculating drive times (default is 30 mph).
* `--max-drive-minutes-per-transfer FLOAT` – Maximum allowed driving time between clusters/routes on the same day (default is 30 minutes; the planner will not schedule two routes back-to-back if the drive between them exceeds this).
* `--review` – If this flag is set, the final plan will be output in a format for AI review (for development/debugging purposes).

## Road Connectors

The planner can integrate road sections dynamically using the OpenStreetMap data file. If you specify the Idaho OSM PBF file with the `--roads` option (as shown above), you **do not** need any manual preprocessing to use road connectors. The planner will extract any needed road segments on the fly from `data/osm/idaho-latest.osm.pbf`. This means after running `run/get_data.sh` to download the OSM file, you can simply include `--roads data/osm/idaho-latest.osm.pbf` in your planner command to allow road shortcuts in your routes.

For example:

```bash
python -m trail_route_ai.challenge_planner --start-date 2024-07-01 --end-date 2024-07-31 \
    --time 4h --pace 10 --grade 30 \
    --roads data/osm/idaho-latest.osm.pbf ...
```

Any road connectors that meet your criteria (`--max-road`, `--road-threshold`) will be considered by the planner to shorten routes where appropriate.

## Download SRTM DEM

To incorporate elevation gain data, you'll need to download a Digital Elevation Model covering the Boise trails area. The planner uses NASA SRTM data to calculate elevation profiles and more accurate time estimates (with the `--grade` factor). Use the provided script to obtain and prepare the DEM:

```bash
pip install elevation rasterio geopandas shapely  # install tools for DEM processing
# GDAL utilities are required by the `elevation` package
# On Debian/Ubuntu you may need `apt-get install gdal-bin`
# If a download fails, remove any incomplete tile files and try again:
#   python -c "import elevation; elevation.clean()"

./clip_srtm.py --trails data/traildata/Boise_Parks_Trails_Open_Data.geojson \
    --out data/srtm_boise_clipped.tif --buffer_km 3
```

This script will download the necessary SRTM tiles for the Boise region and then crop (clip) them to a 3 km buffer around the trail network (as defined by the `Boise_Parks_Trails_Open_Data.geojson` file). The output is a trimmed DEM file `data/srtm_boise_clipped.tif` that covers all trails plus a small surrounding area. This file is only a few megabytes and is **git-ignored**, so it remains on your local machine.

**Arguments for `clip_srtm.py`:**

* `--trails PATH` – Path to the Boise trails GeoJSON file (required, this outlines the area of interest for cropping the DEM).
* `--out PATH` – Output path for the GeoTIFF DEM (default is `data/srtm_boise_clipped.tif`).
* `--buffer_km FLOAT` – Buffer distance around the trail network to include in the DEM (default is 3 km).

Once this DEM file is prepared, you can supply it to the planner with the `--dem` option as shown in the usage example. Having the DEM improves the accuracy of elevation gain calculations and allows the HTML report to display elevation profiles for each route.

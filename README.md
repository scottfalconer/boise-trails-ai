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
* **Route optimization** – Within each cluster of segments, the planner builds a loop or out-and-back path that covers all segments. It first connects segments in a reasonable order (greedily) and then applies a 2-opt optimization to shorten the path and eliminate unnecessary backtracking before returning to the start. An optional *advanced optimizer* performs a multi-objective 2‑opt search for even less redundant mileage. The planner automatically Eulerizes disconnected segments when needed to keep loops efficient.
* **Daily scheduling** – Planned routes are scheduled across the challenge timeframe with an eye toward reducing driving between days. Geographically isolated groups of segments are prioritized early in the schedule so that remote areas are completed first, avoiding a situation where only hard-to-reach segments remain at the end.
* **Time and effort estimates** – For each day, the planner calculates total distance, elevation gain, and an estimated moving time based on your provided pace and an adjustment for climb (e.g. adding extra seconds per 100 ft of elevation gain). It also estimates driving time from a home base to the trailhead and between trail clusters if applicable, giving you a realistic sense of the total time commitment per day.
* **Multiple output formats** – The tool produces a CSV summary of the plan, individual GPX track files for each day’s route (plus an optional combined GPX of all routes), and an HTML overview report with interactive maps and elevation profiles for each day.
* **Detailed directions** – HTML outputs now include turn‑by‑turn instructions, labels for drive transitions, per‑part stats, and warnings for long road walks or repeated segments.

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
   # installs tqdm, geopandas, rasterio and other libraries
   ```

   This will install all required libraries as listed in `requirements.txt` (which is generated from `requirements.toml`).

Alternatively, you can run the provided setup script to create the environment and install dependencies in one step. The script now installs
system packages such as GDAL and PROJ via `apt-get` before using `pip` for the Python requirements:

   ```bash
   bash run/setup.sh
   ```
   This installs the system packages as well as all Python requirements,
   including tqdm, geopandas, rasterio, and other dependencies needed for
   running the tests.

Once dependencies are installed, install the project in editable mode so the
`trail_route_ai` package is available on your `PYTHONPATH`:

```bash
pip install -e .
```

This command uses the project's `pyproject.toml` and `setup.py` to install
`boise-trails-ai` in editable mode.

All examples below assume the package has been installed this way.

## Download Data Assets

Certain data files are required for the planner but are not included in the repository. Run the helper script to fetch these external assets:

```bash
bash run/get_data.sh
```

This currently downloads the latest OpenStreetMap data for Idaho (`idaho-latest.osm.pbf` from Geofabrik) and places it under `data/osm/`. This OSM file is used for road segments (if you allow road connectors in your routes).

**Trail data:** Ensure you have the Boise trail segments data file (`data/traildata/trail.json`) available. This file defines all the trail segments for the challenge and is needed for planning and for processing GPX files. (The repository includes this file, or you can obtain it from the Boise Trails Challenge organizers.)

## Running Tests

If you want to run the test suite (for development purposes), install the development requirements which pull in the runtime dependencies as well:

```bash
pip install -r requirements-dev.txt
pytest -q
```

After making changes to the planner itself, re-run `pytest -q` to confirm your
modifications didn’t introduce regressions.

This will execute the unit tests to verify that everything is working properly.

## Cache

The planner caches expensive intermediate data such as shortest path results
under `~/.boise_trails_ai_cache` by default. Cached values are reused across
runs when the same parameters are supplied, greatly speeding up subsequent
executions. Cache load/save events are logged when running with `--verbose`.
You can remove all cached data at any time:

```bash
python -m trail_route_ai.cache_utils --clear
```

Set the `BTAI_CACHE_DIR` environment variable to override the cache location.

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

Run the planner directly with Python. For example, to plan a challenge for July
2024 with a 4‑hour daily running window and a 10 min/mile base pace (with 30
seconds added per 100 ft of climb), run:

```bash
python -m trail_route_ai.challenge_planner --start-date 2024-07-01 --end-date 2024-07-31 \
    --time 4h --pace 10 --grade 30 --year 2024 \
    --dem data/srtm_boise_clipped.tif
```

### Example with road connectors

To cut down on redundant mileage, you can allow short road links between
trailheads and enable the advanced optimizer. The command below keeps each
road connector under half a mile and only uses a road if it saves more than
about 15 % of the time compared to staying on trail:

```bash
python -m trail_route_ai.challenge_planner --start-date 2024-07-01 --end-date 2024-07-31 \
    --time 4h --pace 16 --grade 30 \
    --dem data/srtm_boise_clipped.tif \
    --roads data/osm/idaho-latest.osm.pbf \
    --max-foot-road 0.4 --road-threshold 0.15 \
    --path-back-penalty 1.5 \
    --advanced-optimizer --debug debug --verbose
```

If you have the full Boise trail network data available, you can also provide
`--connector-trails data/traildata/Boise_Parks_Trails_Open_Data.geojson` so the
planner can use additional non-challenge trails to connect official segments
and return to the trailhead more efficiently.

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
Each daily summary also lists **Redundant miles (post-optimization)** showing any duplicated distance remaining after the optimizer runs.

## Optional Features and Customization

The challenge planner has several options to accommodate different preferences and scenarios:

* **Allowing road links between trails:** Connector trails and short road sections are now used by default. Provide a road network with `--roads` (for example, `--roads data/osm/idaho-latest.osm.pbf`) to enable these links. The behavior can be tuned with `--max-foot-road` (maximum road distance allowed while walking, also used as the road limit for the RPP solver, default 3 miles) and `--road-threshold` (fractional speed advantage required to stay on trail, default 0.25). Driving between clusters is governed separately by `--max-drive-minutes-per-transfer`. Road sections in the output GPX will be clearly marked so you know when you are on a road.
* **Variable daily time budgets:** If your available time differs on certain days, you can provide a JSON file with per-day hours using `--daily-hours-file`. In this file, specify a mapping from dates to hours available on those dates. Any date not listed will use the default daily time (from `--time`). For instance, you might give yourself 4 hours on most days but only 1.5 hours on a busy day like 2024-07-05. The planner will then plan a shorter route on that day. If a `config/daily_hours.json` file exists, the planner will automatically use it without the need to specify `--daily-hours-file` on the command line.
* **Incorporating completed segments:** The planner can account for trails you have already completed so it doesn’t schedule them again. If you used the GPX-to-CSV utility or otherwise updated `data/segment_perf.csv` with past segment completions, the planner will consider those segments "done" and exclude them from the new plan. You can also explicitly list segments to include via the `--remaining` option (which accepts a comma-separated list of segment IDs or a path to a file containing remaining segment IDs). By default, the planner filters out any segments marked as completed in the current year (or in the year specified by `--year`).
* **Custom output locations:** By default, outputs are saved to the current directory (`challenge_plan.csv`, `challenge_plan.html`, and a `gpx/` folder). You can customize these with `--output` to specify the CSV/HTML filename and `--gpx-dir` to specify a directory for the GPX files. Use `--output-dir` or `--auto-output-dir` if you want everything for a run placed in its own folder. For example, you might run `--output plans/my_plan.csv --gpx-dir plans/gpx` or simply `--auto-output-dir` to store results in an automatically created folder.
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

> *Note:* The planner reads your progress from `config/segment_tracking.json`.  
> Simply download the **GETAthleteDashboard_v2.json** file from the Boise Trails Challenge website and save it as `config/segment_tracking.json`. The planner parses the `CompletedSegmentIds` list from that file to determine which segments you've finished.

## Mid-Challenge Re-Planning

Need to skip a day, add an extra loop, or mark something complete that wasn’t in the original plan? The planner can re-plan from any date forward while automatically ignoring everything you’ve already finished. All you have to do is update one file and run the planner again.

### 1. Mark what you've finished (`segment_tracking.json`)

Download the official **GETAthleteDashboard_v2.json** file from the challenge website
and save it as `config/segment_tracking.json`. The planner reads the
`CompletedSegmentIds` list in that file to know which segments you've already
completed. Any listed segment is automatically removed from new plans.

### 2. Re-run the planner from today

Pick a new start date (today) and keep the original challenge end date:

```bash
python -m trail_route_ai.challenge_planner \
    --start-date 2025-06-12 \
    --end-date   2025-06-30 \
    --time 4h --pace 10 --grade 30 \
    --output   plans/mid_challenge_plan.csv \
    --gpx-dir  plans/midchallenge_gpx
    --advanced-optimizer
```

The planner automatically ignores every segment marked `"completed": true`. It creates a brand-new schedule for the remaining segments only. Outputs go to the CSV/HTML/GPX paths you specify; your original plan stays untouched.

### 3. Keep your old plans

Outputs are not auto-archived, so:

* Use `--auto-output-dir` to have the planner create a timestamped folder under `outputs/` for each run, or
* Give each run a unique file/dir name manually and move old CSV, HTML and GPX files into an `archive/` folder before the next run.

### What happens under the hood?

The planner loads all trail segments → subtracts the set marked completed → plans only what’s left. No special “resume” flag is needed—the filter runs every time. If you later complete more segments, update the JSON again and re-run; the planner will pick up right where you are.

## Conformance with Planning Objectives

This section details how the Boise Trails Challenge Planner addresses key planning objectives to generate efficient, realistic, and comprehensive trail completion schedules.

### 1. 100% Segment Completion by Challenge Deadline

*   **Goal:** Finish every official trail segment within the specified challenge dates.
*   **Planner Approach:**
    *   The planner loads all segments defined in `--segments` (e.g., `data/traildata/trail.json`) and filters out those already marked as completed (via `config/segment_tracking.json` or `--perf` data).
    *   It attempts to schedule every remaining unique segment into daily activities within the `--start-date` and `--end-date`.
    *   The `main()` function includes a final check: if any `unplanned_macro_clusters` remain after all days are planned, it prints a message indicating that not all segments could be scheduled, typically due to insufficient time budget or too short a duration.
    *   Furthermore, the planner now includes a final validation step after all scheduling attempts: if any required segments remain unscheduled (i.e., are part of the challenge but not found in any day's plan), the planner will abort with an error message listing the missing segment IDs, ensuring that incomplete plans are flagged.
    *   **Configuration:** Success depends on realistic user inputs for `start_date`, `end_date`, and daily time availability (`--time` or `--daily-hours-file`).

### 2. Respect All Event Rules

*   **Goal:** Segments completed in a single activity; required-direction "climb" segments done uphill.
*   **Planner Approach:**
    *   **Single Activity:** Each trail segment is treated as an indivisible part of a larger daily activity route.
    *   **Required Direction:** The `Edge` data structure stores segment `direction` ("ascent", "descent", "both"). Routing functions like `_plan_route_greedy` and `_plan_route_for_sequence` check `e.direction != 'both'` before reversing a segment. RPP and tree traversal methods also respect segment unidirectionality where applicable by the graph construction.

### 3. Maximize Personal Efficiency

#### a. Minimize Redundant Effort (Re-hiking)

*   **Goal:** Avoid re-hiking or re-running the same segment wherever possible.
*   **Planner Approach:**
    *   **Route Optimization:** `plan_route()` uses either RPP (`plan_route_rpp`) or a greedy approach (`_plan_route_greedy`) followed by 2-opt optimization to create efficient loops. These methods inherently try to cover required segments with minimal path length.
    *   **Specific Redundancy Reduction:** If `redundancy_threshold` (PlannerConfig) is set, `planner_utils.optimize_route_for_redundancy` is called to further refine routes.
    *   **Advanced Optimizer:** `use_advanced_optimizer = True` (PlannerConfig) enables `optimizer.advanced_2opt_optimization` for potentially lower redundancy.
    *   **Reporting:** CSV and HTML outputs include "unique" vs. "redundant" mileage and elevation gain, with percentages.
    *   **Configuration:**
        *   `redundancy_threshold`: Lower values (e.g., 0.1) target stricter redundancy control.
        *   `use_advanced_optimizer = True`.
        *   `path_back_penalty` (in `_plan_route_greedy` / `--path-back-penalty` CLI option): Influences how return paths to loop starts are chosen, discouraging segment reuse.

#### b. Minimize Drive Overhead

*   **Goal:** Group nearby trails; avoid "micro-drives" unless clearly beneficial.
*   **Planner Approach:**
    *   **Clustering:** `identify_macro_clusters` and `cluster_segments` group trails geographically.
    *   **Daily Scheduling Logic (in `main()`):**
        *   Prioritizes adding more activities to the current day if budget allows, using `last_activity_end_coord` as the origin for drive time estimation to the next potential cluster.
        *   Uses a set of constants to decide if walking a connection is better than driving: `DRIVE_FASTER_FACTOR`, `MIN_DRIVE_TIME_SAVINGS_MIN`, `MIN_DRIVE_DISTANCE_MI`, `DRIVE_PARKING_OVERHEAD_MIN`, and `COMPLETE_SEGMENT_BONUS`.
        *   Sorts candidate clusters for a day first by `drive_time` from current location (home or last activity).
    *   **Configuration:**
        *   `max_drive_minutes_per_transfer`: Limits drive time between activities *within* the same day.
        *   The global constants mentioned above can be tuned based on user preference for driving vs. extra walking.

#### c. Minimize Extra Climb

*   **Goal:** Cut needless repeated ascents/descents by choosing smarter loop routes or spur sequencing.
*   **Planner Approach:**
    *   **Time Estimation with Grade Penalty:** The primary mechanism is the `grade` parameter (PlannerConfig), which adds a time penalty for elevation gain (`planner_utils.estimate_time`). Routes with more climb become "longer" in estimated time. Pathfinding algorithms then naturally prefer less climb if it results in a faster overall route.
    *   **Loop Preference:** Loop routes generated by `plan_route` help avoid unnecessary out-and-backs on hilly terrain if a loop is more efficient time-wise.
    *   **Spur Logic:** `spur_length_thresh` and `spur_road_bonus` (PlannerConfig) in `_plan_route_greedy` provide slightly more flexible road connection allowances when exiting short spurs, which can prevent inefficient detours solely to avoid minor road use after a spur.
    *   **Configuration:**
        *   `grade`: Increase to make climbs more "costly" in time estimations, further discouraging them if alternatives exist.

### 4. Realistic, Self-Contained Outings

#### a. Start/End at Legal Trailheads/Access Points

*   **Goal:** Each activity starts/ends at a legal trailhead or clearly identified access point.
*   **Planner Approach:**
    *   `ClusterInfo` objects store `start_candidates`. These are populated using:
        *   Official trailheads from `--trailheads` file (`trailhead_lookup`).
        *   `AccessFrom` tags in segment data (`access_coord_lookup`).
        *   If no official points, nodes near roads are considered.
        *   As a last resort, the closest node in the cluster to the cluster's centroid.

#### b. Prefer Loop/Lollipop Routes

*   **Goal:** Prefer loop or lollipop routes; use out-and-back only when no loop is feasible.
*   **Planner Approach:**
    *   `_plan_route_greedy` explicitly paths back to the start.
    *   `plan_route_rpp` inherently generates circuits (loops).
    *   `_plan_route_tree` (for tree-like sections) effectively creates a route covering all segments and returning to start (an out-and-back covering of the tree).
    *   The overall `plan_route` orchestrates these to prioritize complete, returning routes.

#### c. Coherent Cluster Per Day, Limited Intra-day Drives

*   **Goal:** Keep one coherent cluster per day when possible; limit intra-day drives.
*   **Planner Approach:**
    *   The main loop in `main()` attempts to fill the current day's budget using segments from a chosen cluster or additional nearby clusters if they fit.
    *   A drive is only added if the next cluster is not walkable (per drive vs. walk logic) and the drive time is within `max_drive_minutes_per_transfer`.
    *   The selection of the *first* cluster for the day also considers drive time from home.
    *   **Configuration:** `max_drive_minutes_per_transfer`.

### 5. Continuous Progress & Risk Management

#### a. Clear Entire Geographic Areas Before Moving On

*   **Goal:** Avoid large regions of unfinished segments late in the schedule.
*   **Planner Approach:**
    *   **Isolation Score:** In `main()`, an `isolation_lookup` score is calculated for each cluster (distance to nearest other cluster). When selecting the next cluster for a day (especially the first cluster when starting from home), the planner gives preference to more isolated clusters (higher score), all else being equal. This helps tackle remote areas earlier.
    *   The `smooth_daily_plans` function can also fill in gaps in earlier days with smaller remaining clusters, which might help consolidate progress in certain areas.

#### b. Fallback Resilience: Split Unroutable Clusters

*   **Goal:** If a cluster cannot be routed continuously on foot, automatically split into sub-loops with a single drive between them.
*   **Planner Approach:**
    *   In `main()`, if `plan_route` fails for an initial macro-cluster, `split_cluster_by_connectivity(cluster_segs, G, args.max_foot_road)` is called. This function breaks the cluster into sub-clusters based on whether their segments can be connected with less than `args.max_foot_road` miles of road.
    *   These smaller sub-clusters are then added back to the pool of `unplanned_macro_clusters` to be scheduled individually.
    *   The main daily planning loop can then schedule these sub-clusters on the same day (with a drive if needed and if it fits the budget) or on different days.
    *   **Reporting:** The `debug` log notes when such splits occur. The user-facing CSV/HTML `rationale` will indicate "Includes drive transfers between trail groups" if a drive is inserted.

#### c. Fallback Resilience: RPP to Greedy

*   **Goal:** If advanced route optimization (RPP) fails, fall back to the greedy method.
*   **Planner Approach:**
    *   In `plan_route()`, if `use_rpp = True`, `plan_route_rpp` is attempted first.
    *   If `plan_route_rpp` does not return a valid route (e.g., it's empty due to timeout or failure), the code explicitly falls back to using `_plan_route_greedy` followed by 2-opt refinement.
    *   `debug_log` messages track whether RPP was successful or if a fallback occurred.
    *   **Configuration:** `rpp_timeout` controls how long RPP attempts before potentially failing.

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
* `--connector-trails PATH` – Supplemental trail network GeoJSON for connector segments (optional; no default file is assumed).
* `--dem PATH` – Path to a digital elevation model (GeoTIFF) for computing elevation gain (optional but recommended for accurate stats).
* `--roads PATH` – Path to a road network file (GeoJSON or OSM PBF) to enable road connectors (optional).
* `--trailheads PATH` – Path to a trailheads file (JSON or CSV) if you have custom trailhead locations to consider (optional).
* `--home-lat FLOAT` – Home latitude (for drive time calculations).
* `--home-lon FLOAT` – Home longitude.
* `--max-foot-road FLOAT` – Maximum distance (in miles) for any single road connector. This limit is also used by the RPP solver (default is 3 miles).
* `--road-threshold FLOAT` – Fractional speed advantage required for using a road connector (default 0.25).
* `--road-pace FLOAT` – Pace on roads in minutes per mile (default is 12, assuming faster road running).
* `--perf PATH` – Path to a CSV of past segment completions (e.g. the `segment_perf.csv` generated by GPX-to-CSV).
* `--year INT` – Filter the completions to consider (e.g. `--year 2024` ignores completions from other years in `segment_perf.csv`). Defaults to the current year.
* `--remaining LIST` – Comma-separated list of segment IDs (or a filename containing segment IDs) to force as the set of remaining segments. If provided, the planner will only plan these segments (overriding the automatic detection of remaining segments).
* `--output PATH` – Output CSV file path for the summary (also determines the name of the HTML report). Default is `challenge_plan.csv` in the current directory.
* `--gpx-dir DIR` – Directory to save the daily GPX files (default is a `gpx/` subdirectory).
* `--output-dir DIR` – Directory to store all outputs for this run.
* `--auto-output-dir` – Automatically create a timestamped directory under `outputs/` when `--output-dir` is not supplied.
* `--no-mark-road-transitions` – If set, do *not* add special markers in GPX files for road-to-trail transitions. (By default, road segments are highlighted with waypoints and separate track segments in the GPX output.)
* `--average-driving-speed-mph FLOAT` – Assumed driving speed in miles per hour for calculating drive times (default is 30 mph).
* `--max-drive-minutes-per-transfer FLOAT` – Maximum allowed driving time between clusters/routes on the same day (default is 30 minutes; the planner will not schedule two routes back-to-back if the drive between them exceeds this).
* `--review` – If this flag is set, the final plan will be output in a format for AI review (for development/debugging purposes).
* `--debug PATH` – Write per-day route rationale to the given file.
* `--verbose` – Echo debug log messages to the console.
* `--advanced-optimizer` – Enable the experimental multi-objective 2‑opt optimizer for reduced redundancy.
* `--optimizer greedy2opt` – Select the built-in optimizer. This mode uses internal Eulerization and 2‑opt refinement.
* `--draft-daily` – Write draft CSV/HTML outputs after each day in a `draft_plans/` folder.
* `--strict-max-foot-road` – Do not walk connectors longer than `--max-foot-road` (split the route instead).
* `--draft-every N` – Write draft outputs every `N` days instead of only at the end.
* `--first-day-seg ID` – Start the first day from the specified segment ID.
* `--force-recompute-apsp` – Rebuild the All-Pairs Shortest Paths cache.
* `--num-apsp-workers INT` – Number of worker processes for APSP pre-computation.
* `--focus-segment-ids LIST` – Comma-separated list of segment IDs for focused planning.
* `--focus-plan-days INT` – Number of days to plan when using `--focus-segment-ids`.
* `--spur-length-thresh FLOAT` – Trail length in miles below which spur detours are considered.
* `--spur-road-bonus FLOAT` – Additional road miles allowed when exiting a short spur.
* `--redundancy-threshold FLOAT` – Maximum acceptable redundant distance ratio.
* `--no-connector-trails` – Disallow using non-challenge trail connectors.
* `--rpp-timeout FLOAT` – Time limit in seconds for the RPP solver.

## Road Connectors

The planner can integrate road sections dynamically using the OpenStreetMap data file. If you specify the Idaho OSM PBF file with the `--roads` option (as shown above), you **do not** need any manual preprocessing to use road connectors. The planner will extract any needed road segments on the fly from `data/osm/idaho-latest.osm.pbf`. This means after running `run/get_data.sh` to download the OSM file, you can simply include `--roads data/osm/idaho-latest.osm.pbf` in your planner command to allow road shortcuts in your routes.

For example:

```bash
python -m trail_route_ai.challenge_planner --start-date 2024-07-01 --end-date 2024-07-31 \
    --time 4h --pace 10 --grade 30 \
    --roads data/osm/idaho-latest.osm.pbf ...
```

Any road connectors that meet your criteria (`--max-foot-road`, `--road-threshold`) will be considered by the planner to shorten routes where appropriate.

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

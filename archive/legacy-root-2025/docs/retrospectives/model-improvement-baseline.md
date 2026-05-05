# Model Improvement Baseline for the Next Boise Trails Run

Date created: 2026-05-03

This baseline exists to compare future model/tooling performance against the approximate state of last year's Boise Trails planner work.

## What Last Year's Agent Work Did Well

### It found the right formal problem family

The project recognized the challenge as a blend of rural postman, capacitated arc routing, mixed directed routing, and asymmetric elevation-aware costs. That is substantially better than treating the challenge as a simple list of waypoints.

### It produced multiple candidate architectures

The repo contains evidence of three useful framings:

- Continuous CPP/TSP route for topology analysis.
- VRP/CARP daily planner for capacity-constrained routing.
- Trailhead-based planner for human-realistic hiking loops.

The final architectural direction should keep the third framing and borrow graph algorithms from the first.

### It used actual historical activity data

The 2024 GPX and `segment_perf.csv` artifacts are valuable because they show how real hikes group segments. They reveal natural systems like Military Reserve, Table Rock, Three Bears, Dry Creek, Polecat, Seaman Gulch, and Cartwright.

### It captured good route-quality goals

The docs define meaningful output metrics:

- 100% official segment completion.
- Official distance and elevation progress.
- Redundant official distance.
- Connector trail distance.
- Road distance.
- Total on-foot distance.
- Drive distance and time.
- Loop practicality.

These are the right evaluation categories.

## What Last Year's Agent Work Missed

### Data-schema verification

The model did not sufficiently verify that loader assumptions matched the live GeoJSON schema. It used old fields like `TRAIL_NAME` and `Shape_Length` against a file that uses `TrailName`, `Name`, `TrailMiles`, and `TrailSubSystem`.

Improvement test:

- A new model should inspect representative records before writing route logic.
- It should flag schema mismatch before claiming connector-trail routing works.

### Runnable state preservation

The working tree ended with tests targeting deleted modules.

Improvement test:

- A new model should preserve or explicitly migrate tests during architecture pivots.
- It should leave a small passing core test suite even if advanced tests are deferred.

### Separation of concerns

The planners mixed data loading, graph construction, optimization, decoding, reporting, and route-quality accounting.

Improvement test:

- A new model should create or preserve clear modules for:
  - Data normalization.
  - Segment matching.
  - Graph building.
  - Local loop generation.
  - Daily scheduling.
  - Reporting and validation.

### Human-route realism

The VRP output had 39 hikes, 337.37 on-foot miles, and 111.16 between-hike drive miles. The trailhead output reduced mileage but still had 64 hikes and 28 trailheads.

Improvement test:

- A new model should optimize for coherent trailhead outings, not only mathematical coverage.
- The target should be closer to 25-35 hikes, mostly 1-2 outings per day, with bounded trailhead churn.

### Connector trail use

The trailhead architecture depended on connector trails but generated `connector_miles: 0`.

Improvement test:

- A new model should show real connector trail use, with names and distances.
- It should distinguish official challenge segments, connector trails, virtual graph healing, and roads.

### Prior-year completion reconstruction

The local repo does not contain the final user-reported prior-year completion of 41.82% and 68.90 miles.

Improvement test:

- A new model should explicitly reconcile:
  - Official dashboard data.
  - Strava activities.
  - Local GPX files.
  - Segment matching against official challenge segments.
- It should report which source is authoritative for each metric.

## Suggested Model Comparison Rubric

Score each future model/run from 0 to 3 in each category.

| Category | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| Data understanding | Assumes schema | Reads files but misses mismatches | Finds schema issues | Normalizes and tests schema |
| Historical reconstruction | Ignores prior data | Uses stale local snapshots | Reconciles some sources | Reconciles Strava/dashboard/GPX with caveats |
| Architecture | Monolithic script | Partial modularity | Clear stages | Clear stages with validations between them |
| Route realism | Segment delivery | Some clustering | Trailhead/system loops | Trailhead/system loops with logistics |
| Connector handling | No connectors | Virtual connectors only | Real connectors parsed | Real connectors optimized and reported |
| Direction/elevation | Ignored | Partially modeled | Modeled in graph | Validated in output |
| Test hygiene | Broken tests | Some smoke tests | Core suite passes | Core plus metric regression tests |
| Reporting | Raw GPX/CSV only | Basic summary | Metrics and caveats | Actionable report with audit trail |

## Baseline Metrics to Beat

These are not target final goals. They are the observed state to improve on.

### VRP daily planner artifact

- Hikes: 39
- Days: 17
- On-foot miles: 337.37
- Between-hike drive miles: 111.16
- Segment-name mentions: 322 for 247 unique segment names

### Trailhead-based planner artifact

- Hikes: 64
- Days: 13
- Total miles: 287.5
- Redundancy: 44.3%
- Unique trailheads: 28
- Average hikes per day: 4.9
- Real connector trail miles reported: 0
- Required duplicates in detailed output: 34

### Historical activity artifacts

- 2024 GPX files: 24
- 2024 GPX total track distance: 157.75 miles
- `segment_perf.csv` activities: 24
- `segment_perf.csv` unique segment IDs: 145

## Better Next-Year Workflow

1. Build an immutable data audit report first.
2. Refresh official challenge data and completion data.
3. Normalize open trail GeoJSON into a tested connector schema.
4. Reconstruct prior-year completion from authoritative sources.
5. Generate route candidates per trail system.
6. Score candidates by official coverage, redundancy, elevation, and trailhead logistics.
7. Schedule candidates across available days.
8. Produce GPX, CSV, and a human report.
9. Run metric tests before calling the plan usable.

## Key Question for Model Improvement

The main question is not whether a newer model can write more code faster. The useful question is whether it can avoid last year's failure modes:

- Does it verify assumptions against data?
- Does it keep the repo runnable during a pivot?
- Does it choose the right abstraction for the user workflow?
- Does it expose uncertainty instead of hiding it behind generated artifacts?
- Does it produce a plan that looks like how a person would actually complete the challenge?


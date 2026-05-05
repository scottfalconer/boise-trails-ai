# Boise Trails Planner 2025 In-Situ Retrospective

Date of retrospective: 2026-05-03

Purpose: document the approximate state of the Boise Trails Challenge planner where last year's work left off, so future route-planning runs can be compared against a stable baseline instead of relying on chat memory.

## Executive Summary

The 2025 planner work had strong problem framing but ended in an unfinished transition between architectures.

The best conceptual move was recognizing that the challenge is not a generic point-delivery problem. The useful model is closer to: build natural trail-system loops from real trailheads, then schedule those loops across the challenge window. The weaker implementation path was trying to force individual official trail segments through a generic VRP/CARP solver, which produced mathematically plausible but human-unfriendly hiking plans.

The current workspace is not cleanly runnable. The git branch has a committed older planner lineage, while the working tree has newer untracked and modified files. The local test suite does not currently collect because many tests still import deleted modules such as `trail_route_ai.challenge_planner` and `trail_route_ai.planner_utils`.

The most important technical lesson is that data normalization was underbuilt. The newer planners load the open trail GeoJSON using old or incorrect field names, so connector trails appear with weak names and zero lengths. That likely explains why the trailhead-based router never actually used real connector trails, despite the architecture depending on them.

## Known Completion Context

User-reported final result for the prior challenge:

- Completion: 41.82%
- Official distance: 68.90 miles
- Qualitative note: lower than the previous year

This final result is not reproducible from the local repository alone. The local `GETAthleteDashboard_v2.json` snapshot only shows 11 completed segment IDs and 7.5682% progress, so it is stale or partial. The repo does contain richer historical activity artifacts for 2024 under `data/results/2024/` and `data/segment_perf.csv`.

## Problem Framing We Had

The project correctly identified the formal routing problem as a mix of:

- Rural Postman Problem: only official challenge segments must count.
- Capacitated Arc Routing Problem: routes need to be split across days.
- Mixed Chinese Postman Problem: some segments have direction requirements.
- Windy Postman Problem: uphill and downhill costs differ.
- Trailhead logistics: every practical hike should return to the parked car.

That framing is still useful. The problem is that the implementation alternated between global mathematical solvers and hiking-practical heuristics without fully reconciling the two.

## Architecture Timeline

### 1. Continuous Route Planner

File: `src/trail_route_ai/continuous_route_planner.py`

Intent: create one massive continuous route covering all official segments.

Approach:

- Load official challenge segments.
- Split disconnected trail systems into graph components.
- Use a Chinese Postman-style solver inside components.
- Use a TSP heuristic to order disconnected components by driving distance.

Assessment:

- This was the cleanest graph-theory implementation.
- It was useful for understanding topology and unavoidable duplication.
- It was not the right user experience for a month-long challenge, because it optimized a single continuous effort rather than practical daily loops.

### 2. Daily VRP / CARP Planner

File: `src/trail_route_ai/daily_planner.py`

Intent: use OR-Tools to split all required arcs into day-sized routes.

Approach:

- Build a directed graph from official trails and optional trails.
- Treat each required directed arc as a VRP location.
- Use configured trailheads as depots.
- Use vehicle capacities as day mileage limits.
- Decode the solver output back into hikes, splitting hikes when connectors exceed a drive threshold.

Assessment:

- Good formal ambition.
- Bad abstraction for real hiking.
- The solver treated trail segments as deliveries rather than parts of trail systems.
- Capacity was based on required arc demand, not the full reconstructed hike including connectors and return-to-car legs.
- The generated output covered all segment names but repeated several of them and produced excessive mileage.

Observed generated output:

- 39 hikes over 17 days.
- 337.37 on-foot miles.
- 111.16 between-hike drive miles.
- 247 unique segment names in the summary, with 322 segment-name mentions.
- Several repeated segments, especially in Military Reserve and Central Ridge systems.

### 3. Trailhead-Based Router

File: `src/trail_route_ai/trailhead_router.py`

Intent: mimic how people actually hike the challenge: park at a trailhead, complete natural loops, then schedule those loops.

Approach:

- Discover or infer trailheads.
- Group accessible official segments by trail-system naming.
- Combine related families.
- Try to add connector trails.
- Attempt CPP-like traversal inside each family.
- Greedily select loops to cover all required segments.
- Bin-pack selected loops into days.

Assessment:

- This is directionally the right architecture.
- The implementation is still heuristic and internally inconsistent.
- It uses virtual road connections rather than a real OSM road graph.
- It reports 0 connector miles, even though connector trails are core to the design.
- It creates too many hikes and too many trailheads.
- Its generated summary says all 247 required IDs are covered, but with 34 duplicate required segment mentions.

Observed generated output:

- 64 hikes over 13 days.
- 287.5 total miles.
- 44.3% redundancy.
- 28 unique trailheads.
- 4.9 average hikes per day.
- Day 12 has 29 hikes.
- Only non-required route segments visible in the detailed plan are `Road Connection`.

## Major Findings

### The data loader broke the connector layer

The open trail GeoJSON uses fields such as:

- `TrailName`
- `Name`
- `TrailMiles`
- `TrailSubSystem`
- `TrlSurface`
- `AccessFrom`
- `AccessTo`

The newer planners look for fields such as:

- `TRAIL_NAME`
- `Shape_Length`
- `CART_ID`
- `Surface`
- `Exposure`

In the local GeoJSON audit:

- 334 features exist.
- 0 features have `TRAIL_NAME`.
- 0 features have `Shape_Length`.
- 334 features have `TrailMiles`.
- 245 features have `TrailSubSystem`.
- 243 features have `AccessFrom`.

This means optional connector trails were effectively loaded with placeholder names and zero length. The planner architecture relied on connector trails, but the implementation did not have a valid connector dataset.

### The VRP abstraction was too low-level

The VRP approach optimized over individual segment arcs. That created routes that were technically connected through graph paths but not necessarily good hikes. It also did not naturally capture trailhead locality, named trail systems, or the human preference for one or two coherent outings per day.

### The trailhead approach was correct but unfinished

The trailhead router made the right product-level shift: cluster by hiking systems first, solve local loops second, then schedule. But it still had:

- Weak data normalization.
- Virtual roads instead of real roads.
- No robust trail connector graph.
- Simple bin packing instead of logistics-aware daily scheduling.
- Heuristic distance accounting.
- Generated plans with implausible trailhead sprawl.

### Test coverage did not protect the transition

The repo has tests for the older committed planner and newer design goals, but the working tree deletes the modules many tests import. `pytest -q` currently fails during collection with 17 errors.

This is important for model-comparison purposes: last year's agent/code state included broad ambition and many tests, but the implementation was not preserved in a runnable, internally consistent state.

## Where We Left It

The honest state is:

- Problem formulation: strong.
- Historical activity artifacts: useful but incomplete for 2025 final status.
- Continuous-route solver: useful for topology analysis, not practical planning.
- VRP daily planner: functional artifact, but wrong abstraction.
- Trailhead router: right architecture, incomplete implementation.
- Data normalization: key blocker.
- Tests: broken in current working tree.
- Strava/dashboard integration: not complete locally.

## Recommended Starting Point for This Year's Work

Before generating any new plan:

1. Reconstruct actual prior-year completion from Strava and/or the official dashboard.
2. Normalize all input data into a single schema.
3. Fix connector trail parsing before touching optimization.
4. Restore a passing minimal test suite around data loading, segment matching, and route metrics.
5. Build trail-system route candidates first.
6. Use scheduling after route candidates exist, not as the primary route constructor.

## Comparison Baseline

A new model/tooling run should be judged against this baseline:

- Does it detect the GeoJSON schema mismatch without being told?
- Does it preserve runnable tests while refactoring?
- Does it separate data normalization, graph construction, local route generation, and scheduling?
- Does it use actual prior-year completion data instead of stale dashboard snapshots?
- Does it produce fewer, more natural hikes than the old VRP output?
- Does it explicitly report uncertainty where data is missing?


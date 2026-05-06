# Boise Trails Challenge Home-to-Home Mathematical Proof Spec v0.1

**Date:** 2026-05-06  
**Status:** Accepted policy formalized; proof instance ready for private home/bounds inputs and graph build.  
**Primary shift:** the incumbent is no longer just a route menu. The proof target is a **home-to-home completion schedule**.

## Accepted policy

> A valid Boise Trails Challenge completion plan is a set of home-to-home field days. Each field day starts at home with one car, drives to one or more legal parking starts, completes one or more legal single-car run loops, optionally drives between parked starts, and returns home. Every run loop must start and end at the same legal parked car, use only legal runnable graph edges, obey official trail direction rules, and have continuous GPX. A schedule is feasible only if it covers every required official segment and every field day stays within the user’s p90 personal daily bounds. The objective is to minimize total p75 home-to-home completion time, with daily stress, p90 risk, grade-adjusted miles, on-foot miles, field-day count, and parking risk used as tie-breakers.

## Data freeze used for v0.1

| Source | Role | Count / key fact | SHA-256 |
|---|---|---:|---|
| `official_foot_segments.geojson` | Official target segment set T | 251 segments / 164.43452 mi | `4078e8fbc829f0aaa9c031b270b445209b6fd648b1fdb3680778e07ab31ee871` |
| `trails.json` | Full BoiseTrails source | 274 segments / 183.31739 mi / updated 2026-05-01T19:14:44 | `4b2becb43ae2965fd0762b10ebf13e890a77271027dd00f9ccc90b52b172e978` |
| `official_foot_master_trails.json` | Foot/both master trails | 101 master trails | `8ac205cf357f0cd1ce94dfa930689fb5698a2d44fe0c0ee74a22e41b172c5393` |
| `strava-parking-anchors-v1.geojson` | Candidate parking starts P seed | 31 anchors | `9df9e548bbd02ce648ce379335110e2448da8dfad848d79b119e460e10a1f68a` |
| `boise_parks_trails_open_data.geojson` | Access/status cross-check | 340 features | `47d7924be26dcc6284a9439720db52d879039e600730caaaf48ad901ad5b4da0` |
| `boise_planning_bbox.osm.pbf` | Drive graph + road-running connector source | frozen PBF | `67fb73cae81290ce29c2de61c1ee473f09e64beb8142509047cc378002c08603` |
| `route-efficiency-audit-2026-05-06.json` | Current route-menu incumbent | verdict=proven; achieved=True | `9de762e19a9870f8d21e08131912fa7295cfd8dafb3e03975b44345f2daea2cf` |

The official target set contains 232 both-use and 19 foot-only segments. Direction rules include 228 bidirectional segments and 23 ascent-directed segments.

## Formal objects

Let:

- **H** = the private home vertex. This is required but not published.
- **P** = legal parking/start vertices.
- **T** = required official foot segment set.
- **G_run = (V_run, E_run)** = all legal runnable trails, public roads/sidewalks/shoulders, public paths, and validated connectors.
- **G_drive = (V_drive, E_drive)** = legal drivable graph connecting H and P.
- **B** = personal daily p90 bounds.
- **M** = p75/p90 time and DEM effort model.

## Legal running edge definition

An edge is allowed in **G_run** only if it is one of:

1. official trail open to foot traffic,
2. public road, sidewalk, or shoulder where pedestrian running is legal,
3. public access path,
4. field-validated connector,
5. explicit roadside parking access.

An edge is rejected if it is private, no-foot/no-pedestrian, a fake straight-line map gap, an inferred shortcut without public tread/access, a graph-only route without continuous GPX, or a known closure at proof time.

## Run loop constraint

A run loop `r` is feasible only if:

```text
start_run(r) = end_run(r) = p, where p ∈ P
```

Every run loop must:

- use only edges in `G_run`,
- return to the same parked car,
- obey official direction rules,
- pass GPX continuity checks,
- expose p75/p90 time and DEM effort metrics,
- report geometry-level official segment coverage.

## Field day constraint

A field day is a home-to-home sequence:

```text
H -> p1 -> run_loop_1 -> p2 -> run_loop_2 -> ... -> pk -> run_loop_k -> H
```

Driving is allowed only between home and parking vertices, or between parking vertices after the previous run loop has returned to the car. The car may not move inside a run loop.

Field-day p75 cost is:

```text
drive_p75(H, p1)
+ setup_transition_p75(p1)
+ run_loop_p75(r1)
+ drive_p75(p1, p2)
+ setup_transition_p75(p2)
+ run_loop_p75(r2)
+ ...
+ drive_p75(pk, H)
```

## Daily feasibility bounds

A field day is selectable only if it satisfies every active p90 bound:

```text
daily_on_foot_miles            <= MAX_DAILY_ON_FOOT_MILES
daily_grade_adjusted_miles     <= MAX_DAILY_GRADE_ADJUSTED_MILES
daily_ascent_ft                <= MAX_DAILY_ASCENT_FT
daily_moving_p90_minutes       <= MAX_DAILY_MOVING_P90_MINUTES
daily_door_to_door_p90_minutes <= MAX_DAILY_DOOR_TO_DOOR_P90_MINUTES
daily_parking_starts           <= MAX_PARKING_STARTS_PER_DAY
daily_run_loops                <= MAX_RUN_LOOPS_PER_DAY
```

Use **p90 for feasibility** and **p75 for optimization**.

## Coverage constraint

A schedule `S` is complete only if:

```text
for every official segment s ∈ T:
    covered_length(S, s) >= required_length(s)
```

The proof checker must verify both segment IDs and geometry-level coverage, and must honor direction where `direction != both`.

## Objective

The optimization is lexicographic:

1. minimize `total_p75_home_to_home_minutes`,
2. minimize `max_daily_stress_ratio`,
3. minimize `total_p90_home_to_home_minutes`,
4. minimize `total_grade_adjusted_miles`,
5. minimize `total_on_foot_miles`,
6. minimize `field_day_count`,
7. minimize `total_parking_risk_score`.

The old on-foot/official ratio remains a diagnostic, not the primary objective.

## Exact proof theorem

Given frozen `{T, G_run, G_drive, P, H, B, M}`, the selected schedule `S*` is proven optimal when:

```text
S* ∈ FeasibleSchedules(T, G_run, G_drive, P, H, B, M)

and for every feasible schedule S:
    Objective(S*) <=lex Objective(S)
```

The solver certificate must include a feasible incumbent upper bound, a global lower bound over the declared universe, zero primary-objective gap, and either a lexicographic tie-break certificate or a no-dominating-schedule certificate.

## What v0.1 creates

This pass creates:

- `boise-home-schedule-proof-instance-v0.1.json` — machine-readable policy, sets, objective, source hashes, incumbent summary, and required missing private inputs.
- `boise-home-schedule-proof-instance-v0.1.schema.json` — lightweight JSON schema for the instance file.
- `boise_home_schedule_proof_checker.py` — independent checker skeleton that verifies the instance hash manifest and, when a schedule is supplied, validates home-to-home days, single-car loops, p90 bounds, coverage, and objective totals.

## Required private/user-specific inputs before optimization

These are intentionally not guessed:

- private home vertex or private home profile,
- daily p90 personal bounds,
- legal parking ledger derived from the 31 parking anchors plus access review,
- connector ledger classifying every road/path/gap/connector as allowed, rejected, or field-required,
- home-to-parking and parking-to-parking drive p75/p90 matrix,
- legal run graph built from official trails, OSM/public roads, public paths, and accepted connectors.

## Incumbent imported from current route-menu audit

The existing audit remains the incumbent route menu:

- official miles: 164.41
- on-foot miles: 268.2
- ratio: 1.631
- runnable component count: 26
- covered official segments: 251
- manual holds: 0

The home-schedule optimizer is allowed to beat this incumbent by combining run loops into fewer/more different home-to-home field days, driving between trailheads inside a day, or choosing different legal starts/connectors, as long as every hard constraint above is satisfied.

# Article Evidence Summary

## Thesis

The useful article angle is not "AI solved route planning." It is:

> The model got good enough to produce a plausible route quickly, which meant
> the bottleneck moved from making a map work to making that map survive real
> field use.

## 2026 Current Challenge Ground Truth

Source: `years/2026/inputs/official/api-pull-2026-05-04/official_foot_summary.json`

- Official on-foot trails: 101
- Official on-foot segments: 251
- Official on-foot distance: 164.43 miles
- Direction rules: 228 bidirectional, 23 ascent-only
- Challenge window: 2026-06-18 through 2026-07-18, America/Boise

## 2025 Baseline

Sources:

- `archive/legacy-root-2025/docs/retrospectives/2025-planner-in-situ-analysis.md`
- `archive/legacy-root-2025/docs/retrospectives/model-improvement-baseline.md`
- `archive/years/public-history-summary-2026-05-04.md`

Useful points:

- Scott's 2025 public result: 41.82%, 68.89 miles, rank 491.
- The 2025 planner had good problem framing: rural postman, capacitated arc
  routing, mixed directed routing, windy/elevation-aware costs, and trailhead
  logistics.
- The implementation was not cleanly runnable at retrospective time; tests did
  not collect because many tests imported deleted modules.
- The 2025 data loader missed live GeoJSON schema fields, which likely broke
  connector-trail handling.
- Old generated outputs were mathematically plausible but human-unfriendly:
  the VRP artifact had 39 hikes and 337.37 on-foot miles; the trailhead artifact
  had 64 hikes, 287.5 miles, 28 trailheads, and zero reported real connector
  miles.

## 2026 Current Value

Sources:

- `README.md`
- `years/2026/README.md`
- `years/2026/outputs/examples/2026-outing-menu.example.md`
- `years/2026/experiments/2026-05-04-outing-execution-simulation/outing_execution_graph_validated.md`

Useful points:

- The repo now frames the planner around the actual day-of question: "I have a
  fixed amount of door-to-door time today. Where should I park, what should I
  run, how do I get back to the car, and what official segments does that knock
  out?"
- The current public outing menu represents all 251 official segments.
- Current example plan metrics: 164.42 official miles, 280.23 on-foot miles,
  23 runnable outings, 1 manual design hold, 1.7x on-foot/official ratio.
- The route surface is now outing-first: door-to-door buckets, park/start,
  official miles, total on-foot miles, remaining segments, package context, and
  trails.

## First Field-Test Learning

Sources:

- `years/2026/field-tests/pre-challenge/2026-05-05-test-01/README.md`
- `years/2026/field-tests/pre-challenge/2026-05-05-test-01/analysis.md`
- `years/2026/field-tests/pre-challenge/2026-05-05-test-01/strava-summary.json`

Useful points:

- Planned outing: 1B Harrison Hollow.
- Planned: 96 minutes door-to-door, 4.72 official miles, 5.69 on-foot miles, 12
  official segments.
- Actual: 119 minutes door-to-door, 4.74 miles, 100.9 moving minutes, 918 feet
  gain.
- Local preliminary geometry match: 10 of 12 planned segments matched, 3.64 of
  4.72 planned official miles matched.
- Missed planned segments: Who Now Loop Trail 2 and Hippie Shake Trail 1.
- Interpretation: Hippie Shake was intentionally skipped after the user
  realized something had gone wrong; Who Now Loop Trail 2 is the likely root
  miss.
- The route stayed near the planned corridor; the failure was not a large
  off-route excursion. The problem was confusing field execution around reused
  corridors and junctions.
- Product learning: the phone packet needs signpost-style cues, such as "After
  Kemper's Ridge, take #50 Hippie Shake. Do not drop onto #51 Who Now."

## Git-History Evidence

Sources:

- `projects/research-20260505-boise-trails-ai-article/evidence/git-history-2025-06-09-to-2025-06-21.log`
- `projects/research-20260505-boise-trails-ai-article/evidence/git-history-2025-focused-routing-and-fixes.log`
- `projects/research-20260505-boise-trails-ai-article/evidence/git-history-2025-daily-counts.txt`
- `projects/research-20260505-boise-trails-ai-article/evidence/git-history-2026-05-03-to-2026-05-05.log`

Useful points:

- The 2025 window shows a dense burst from 2025-06-09 through 2025-06-21.
- The focused 2025 log captures routing, clustering, one-way, test, dependency,
  memory/OOM, import, and bug-fix churn.
- The generated data-points JSON records 621 commit lines across all local refs
  in that 2025 window, with 332 matching the focused routing/fix keyword set.
- These are git-log evidence-volume counts across all local refs, not a clean
  main-branch release count.

## Article Cautions

- Do not claim the 2026 plan is field-ready. The repo explicitly says current
  Ridge to Rivers conditions, signage, closures, and logistics checks are still
  required.
- Do not claim the May 5 geometry match is official challenge credit.
- Do not overstate 2025 as total failure. It produced strong framing and useful
  artifacts; it just did not cross the practical-use threshold in time.
- Do not frame the lesson as "AI eliminates work." The evidence supports "AI
  creates earlier value and exposes better, more specific work."


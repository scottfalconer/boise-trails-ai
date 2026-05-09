---
name: btc-edge-coverage-audit
description: Audit Boise Trails Challenge route or activity geometry against official segment-edge requirements. Use when reviewing challenge completion, segment lists, trail-system packages, route optimization, activity proof, GPX coverage, or ascent-direction correctness.
---

# BTC Edge Coverage Audit

Core heuristic:
BTC is edge coverage under human constraints, not waypoint visitation.

## Procedure

1. Load the current-year official segment source, defaulting to `years/2026/inputs/official/api-pull-2026-05-04/official_foot_segments.geojson` unless the user requested another year.
2. Identify the required official segment edges and their direction rules. Treat trailheads and trail names as access or grouping hints, not completion objects.
3. Compare the proposed route or activity geometry against each required official segment endpoint-to-endpoint within the project tolerance.
4. For `ascent` segments, verify the segment is climbed in the required direction.
5. Separate official new miles, official repeat miles, connector trail miles, road miles, and deadhead miles.
6. Record partial overlaps, crossings, and near misses as evidence, but do not count them as complete.
7. Before ranking or promoting, confirm every claimed segment id is covered from one official endpoint to the other.

## Do Not Infer

- Visiting a trailhead covers adjacent official segments.
- A shortest waypoint tour covers required trail geometry.
- A trail name covers every official segment with that name.
- A GPX overlap proves full endpoint-to-endpoint completion.
- Multiple activities can be stitched into one segment credit.
- Bike, vehicle, or mixed-mode travel counts for the on-foot category.

## Output

- Coverage status: `complete`, `partial`, `direction_failed`, `not_covered`, or `needs_activity_geometry`.
- Official segment ids covered, missed, partially overlapped, and direction-failed.
- Official new miles, official repeat miles, connector miles, road miles, and total on-foot miles.
- Evidence source and date.
- Repair needed before the route or activity can support a completion claim.

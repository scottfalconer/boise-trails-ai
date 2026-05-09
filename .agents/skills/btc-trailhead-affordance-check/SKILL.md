---
name: btc-trailhead-affordance-check
description: Verify access assumptions for Boise Trails Challenge routes. Use when a route starts, ends, parks, shuttles, re-parks, or transitions at a mapped trailhead, pullout, road crossing, residential road, OSM parking feature, private Strava-derived anchor, or informal access point.
---

# BTC Trailhead Affordance Check

Core heuristic:
A trailhead is not a start until access is real.

## Procedure

1. Load `docs/BTC_LOCAL_REALITY.md` before accepting or rejecting an access point.
2. Identify every proposed start, finish, re-park, shuttle, and access node.
3. Classify each access point as a known parking area, mapped trailhead, road shoulder, pullout, residential road start, Strava-derived prior anchor, user-reviewed anchor, or unknown access point.
4. Check legal and practical access: parking, road passability, gate risk, signage, seasonal closure, private-property ambiguity, mud/condition constraints, and whether imagery is stale.
5. Separate official evidence, Ridge to Rivers or land-manager evidence, OSM/map evidence, imagery evidence, field-photo evidence, private Strava-derived evidence, and user-provided evidence.
6. State unsupported assumptions explicitly.
7. If access is uncertain, keep the route parking-gated or propose a known-access fallback and account for added connector, road, repeat, distance, and time.

## Do Not Infer

- Legal parking from a trailhead label.
- Current access from old imagery.
- Road passability from map geometry alone.
- Public use from a service road, cat track, or shoulder without evidence.
- Publication readiness from private exact-coordinate evidence.
- A need to re-block an anchor already reviewed as `yes` unless access changed or uncertainty is specific.

## Output

- Access status: `known`, `accepted_user_reviewed`, `parking_gated`, `blocked`, or `unknown`.
- Evidence used, grouped by source type.
- Unsupported assumptions.
- Fallback access option, when needed.
- Added distance, duplicated official mileage, road mileage, and time impact.
- Human-validity note for whether the route can be recommended now.

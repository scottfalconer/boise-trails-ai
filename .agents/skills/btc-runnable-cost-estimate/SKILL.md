---
name: btc-runnable-cost-estimate
description: Evaluate Boise Trails Challenge outings by real field cost instead of map cost. Use when ranking, replacing, or recommending best, fastest, shortest, most efficient, or most practical outings, especially when p75 timing, DEM effort, heat, water, bailout, parking, or hard stops matter.
---

# BTC Runnable Cost Estimate

Core heuristic:
Runnable cost beats map cost.

## Procedure

1. Load `docs/BTC_LOCAL_REALITY.md` before ranking or recommending route alternatives.
2. Use `time_estimates_minutes.door_to_door_p75` as the conservative user-facing planning number when available. Preserve raw model sums as `raw_total_minutes`.
3. Verify DEM effort fields: `ascent_ft`, `descent_ft`, `grade_adjusted_miles`, `estimated_moving_minutes_p50`, and `estimated_moving_minutes_p75`.
4. Add drive time, parking/prep time, same-day transfers, route-finding complexity, and overlap/double-back penalties.
5. Account for heat exposure, shade, water, bailout, mid-route car access, mud/closure status, and family/work hard stops.
6. Compare alternatives on door-to-door risk first, then official new miles, total on-foot miles, connector/road mileage, and future-day impact.
7. Do not promote a faster-looking replacement unless it also has p75 timing, DEM effort, continuous navigation GPX, access status, and field-valid cue text.

## Do Not Infer

- Fewer miles is automatically better.
- Fewer starts is automatically better.
- A lower-bound optimizer result is a runnable plan.
- A route with no p75 timing or DEM effort can replace a certified card.
- A slower split is worse when it creates car access, water, bailout, or hard-stop safety.

## Output

- Cost status: `runnable`, `needs_timing`, `needs_effort`, `logistics_blocked`, or `paper_only`.
- Door-to-door p75, moving p75, drive/prep/transfer time, and route-finding penalty.
- Official new miles, official repeat miles, connector miles, road miles, and total on-foot miles.
- Heat, water, bailout, car-pass, and hard-stop notes.
- Recommendation and the missing evidence needed before promotion.

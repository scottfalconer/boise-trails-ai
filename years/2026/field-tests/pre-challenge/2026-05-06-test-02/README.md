# Pre-Challenge Test Day 2: One-Hour Field Packet Check

Date: 2026-05-06
Status: planned
Phase: pre-challenge field test

## Purpose

Today's available window is about one hour door-to-door. The current canonical
outing menu does not have a full official outing that truly fits that bound:
the shortest full outing is `Scott's Trail` at about 79 min p75 / 89 min p90
door-to-door.

So this test should not be treated as a completion attempt. It is a navigation,
phone UX, and time-calibration test.

## Recommended Test

Primary recommendation:

```text
Harrison Hollow cue micro-test
```

Why this test:

- It directly revisits the confusing area from 2026-05-05.
- It tests whether the phone field packet now explains the actual trail
  transition, not just official segment order.
- It tests whether Gaia / GPX / field-packet map rendering is understandable
  when a route crosses or reuses nearby corridors.
- It keeps the outing close enough to turn around cleanly inside a hard stop.

Suggested execution:

1. Park at Harrison Hollow Trailhead.
2. Start Strava and the phone field packet before leaving the car.
3. Follow the `1B. Harrison Hollow` instructions only until the Who Now /
   Harrison Ridge / Kemper's Ridge decision area is clear.
4. At each signed junction, note whether the field packet language matches the
   physical signpost number, trail name, and arrow.
5. Treat this as a strict timebox: if the outbound test has not reached the
   target decision area by the halfway point of the door-to-door window, turn
   around instead of chasing the route card.
6. Turn around early enough to be back at the car and home inside the one-hour
   door-to-door window.

Expected official-credit result:

```text
No planned official completion credit.
```

If the activity happens to complete full official segments, analyze them after
the run, but do not mark anything complete just because the test touched part of
a segment.

## Backup If The Window Expands

If the real available window becomes closer to 90 minutes, use:

```text
Scott's Trail
Upper Interpretive Trailhead
79 min p75 / 89 min p90 door-to-door
1.05 official mi / 2.01 on-foot mi
```

This is the shortest full outing in the current canonical menu, so it is the
best end-to-end route-card test if there is enough time.

## What To Record

Record these after the test:

- actual door-to-door start and finish
- time from parking to first trail junction
- app used for GPX navigation
- whether the field packet made the next turn obvious
- signpost numbers/names/arrows that were helpful or missing
- any place where the GPX line overlapped itself or looked ambiguous
- whether the route card should have recommended turning around earlier
- Strava activity id or local pull folder after import

## Current Planner Context

Today's proof work found an important boundary:

- Current menu coverage still covers 251/251 official segments.
- The connector-graph rural-postman lower bound is useful but is only a lower
  bound, not a runnable field plan.
- The first route-efficiency proof was too abstract to call a real execution
  proof because it did not enforce home-to-home p90 field-day limits.
- The new field-day feasibility proof is reality-based against current personal
  bounds and currently fails: 14 of 26 runnable loops exceed the largest p90
  daily bound.
- A newer single-segment split probe fixes some of that gap: 14 of 29
  previously p90-missing official segments now have graph-validated,
  continuous, under-260-minute diagnostic loops. Fifteen remain unresolved.
- Manual access anchors look promising for Harlow/Spring and Sweet Connie, but
  those anchors are not field-ready until parking/access is verified. Shingle
  Creek is still the main strict-p90 blocker.
- The forced-anchor probe narrowed this further: Dry Creek `1545` and Sweet
  Connie `1667` have strict field-ready solutions under 260 minutes p90.
- Parking/access verification promoted Avimor Spring Valley Creek / Twisted
  Spring parking for the Harlow/Spring cluster, so Harlow/Spring is no longer
  the active strict blocker.
- Shingle Creek `1656` remains the only missing p90 segment. The best current
  source-verified route is 292 minutes p90 / 260 minutes p75.
- The repaired candidate-universe audit now covers 250/251 official segments
  under the strict 260-minute p90 bound. Shingle `1656` is the only strict miss;
  adding the 292-minute Shingle exception would cover 251/251 but would not
  satisfy the current p90 rule. It also remains a candidate-coverage repair, not
  a schedule: the exact exception set cover still selects 80 loop candidates.
- The active proof audit for the stricter field-day goal is now:
  `years/2026/checkpoints/field-day-p90-completion-audit-2026-05-06.md`.
- The repaired field-day pack audit adds a second planning lesson: even if
  Shingle were manually accepted at 292 minutes p90, the selected candidate set
  remains too fragmented for the current weekday/weekend availability split.
  That should not affect today's one-hour field test, but it is why the proof
  still should not be called complete.
- A wider joint optimizer confirms the same shape: route selection plus
  field-day packing still cannot produce a feasible schedule under the current
  availability profile. Today's field test should stay focused on navigation
  cue quality and timing calibration, not full-plan completion proof.
- The best current max-coverage schedule under strict bounds is 219/251
  segments. That is not a field-test target for today, but it is the planning
  baseline future field tests should improve.
- Sensitivity testing found that the first full-clear scenario in the current
  generated universe is 360 minutes on both weekdays and weekends. That is a
  stress-test profile, not today's plan.
- The closest tested near-miss scenario is 292-minute weekdays / 360-minute
  weekends: 249/251 segments, missing Deer Point `1540` and Central Ridge Spur
  `1558`. Both have individual field-day options, so the issue is route/day
  packing rather than impossibility. Today's one-hour test should still stay
  focused on navigation clarity and timing calibration.
- The relaxed-drive full-clear draft now has loop-level GPX sources for all 50
  selected loop rows, and the day-level GPX export now validates for all 31
  dated days. That still does not make it today's plan because the draft uses a
  relaxed 292/360 timing profile and today's window is only about one hour.
- The strict-profile recovery audit classifies the 32 segments missing from the
  honest 260/180 fallback: Shingle Creek `1656` has no strict candidate, while
  the other 31 missing segments have strict candidates but lose schedule
  tradeoffs. That is the current proof boundary, and today's test should feed
  timing/navigation calibration rather than chase completion credit.
- The strict-profile swap audit then forced each missing segment into the
  schedule: 10 are one-for-one preference swaps, 21 lower total covered segment
  count, and Shingle is still the only no-candidate row. Today's field evidence
  should therefore focus on improving the route/time model, not just swapping
  one segment into the same strict schedule.
- A Shingle-specific exhaustive anchor probe tested all 74 known public,
  manual, and private Strava-derived parking anchors. None produced an
  under-260-minute graph/track-valid Shingle route; the best field-ready row is
  still 292 min p90 / 260 min p75 from Dry Creek / Sweet Connie roadside
  parking. Today's short field test should therefore calibrate navigation and
  timing, not try to solve Shingle in the field.

The next planning step is not to pretend the proof passed. It is to split or
redesign oversized loops and keep the field menu honest about what fits inside
real door-to-door windows.

See also: `years/2026/notes/daily-work-log.md`.

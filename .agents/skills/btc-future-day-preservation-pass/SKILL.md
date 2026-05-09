---
name: btc-future-day-preservation-pass
description: Check whether a Boise Trails Challenge route choice, field completion, split-start replacement, closure, access change, or time-window decision preserves the remaining certified menu and future challenge feasibility.
---

# BTC Future-Day Preservation Pass

Core heuristic:
Today's route must preserve the remaining certified menu.

## Procedure

1. Identify how the proposed route, completion, miss, extra segment, closure, access blocker, or route replacement changes the remaining official segment set.
2. Treat already-completed segments that remain physically necessary as official repeat mileage or connector context, not as new remaining credit.
3. Run or request recertification after meaningful state changes: proven completions, missed segments, extra covered segments, closures, route-list changes, access blockers, condition blockers, and route/parking edits.
4. Check remaining certified-calendar capacity, p75/p90 time windows, hard stops, heat windows, and bailout options.
5. Compare today's benefit against future-route damage. Preserve less-optimal backups when they keep the month feasible.
6. If a split is slower but improves bailout, water, heat, car access, or hard-stop fit, keep that value visible instead of rejecting it on speed alone.

## Do Not Infer

- More official miles today is always better.
- A completed overlap means the later physical route disappears.
- A slower split is worse if it improves future logistics.
- Future capacity remains valid after closure, access, or completion state changes.
- Manual deletion from a future outing is enough without recertification.

## Output

- Future status: `preserved`, `needs_recertification`, `future_capacity_risk`, `blocked`, or `better_as_backup`.
- Remaining official segment impact.
- Repeat/connector role for completed overlaps.
- Future day capacity and hard-stop risk.
- Recommended repair or scheduling change.

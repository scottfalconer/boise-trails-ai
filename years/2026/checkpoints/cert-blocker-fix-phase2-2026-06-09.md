# Certification Blocker Fixes — Phase 2 (route source + dominance) — 2026-06-09

Branch `cert-blocker-fixes-2026-06-09`, after phase 1 (commit a3739a0). Scott
approved: re-anchor the 6 dominated routes, restore H1 for 10A, dedupe 1680, run
the full re-baseline and verify.

## Headline correction discovered during execution

The "6 dominated routes" were **not** 6 wrong starts. Tracing each one:

- **5 of the 6 (5A, 16A-2, 1A-2, 16C-1, 15B) were a dominance-gate false positive.**
  Each route is *already parked at the exact anchor the gate told it to move to*
  (5A is already at West Hidden Springs, 15B at Bob's, etc.). The gate flagged
  them because it compared the route's real GPX-traversal mileage against the
  multi-start audit's *optimistic standalone estimate* for that same anchor.
  `best_dominating_alternative` excluded `applied_to_current_route` alternatives,
  but `multi_start_same_credit_alternatives` never set that flag — so a route
  could be "dominated" by its own anchor. Fixed in `build_route_review_pack.py`
  (`_same_anchor` token-set match → mark same-anchor alternatives
  `applied_to_current_route`). Token-set matching strips only generic words
  (Trailhead/Parking/anchor/road…) so "Full Sail" stays distinct from "Full Sail
  Trailhead, N 36th St Parking" (the real FD14D case is preserved — its test
  still passes). Regression test added.

- **1A-1 was a genuine residual, but also not a wrong start.** It is already at
  the accepted lower N 36th St / strava-parking-anchor-13 parking (shown
  privately as "prior parking anchor 13"). The ~1.7 mi gap is route *shape*: the
  shorter accepted FD14D card geometry is not applied. Resolved with a
  route/source-hashed **waiver** (private, gitignored) documenting the honest
  reason; gate now passes with 1A-1 waived (expires 2026-07-18).

So the re-baseline did **not** materially shift ~27 routes' mileage (my earlier
estimate compared pre-reconcile *estimates* to post-reconcile *truth*). The only
real route change is route 18A losing the double-counted segment 1680.

## What was fixed

1. **Segment 1680 dual-claim** — deduped at the block source; route 17 keeps it,
   18A declares it owned-elsewhere (official 5.58→4.43). Canonical now 251/251
   unique segments, plan-wide official miles **164.44** (was 165.59; matches the
   164.43 official total). Latent-credit audit: 0 dual claims.
2. **Dominance-gate self-anchor false positive** — real bug fix (above);
   5 routes correctly clear to PASS_NON_DOMINATED.
3. **Registry** — now consumes the real review pack and reports 0 unwaived
   failures, all 31 HOLD_CURRENT_RECERTIFIED (1A-1 waived).

## What was NOT done (blocked / deferred)

- **10A → H1 Avimor: BLOCKED.** `promote_harlow_h1_route_card.py` is structurally
  stale — it matches routes by `field_menu_label`, which manual components
  (`manual-10a`) no longer carry in the canonical (only multi-start components
  get one). It raises on retired FD labels. Restoring H1 needs the promotion
  infra reworked to match by candidate_id / segment-set — real surgery, not safe
  to do blind 9 days out. **10A remains the Harlow's west-access probe (21.84 mi,
  manual_required parking, ships field_ready).** It is not a dominance failure,
  so it does not block certification, but it is the one review item unresolved.
  Same infra blocks applying the shorter FD14D shape for 1A-1.
- **Deferred exporter-quality polish** (real but not certification-blocking, and
  carrying open design questions risky to rush): real per-route
  `start_justification` (still exporter boilerplate), the impossible
  repeat-official mileage cue cap (54 cues), live-map overlap chevrons / GPS
  readout, and water/heat/bailout annotations.

## Verification (re-baselined packet)

Full 8-command certification chain passes on the regenerated packet:
export; latent-credit (0 dual claims); progress; recertification; route-edge
cover (0 hard failures, 0 advisories); field-tool completion (20/20
requirements, special-management gate passed, 31 field-ready); field-route
walkthrough (31/31); post-credit connector (0 findings). Dominance gate passes
(1A-1 waived). Registry: 0 unwaived failures. Full `pytest` suite: see commit.

## Recommended follow-ups (focused, separate sessions)

1. Rework the H1 (and FD14D) accepted-card promotion path to match by
   candidate_id/segment-set so 10A → H1 and 1A-1 → FD14D shape can be applied;
   then drop the 1A-1 waiver.
2. Exporter polish: real start_justifications, repeat-mileage cue cap + test,
   live-map chevrons/GPS readout, water/heat/bailout annotations.
3. Scrub the 3 committed experiment Strava-sim files (phase-1 follow-up).

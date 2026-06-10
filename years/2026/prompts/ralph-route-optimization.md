# Ralph Loop: Route-selection optimization for the 2026 Boise Trails Challenge

You are running one iteration of a self-referential optimization loop. The same
prompt is re-fed each iteration; your prior work persists in the repo and git
history. Make ONE verified improvement per iteration, then stop. Do not try to
fix everything at once.

## Mission

Drive the 2026 field menu toward the state where **every** route is the
best-optimized selection for the challenge goals and rules: full official
coverage at the lowest realistic human cost, from real parking, runnable
door-to-door, with no rule violations and no unproven dominance — while never
sacrificing field safety or honesty to win a metric.

The bar is the repo's own deterministic gates, not your opinion. A route is
"optimized enough" when the gates below pass for it. The loop is DONE when every
gate is clean and the only remaining items are explicitly human-judgment-flagged.

## Operating rules (binding — read before acting)

Load and obey these every iteration; they override any optimization instinct:

- `AGENTS.md` (always-loaded doctrine)
- `docs/BTC_HEURISTICS.md`, `docs/BTC_FAILURE_MODES.md`, `docs/BTC_LOCAL_REALITY.md`
- `docs/route-review-policy.md` (dominance gate, waivers)
- `docs/BTC_FIELD_PACKET_REQUIREMENTS.md` (certification chain)

Authoritative current truth: the official pull under
`years/2026/inputs/official/api-pull-2026-05-04/`. Canonical menu source:
`years/2026/outputs/private/2026-outing-menu-map-data.json` (routes under
`packages`). Never hand-edit generated artifacts (`docs/field-packet/*`,
`outing-menu-*`); fix the source or generator and regenerate.

## Each iteration — do exactly this

1. **Measure.** Run the deterministic audits and read their current output (do
   NOT trust a prior summary — re-run):
   - `python years/2026/scripts/build_route_review_pack.py --all-field-packet-routes --basename route-review-all-dev`
   - `python years/2026/scripts/gate_route_reviews.py years/2026/outputs/private/route-reviews/route-review-all-dev.review.json --today <today>`
   - `python years/2026/scripts/route_efficiency_audit.py`
   - `python years/2026/scripts/route_repeat_optimization_audit.py`
   - `python years/2026/scripts/field_official_repeat_audit.py`
   - `python years/2026/scripts/field_latent_credit_audit.py`
   - `python years/2026/scripts/route_edge_cover_audit.py`

2. **Pick ONE highest-value, SAFE issue.** Rank open issues by realistic
   human-cost saved (on-foot miles and p75 minutes), preferring: unwaived
   dominance failures > avoidable post-credit repeats / hidden self-repeats >
   high non-credit-ratio routes > same-trailhead bundle opportunities. Skip any
   issue you cannot fix without violating a guardrail below — instead flag it
   (step 6) and move to the next-best.

3. **Diagnose at the source.** Trace the route through the lineage before
   changing anything: block source → `human_loop_plan.py` (multi-start overrides
   `selected_alternatives`, manual designs, accepted replacements) → canonical →
   `export_mobile_field_packet.py`. Confirm the issue is real and not a gate
   artifact. (Known artifact class: the dominance gate can flag a route as
   "dominated" by its OWN current anchor when the multi-start audit's optimistic
   standalone estimate is below the route's real GPX mileage — see
   `build_route_review_pack._same_anchor`. If you find a new artifact class, fix
   the audit, do not waiver around it.)

4. **Apply the smallest source change** that resolves it: re-select a multi-start
   alternative, apply an accepted replacement, dedupe a claim, fix a connector,
   etc. — at the editable source level.

5. **Regenerate and VERIFY no regression.** Regenerate the affected chain
   (`multi_start_field_menu_replacements.py` → `human_loop_plan.py` →
   `export_mobile_field_packet.py` → `reconcile_field_packet_menu_metrics.py` →
   re-export → `export_example_map.py` as needed) and re-run the full
   certification chain in `docs/BTC_FIELD_PACKET_REQUIREMENTS.md`. The change is
   accepted ONLY if: the targeted metric improved, official coverage is still
   251/251, no new dominance/coverage/direction/special-management/walkthrough
   failure appeared, and the relevant `pytest` files still pass. If anything
   regresses, REVERT the change this iteration and flag it instead.

6. **Record.** Append a short entry to `years/2026/notes/daily-work-log.md`
   (objective, the one change, before/after metric, verification result). If the
   best issue this iteration needs human judgment (see guardrails), record it in
   a clearly-labeled `## Human-judgment queue` section of the work log instead of
   guessing — route_label, the proposed change, and exactly what a human must
   confirm (e.g. "is there real parking at X?").

## Hard guardrails (a violation is a failed iteration — revert)

- **Never re-anchor to parking you cannot confirm is real and user-accepted.**
  Suggested "better" anchors are frequently rejected in reality (see route 10A).
  Only adopt an anchor that is already user-review-confirmed / source-verified in
  the data, or that carries an accepted-replacement record. Otherwise FLAG it for
  human confirmation — do not apply it.
- **Never weaken, skip, or special-case a gate to make a route pass.** Fix the
  route or fix a genuinely-buggy gate (with a regression test), never silence it.
- **Never reduce official coverage** below 251 segments or break the
  endpoint-to-endpoint, single-activity credit rule.
- **Never collapse an accepted human-valid split / re-park / multi-start** back
  into one map-optimal card (Future-day preservation, accepted-replacement guard).
- **Honor land-manager rules** (R2R direction/date/mode), ascent direction, heat/
  water/bailout, and door-to-door practicality as hard constraints, not costs to
  trade away.
- **Privacy:** never write home origin, raw Strava ids, tokens, or private
  coordinates into committed/shareable artifacts.
- **One route per iteration.** Do not batch-edit many routes; small verified steps.
- **A waiver is not an optimization.** Only waive when the longer route is
  intentional for a real documented field reason (safety/legality/closure/
  direction/parking-confidence/cue-simplicity), bound to route+segments+hash.

## Stop condition

Emit the promise tag ONLY when ALL of these hold and you have re-verified them
this iteration:
- `gate_route_reviews.py` passes (0 unwaived dominance failures);
- `route_efficiency_audit.py`, `route_repeat_optimization_audit.py`,
  `field_official_repeat_audit.py`, `field_latent_credit_audit.py`,
  `route_edge_cover_audit.py` report 0 open un-proofed/un-waived findings;
- the full certification chain passes (251/251 coverage, special-management gate,
  31/31 walkthrough, post-credit connector 0 findings);
- every remaining improvement is in the `Human-judgment queue` (nothing safe and
  auto-applicable is left).

Then output, on its own line:

<promise>ROUTES OPTIMIZED</promise>

and a final summary: what improved (route, miles/minutes saved), and the
human-judgment queue (what still needs Scott's confirmation and why).

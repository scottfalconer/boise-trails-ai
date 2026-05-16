# High-Value Route Mapping Optimization

Generated: 2026-05-10

Objective: identify a high-value route-mapping optimization that saves real
on-foot miles, human effort, or field time, while respecting BTC edge-credit
rules, parking/access reality, current accepted route replacements, and the
phone field-packet contract.

## Recommendation

Treat the completed field-day layer as the execution foundation, then focus the
next route-mapping optimization on `10A` certifiable-anchor redesign.

The field-day layer already reframes the plan from standalone route cards into
day-level execution. The remaining high-value problem is converting the best
review-only route redesign into a certified route-card replacement without
weakening access, coverage, cue, GPX, p75/p90, or field-packet truth.

This is different from blindly adding more split routes. The split/re-park
work already promoted the obvious certified replacements (`1A`, `4C`, `5`, and
`15A`). After those replacements, the current multi-start audit has only one
high-leverage remaining area: `10A` Harlow / Hidden Springs access. The exact
best paper split, `10A-MS-08`, is now blocked by access verification, so the
route-level optimization is not promotion; it is a certifiable-anchor redesign
that searches outward from the uncertain road/probe starts and reprices the
added connector cost. The next useful artifact is a regenerated `10A` route
source for the best certifiable-anchor candidate, not another field-day-layer
proof.

## Why This Is High Value

Current proof artifacts already show the leverage:

- The relaxed-drive field-day proof covers 251/251 official segments in 31
  field days under the `responsible_relaxed_18mi_v1` feasibility profile.
- It uses 14 multi-start days, but only 76 total between-start driving minutes.
- Only one selected day has more than 20 minutes of between-start driving.
- The result turned the near-miss from a schedule-pressure problem into a
  full-clear schedule without requiring shuttles.

That is a human-factor win: it trades small, intentional car moves for fewer
oversized outings, better bailout/water/car access, and fewer family/work
hard-stop failures.

## Current Field-Packet Pain Points

The current phone packet still has several standalone route cards whose
non-credit burden and cognitive load are large:

| Route | Official mi | On-foot mi | Non-credit / repeat mi | P75 min | Human-factor issue |
|---|---:|---:|---:|---:|---|
| `13` Freestone / Three Bears / Curlew | 14.35 | 25.12 | 10.77 | 490 | Long same-car route, no car pass, heavy connector/repeat warnings. |
| `16A-2` Shingle / Sheep Camp | 5.53 | 14.96 | 9.43 | 310 | Required Shingle ascent plus long access/return burden. |
| `6` Cartwright / Peggy's | 13.67 | 21.53 | 7.86 | 448 | Long field day with connector complexity and no car pass. |
| `10A` Harlow / Hidden Springs | 7.30 | 13.62 | 6.32 | 360 | Best remaining concrete access-redesign opportunity; exact `10A-MS-08` paper split is blocked. |
| `18` Bogus Mores / Brewers / Tempest | 5.08 | 11.25 | 6.17 | 320 | Long mountain route; schedule and condition placement matter more than map purity. |

These should not all be handled the same way. `13` and `16A-2` look inefficient,
but the current corrected split audit does not show a better certified split.
They need either new legal access evidence or day-level scheduling treatment,
not a fake shortcut or route-card rewrite.

## Concrete Remaining Savings Candidate

The active route-level target is `10A`, but the target changed after access
verification:

| Candidate | Baseline on-foot | Candidate on-foot | Savings | P75 delta | Status |
|---|---:|---:|---:|---:|---|
| `10A-MS-08` North Burnt Car / Harlow west split | 13.62 | 10.24 | 3.38 mi | -43 min | Not certifiable as-is; both starts remain parking/access blocked. |

The next optimization is to find the nearest certifiable Avimor start, such as
a mapped primary trailhead or park/parking surface, then rerun the route with
the honest connector, p75/p90, cue, and coverage costs. If the redesigned
route still beats the 13.62-mile baseline by a material amount, it becomes the
best next route-card replacement candidate. If the added connector erases the
savings, keep `10A` active and use the result as a parking/access lesson rather
than promoting a paper shortcut.

The first certifiable-anchor repair screen produced a better concrete sprint
than continuing to argue for `10A-MS-08`: `10A-MS-13` reanchored through Avimor
Spring Valley Creek parking and tied back to West Creeks Edge Drive. It is still
review-only, but it survives the connector-budget math with 0.73 round-trip
connector miles, 2.13 adjusted on-foot miles saved, and a 6-minute adjusted p75
improvement. Promotion still requires regenerated route source, GPX, cue text,
p75/p90, parking/access proof, segment coverage, and field-packet audits from
the same source.

## Guardrail From This Pass

The route planner should not let a candidate's own required official segments
serve as hidden connector shortcuts while choosing access, between-trail links,
or return-to-car paths. That creates paper savings and confusing field cues:
the runner is told they are earning credit while the route line is also using
the same official trail as connector/repeat movement.

The current uncommitted planner change adds an
`avoid_official_segment_ids` guard to connector path search and candidate
construction. Targeted validation passed:

```bash
python -m pytest years/2026/tests/test_personal_route_planner.py years/2026/tests/test_multi_start_alternative_audit.py
# 54 passed in 0.51s
```

This guard is not a mileage-saving feature by itself. Its value is that it
keeps future savings honest, so the planner optimizes real human movement
instead of hiding self-repeat cost inside cue math.

## Suggested Implementation Path

1. Keep the field-day layer as the completed day-level execution surface and
   use its route-card promotion gaps as the work queue.
2. Run a focused `10A` certifiable-anchor redesign sprint. Do not promote the
   current `10A-MS-08` road/probe split unless new field/source evidence changes
   the access decision. Search outward to known parking/trailhead surfaces,
   reprice the connector, then regenerate the route source and run the full
   certification chain only if residual savings remain material.
3. Add a recurring "pain index" audit that ranks route cards by non-credit
   miles, p75/p90 pressure, cue overlap warnings, car-pass absence, water
   absence, and hard-stop risk. Use it to find the next route-level target
   after `10A`, not just the highest map-distance ratio.

## Completed Foundation And Current Sprint

The generated field-day layer exists:

- `years/2026/checkpoints/human-executable-field-day-layer-2026-05-10.json`
- `years/2026/checkpoints/human-executable-field-day-layer-2026-05-10.md`
- `years/2026/checkpoints/human-executable-field-day-layer-2026-05-10-manifest.json`

That layer is the completed strategy shift. It covers 251/251 official segments
across 31 field days and 50 loops, but it is intentionally not a blanket
publication-ready claim: only 15 loops currently match certified route cards,
and the other 35 remain explicit route-card promotion gaps.

The route-pain index now consumes the structured `10A-MS-08` access decision:

- `years/2026/checkpoints/10a-ms-08-access-verification-2026-05-10.json`
- `years/2026/checkpoints/route-pain-index-2026-05-10.json`
- `years/2026/checkpoints/route-pain-index-2026-05-10.md`

Current result: `10A` remains the top route-mapping target, but as
`certifiable_anchor_redesign`, with `0.0` known actionable unpromoted miles and
`3.38` blocked paper miles.

The certifiable-anchor repair audit then refined that target:

- `years/2026/checkpoints/certifiable-anchor-repair-audit-2026-05-10.json`
- `years/2026/checkpoints/certifiable-anchor-repair-audit-2026-05-10.md`

Current result: five `10A` redesign candidates pass the review-only
connector-budget screen. The leading candidate is `10A-MS-13` from Avimor
Spring Valley Creek parking to the West Creeks Edge tie-in, with 2.13 adjusted
on-foot miles saved and -6 adjusted p75 minutes. This is the next route-card
design sprint, not a field-packet replacement yet.

## Adversarial Frame Shift Pass

### 1. Real Consequence Layer

The protected outcome is not "the optimizer found a shorter line." The protected
outcome is that the runner can leave a legal, repeatable parked start, follow
the field packet without artifact disagreement, earn the intended official edge
credit in the BTC app, return to the car, and keep the outing inside real
heat/water/family/work constraints.

### 2. Object Currently Being Verified

The immediate object is the `10A` route-card replacement opportunity surfaced by
the route-pain index. The tempting first artifact is the paper `10A-MS-08`
multi-start candidate, but that is not the true thing that needs verification.
The true object is a field-certifiable `10A` redesign from legal, cue-able
parking with honest connector cost, GPX/cue/phone agreement, and segment
coverage preserved.

### 3. Verifier Scope And Negative Space

| Check | Proves | Does not prove | False-confidence risk |
|---|---|---|---|
| `route_pain_index.py` | Current field-packet route rows rank `10A` as the top actionable target after access decisions are applied. | Does not prove any replacement is legal, runnable, or field-certified. | Treating a high priority score as permission to promote a route. |
| `10a-ms-08-access-verification` | The exact North Burnt Car / Harlow west paper split is blocked for parked-start certification today. | Does not prove every Avimor/FHP/Spring Valley Creek anchor is unusable. | Rejecting the whole `10A` savings goal because one candidate failed. |
| `certifiable-parking-expansion-audit` | The right next search is nearest certifiable parking plus priced connector/tie-in, not nearest road proof. | Does not prove the current generator can price that shape well enough yet. | Assuming a known park anchor automatically beats the active route. |
| Field-packet and completion audits | Existing cards can be checked for source consistency, GPX/cue mileage truth, coverage, and public-safety boundaries. | Do not prove day-of trail legality, mud, heat, construction, or private/HOA parking rules. | Treating generated packet consistency as field readiness. |
| Field-day layer proof | The full 251-segment challenge can be arranged into human-scale day bundles under a relaxed-drive profile. | Does not certify the 35 loop gaps that lack current route-card production. | Treating day-level coverage as route-card certification. |

### 4. Pass-But-Fail Premortem

Assume every current check passes. The recommendation could still fail if:

- A future Avimor anchor looks public in map/source data but has no-parking,
  resident-only, permit, or HOA signage in the field.
- The redesigned `10A` connector saves miles on paper but adds confusing turns,
  private-road ambiguity, or a GPX/cue mismatch near the neighborhood edge.
- The route earns all official segments but repeats already-satisfied official
  trail as hidden connector mileage, making the field card harder and longer
  than the route score suggests.
- The field-day bundle covers all official segments but sequences loops in a way
  that fails heat, water, transfer, or family/work hard-stop reality.
- A day-of Ridge to Rivers, Forest Service, construction, mud, or fire condition
  blocks the access or trail even though the static route card remains green.
- A parking source is strong for a formal park but weak for the specific
  nonresident trail-access use the runner needs.
- The field packet, GPX zip, live map, and route-pain source drift after a
  generator rerun and the user follows the stale artifact.

### 5. Forced Frame Shifts

- Current frame: "Can we verify `10A-MS-08`?"
  Alternative frame: "Should we search for the nearest certifiable Avimor/FHP
  parking surface, then price the connector/tie-in and p75/p90 against active
  `10A`?"
- Current frame: "Which route has the highest pain score?"
  Alternative frame: "Which route has measured mile/time savings plus a
  solvable evidence blocker?"
- Current frame: "Can a field-day layer cover 251/251 official segments?"
  Alternative frame: "Can the runner execute each day as a sequence of certified
  route cards with visible transfers, car access, water, and p75/p90 bounds?"
- Current frame: "Is the nearest road point close enough to use?"
  Alternative frame: "What is the nearest legal, repeatable, cue-able parking
  surface, and how much connector tax does it add?"
- Current frame: "Can a generated GPX cover the official edges?"
  Alternative frame: "Can the phone packet, cues, GPX, and live map describe the
  same car-to-car movement a tired human will actually follow?"

### 6. Candidate Failure Versus Goal Failure

`10A-MS-08` failing access certification is candidate failure, not goal failure.
The goal remains valid because the active `10A` card still has 6.32 non-credit /
repeat miles and a 360-minute p75, and the best paper candidate shows 3.38 miles
and 43 p75 minutes of potential leverage. The smallest reframing that keeps the
goal alive without weakening standards is:

1. Keep `10A` active.
2. Mark `10A-MS-08` as `not_certifiable`.
3. Search outward for certifiable parking anchors.
4. Allow an explicit connector tie-in waypoint.
5. Regenerate only from a source that can produce coverage, p75/p90, GPX, cue,
   parking, and field-packet evidence together.

### 7. Evidence Ladder

| Claim or option | Evidence class | Promotion status |
|---|---|---|
| Current official `10A` segment requirements and direction rules | Current-year official challenge data under `years/2026/inputs/official/` | Authoritative for required edges. |
| Current active `10A` card has 13.62 on-foot mi and 360 p75 min | Current generated field-packet data | Valid baseline, subject to artifact consistency audits. |
| `10A-MS-08` saves 3.38 mi and 43 p75 min | Current indirect/generated multi-start audit | Useful design target only. |
| `10A-MS-08` parked starts are blocked | Current access-verification checkpoint using public source, map/imagery, and prior review evidence | Blocks promotion of this exact candidate. |
| FHP/Spring Valley Creek or another Avimor anchor may rescue the goal | Source-backed anchor idea plus generated rough pricing | Gated; needs waypoint-constrained route design and certification. |
| Day-level field bundles can reduce human effort across the challenge | Current generated field-day proof and route-card layer | Valid approach, but route-card gaps remain. |
| Day-of legality, mud, heat, and closure status | Field-only/current-condition unknown | Must be checked before final route use. |

## Stop-And-Check Summary

Coverage:

- The verification checks current field-packet route rows, multi-start audit
  alternatives, missing parked-start evidence from the completion audit, the
  structured `10A-MS-08` access decision, and generated manifest integrity.
- It does not prove day-of trail legality, mud/heat state, final route-card GPX
  coverage after a redesigned anchor, or that a newly found parking surface is
  legal for the user's exact use.
- Failure modes that would still slip past: a future Avimor anchor that looks
  like parking in map data but has private/permit/no-parking signage; a
  redesigned `10A` route that saves miles but reintroduces hidden official
  repeat mileage or mismatched cue/GPX/phone artifacts.

Material issue surfaced and addressed:

- The first pain index treated `10A-MS-08` as an access-verification sprint
  even after access review blocked the exact anchors. The audit now ingests the
  access decision and downgrades that paper saving to redesign-needed instead
  of actionable savings.

Frame:

- Wider question: how do we reduce total 2026 BTC human execution cost while
  preserving official edge-credit, access, heat, water, bailout, and
  recertification constraints?
- Original question: what high-value route-mapping optimization or approach
  saves significant on-foot miles, effort, or time?
- Narrower question: can the `10A-MS-08` split replace the current `10A` route
  card?
- Updated narrower question: can `10A-MS-13`, reanchored through Avimor Spring
  Valley Creek parking and an explicit West Creeks Edge tie-in, become a
  certifiable `10A` route-card replacement after regeneration and audits?
- Right decision unit: the original route-mapping approach, with the narrower
  `10A` redesign as the next concrete sprint. The narrower candidate is not
  enough because it failed access certification; the wider full-challenge
  framing is useful but too broad to produce the next route-card action.

## Bottom Line

The high-value optimization is not another global shortest path, and it is not
"build the field-day layer" anymore. That layer is done enough to serve as the
execution foundation. The next high-value move is `10A`, with `10A-MS-13` via
Avimor Spring Valley Creek parking as the current best review-only sprint. That
is where the project can save meaningful effort without weakening
segment-credit, parking, or field-packet truth.

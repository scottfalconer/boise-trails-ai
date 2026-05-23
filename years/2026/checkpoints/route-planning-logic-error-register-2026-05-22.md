# Route And Planning Logic Error Register

Date: 2026-05-22

Purpose: consolidate the known logic errors in the 2026 on-foot route plan,
field packet, route-promotion workflow, and planning proof chain. This is a
public-safe register of known errors and guards. It is not a claim of global
optimality over every possible Boise route.

## Current Packet Snapshot

Current generated field-packet data reports:

- Runnable route cards: 45.
- Field-ready route cards: 40.
- Held route cards: 5.
- Official segment accounting: 251 / 251 represented.
- Field-day publication status: blocked by held route cards, led by
  special-management failures.
- Field-day layer: 31 days, 43 loops, 13 multi-start days, 251 / 251 official
  segments covered, 6,617 total p75 minutes, 359 max p90 minutes, 76 total
  between-start drive minutes.

Current held route cards:

| Route | Current status | Logic issue |
| --- | --- | --- |
| `FD04A` | `blocked_special_management` | On-foot route uses Bucktail Trail where the published rule says downhill bike traffic only and foot/equestrian accommodation should use Two Point. |
| `FD15A` / former route `3` | `blocked_special_management` | Same Bucktail on-foot mode violation, plus a cue-anchor mismatch crossing repeated parked-car passes. |
| `FD18A` | `blocked_special_management` | Polecat Loop traversal violates the published clockwise-through-2026 all-user direction rule. |
| `FD23A` / route `12` | `blocked_navigation_source` | Cue anchors span long GPX/map paths across parked-car passes, so cue mileage and field topology disagree. |
| `FD26A` | `blocked_special_management` | Around the Mountain traversal violates the published counter-clockwise all-user direction rule. |

## Status Legend

- `active_blocker`: current generated route or field-day artifact is not
  field-ready until fixed or proven otherwise.
- `repaired_with_guard`: the concrete route issue has been repaired and a
  generator, audit, policy, or documentation guard now exists.
- `open_backlog`: not blocking the current card as a hard gate, but still a
  known route-quality or planning-proof debt.
- `standing_guard`: recurring risk that must be checked whenever the route set
  changes.

## Logic Error Register

| ID | Status | Logic error | Affected route or planning surface | Evidence | Required durable behavior |
| --- | --- | --- | --- | --- | --- |
| `logic_001` | `active_blocker` | BTC official segment/ascent validation was treated as enough even when land-manager special-management rules block the actual car-to-car traversal. | `FD04A`, `FD15A` / route `3`, `FD18A`, `FD26A`. | `docs/field-packet/field-tool-data.json`; `years/2026/checkpoints/special-management-rule-audit-2026-05-22.md`; `years/2026/checkpoints/field-tool-completion-audit-2026-05-06.md`. | Redesign the routes, schedule only inside valid date/direction/use contexts, or prove with fresher source evidence that the rule changed. Keep the special-management audit in the certification chain. |
| `logic_002` | `active_blocker` | A cue sequence was treated as navigable even when a cue's displayed mileage and GPX/map span describe different physical movement and cross a parked-car pass. | Active blocker on `FD23A` / route `12`; also appears on `FD15A` but special management is the primary block there. `FD12A` was repaired by splitting the collapsed source. | `docs/field-packet/field-tool-data.json`; `years/2026/notes/daily-work-log.md`; current `blocked_navigation_source` route-card status. | Hold the route as `blocked_navigation_source`, repair the canonical route topology or split, regenerate GPX/cues/field days from source, and rerun certification. Do not solve this by hiding the live-map banner or changing display only. |
| `logic_003` | `repaired_with_guard` | Source-artifact route truth drift let generated outputs disagree about whether a route should exist. | H1 replaced five Harlow / Avimor split cards, but a later packet was caught with the old split cards restored. | `years/2026/checkpoints/manual-route-map-challenge-2026-05-16/README.md`; `years/2026/checkpoints/harlow-h1-active-packet-certification-2026-05-12.md`; current `docs/field-packet/field-tool-data.json` contains `H1`. | Treat the canonical route source, field packet, GPX, route reviews, and certification checkpoints as one route truth. If a rollback is intentional, supersede the older checkpoint; otherwise regenerate from the source containing the certified replacement. |
| `logic_004` | `standing_guard` | Exact-credit same-anchor review was too narrow to catch route-family, partition, source, or artifact contradictions. | FD14D-style dominance was fixed, but H1 and route-family map challenges required a separate manual/human frame. | `docs/route-review-policy.md`; `years/2026/checkpoints/manual-route-map-challenge-2026-05-16/README.md`; `years/2026/checkpoints/all-route-adversarial-disproof-2026-05-16.md`. | Keep exact-credit dominance review, but follow it with route-family and artifact-source challenges before calling the packet clean. |
| `logic_005` | `repaired_with_guard` | A certified route card was preserved even though an accepted same-credit start materially reduced human footmiles and p75 time. | Original `FD14D` Full Sail start for 36th Street Chute segment `1482`; current repaired card uses the lower N 36th anchor. | `docs/route-review-policy.md`; `docs/BTC_CASES.md`; `years/2026/checkpoints/route-review-fd14d-dev.public.md`. | Require `start_justification`, exact segment-set dominance review, and route/source-hashed waivers for materially longer same-credit starts. |
| `logic_006` | `open_backlog` | The access optimizer chased nearest roads or residential probes instead of first finding a legal, repeatable, cue-able parking anchor. | `10A-MS-08` North Burnt Car / Harlow west probes remain parking-gated; Avimor Spring Valley Creek repair candidates are review-only. | `years/2026/checkpoints/10a-ms-08-access-verification-2026-05-10.md`; `years/2026/checkpoints/certifiable-anchor-repair-audit-2026-05-10.md`; `docs/BTC_CASES.md`. | Search outward for certifiable parking before rejecting or promoting a route. Price connector tax, p75/p90, and cue complexity from that legal anchor. |
| `logic_007` | `standing_guard` | Public-source ambiguity, private/user-reviewed access evidence, and public-safe publication labels were conflated. | H1 / Avimor public page ambiguity was resolved for private field use by user confirmation, while public-source ambiguity remains documented. | `years/2026/checkpoints/public-source-route-reevaluation-2026-05-16.md`; `years/2026/checkpoints/all-route-adversarial-disproof-2026-05-16.md`. | Keep access scope explicit: user-confirmed or private-derived anchors can be valid planning anchors, but public artifacts need public-safe labels and day-of signage/condition checks. |
| `logic_008` | `repaired_with_guard` | Route-local coverage was allowed to hide official segments completed by one GPX but still assigned as new credit to a later card. | `15A-1` completed Shingle `1656`; old `16A-2` still claimed Shingle. Later FD14B and route `12` ownership promotions removed additional duplicate future work. | `years/2026/checkpoints/16a-2-optimization-deep-dive-2026-05-11.md`; `years/2026/checkpoints/15a-16a-route-promotion-2026-05-11.md`; `years/2026/checkpoints/route-card-credit-promotion-2026-05-12.md`; `years/2026/checkpoints/field-latent-credit-audit-2026-05-11.md`. | Run cross-route latent-credit audit after route changes and progress events. Reconcile unclaimed official credit as owned elsewhere, repeat/completed context, route-card promotion, or repair debt before publication. |
| `logic_009` | `open_backlog` | Once official credit/access purpose was satisfied, repeated movement was still defended as if it were needed for credit rather than repriced as ordinary connector movement. | Harrison Hollow field test around Buena Vista / Kemper's Ridge; the old FD12 collapsed source showed high-repeat pressure before being split; Freestone/Military and Cartwright/Polecat families remain map-challenge areas. | `years/2026/field-tests/pre-challenge/2026-05-08-test-03/README.md`; `years/2026/field-tests/pre-challenge/2026-05-21-west-climb/analysis.md`; `years/2026/checkpoints/manual-route-map-challenge-2026-05-16/README.md`. | Separate official new miles, repeat official miles, connector miles, road miles, and deadhead. Re-optimize repeat movement after credit purpose is satisfied, with elevation and direction cost included. |
| `logic_010` | `open_backlog` | Direct-gap fallback or source cue pricing was treated as close to promotable even though the candidate GPX was not rebuilt continuously from named connector splices. | Bogus B1/B2 repair candidates stayed blocked; H1 direct gaps were repaired before promotion. | `years/2026/checkpoints/bogus-b1-b2-gate-repair-audit-2026-05-13.md`; `years/2026/checkpoints/harlow-h1-gate-repair-audit-2026-05-12.md`. | A replacement needs rebuilt continuous nav GPX, field cue sheet, p75/p90, repeat/ownership accounting, and recertification. Source GPX pricing alone is not field readiness. |
| `logic_011` | `repaired_with_guard` | GPX track length was treated as a route-readiness or route-total decision metric. | Earlier selected field-day certification work blocked on nav GPX mileage mismatch instead of route-card distance authority. | `years/2026/checkpoints/field-day-scoped-certification-frame-shift-2026-05-10.md`; `docs/BTC_FIELD_PACKET_REQUIREMENTS.md`. | Route/card `on_foot_miles`, p75/p90, official miles, repeat miles, connector miles, and road miles come from the route calculation. GPX is for navigation geometry, continuity, and coverage, not total-mileage authority. |
| `logic_012` | `repaired_with_guard` | The full route-card inventory audit was treated as the next route-decision queue even when selected field-day blockers were smaller and higher impact. | Field-day layer now reports selected loop certification status, blocked loops, and day-level execution status. | `years/2026/checkpoints/field-day-scoped-certification-frame-shift-2026-05-10.md`; current `docs/field-packet/field-tool-data.json`. | Route decisions start from selected field days, then route-card proof below them. Full inventory cleanup remains backlog unless it affects a selected day, backup, replacement, or redesign target. |
| `logic_013` | `repaired_with_guard` | Calendar/schedule p75 values diverged from certified route-card door-to-door timing. | Post-H1 cleanup repaired four single-loop timing mismatches, including `16A-2` moving from 310/348 to 106/119 p75/p90 when the route card changed. | `years/2026/checkpoints/post-h1-cleanup-calendar-assignment-2026-05-13-report.md`; current field-day summary in `docs/field-packet/field-tool-data.json`. | For single-loop field days, route-card timing is authoritative. Calendar assignment may add transfer/day context, but stale loop timing cannot survive route-card promotion. |
| `logic_014` | `repaired_with_guard` | Weekday/weekend labels were used as capacity proof, and fake date-specific availability windows were tempting to invent. | Post-H1 cleanup explicitly rejected invented availability windows; current field-day layer marks weekday/weekend as context only. | `years/2026/checkpoints/post-h1-cleanup-calendar-assignment-2026-05-13-report.md`; `docs/BTC_LOCAL_REALITY.md`; current `docs/field-packet/field-tool-data.json`. | Use explicit dated availability and hard stops. If those do not exist, expose dated p90 bounds and assumptions instead of inventing real windows. |
| `logic_015` | `repaired_with_guard` | Accepted split/re-park replacements could silently disappear during recalculation because they were treated as private/manual overrides rather than first-class route choices. | Accepted replacements for packages such as `1A`, `4C`, `5`, `15A`, H1, and FD12 split must persist or be explicitly superseded. | `docs/BTC_FIELD_PACKET_REQUIREMENTS.md`; `years/2026/checkpoints/route-specific-exception-audit-2026-05-09.md`; `years/2026/notes/daily-work-log.md`. | Recertification must assert that accepted replacement packages/candidates remain present or are explicitly superseded. Same-day re-park and multi-start outings are first-class candidates. |
| `logic_016` | `open_backlog` | Route-specific code branches encoded reusable route-quality rules. | Most high-priority Harrison and package-specific exceptions were moved to data/config or generic checks; remaining debt includes candidate-specific summary metrics, private-anchor label rewrites, Bogus access policy constants, day-of rule constants, and Shingle diagnostic naming. | `years/2026/checkpoints/route-specific-exception-audit-2026-05-09.md`. | Keep route-specific guards temporary. Move durable behavior into heuristics, data/config, generic generator logic, audits, or diagnostic-only scripts. |
| `logic_017` | `repaired_with_guard` | Planning proof happened at the wrong abstraction layer: coverage or route-efficiency proof was treated as a runnable field-day plan. | Early p90 route proofs covered segments but did not fit strict day bounds; relaxed-drive proof needed field-day artifacts and route-card linkage. | `years/2026/checkpoints/p90-availability-sensitivity-audit-2026-05-06.md`; `years/2026/checkpoints/p90-near-miss-pressure-audit-drive45-n40-2026-05-06.md`; `years/2026/checkpoints/human-executable-field-day-layer-2026-05-10.json`. | Use field-day layer over certified route cards as the execution surface. A schedule or optimizer solution is not a field guide until it links to certified cards, GPX, transfers, p75/p90, and blockers. |
| `logic_018` | `standing_guard` | Evidence scope drift can make stale artifacts look authoritative. | Examples include archived 2025 data, Strava as planning evidence but not official 2026 proof, public sanitized map data versus private canonical map data, and upstream route-experience artifacts before promotion. | `docs/BTC_EVIDENCE_LADDER.md`; `docs/BTC_FIELD_PACKET_REQUIREMENTS.md`; `AGENTS.md`. | State evidence scope before changing route truth. Current-year official BTC data is official segment truth; BTC app/upload workflow is proof; Strava and old planner files are supporting evidence only. |

## Current Route-Family Map Challenge Backlog

These are not all current hard blockers, but they are the remaining places where
the manual map frame found route-shape pressure that exact-credit review alone
does not close:

| Family | Routes | Current interpretation |
| --- | --- | --- |
| Freestone / Military / Three Bears | `FD04A`, `FD15A` / route `3`, `FD20A` | Current hard blockers include Bucktail special-management failures. Separate from that, the family still has overhead and ownership/repeat pressure. |
| Cartwright / Polecat | `FD08A`, `FD08B`, `FD14A`, `FD14B`, `FD18A` | Same-trailhead and direction-rule pressure. `FD18A` is currently blocked by Polecat direction. |
| Hulls / Crestline / 8th Street | `FD19A`, `FD19B`, `FD22B`, `FD23A` / route `12` | Lower Hulls date-use rules and route `12` navigation-source mismatch make this a rule-aware, cue-aware family review, not only a distance review. |
| Dry Creek / Sweet Connie / Sheep | `15A-1`, `16A-1`, `16A-2`, `10B`, `15B` | Latent Shingle credit was repaired, but wet-weather, heat, parking, and same-start bundling still gate field use. |
| Bogus / Pioneer | `FD07A`, `FD07B`, `FD25A`, `FD25B`, `FD26A`, `18` | Mountain-access ratios can be legitimate, but closure/date checks and special-management direction remain hard gates. `FD26A` is currently blocked by Around the Mountain direction. |

## What Not To Reopen Without New Evidence

- Do not treat FD12A as still having the old collapsed cue 09 source defect.
  Current generated data splits Field Day 12 into `FD12A` and `FD12B`; neither
  shows the old `blocked_navigation_source` warning.
- Do not treat H1 as absent from the current packet. Current generated field
  data contains H1, and the all-route disproof says the field packet/map data
  agree on H1 after user-confirmed Avimor access.
- Do not treat high ratio alone as proof of a bad route. The current all-route
  disproof found zero deterministic same-credit dominance failures and zero
  open repeat-optimization hard failures after repaired source state. High
  ratio remains pressure for route-family review, not automatic deletion.
- Do not treat phone completion state, Strava history, or route-card ownership
  promotion as official 2026 BTC progress. Challenge-window activity geometry
  validation remains required.

## Validation Notes

This pass changed documentation only. It did not run the planner, exporter, or
test suite. Validate future route repairs with the normal certification chain:

```bash
python years/2026/scripts/export_mobile_field_packet.py
python years/2026/scripts/field_latent_credit_audit.py
python years/2026/scripts/field_progress_report.py
python years/2026/scripts/field_recertification_report.py
python years/2026/scripts/field_tool_completion_audit.py
python years/2026/scripts/field_route_walkthrough_audit.py
```

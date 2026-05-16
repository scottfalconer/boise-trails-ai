# Field-Day Scoped Certification Frame Shift - 2026-05-10

## Objective

Find the next high-value route strategy after the field-day layer and route
distance authority work. The user clarified that GPX distance matters only if it
pollutes route totals, so this pass looks for the next blocker that actually
affects field-day decisions.

## Frame Decision

`reframe`

Use the selected field-day layer as the certification decision surface. Keep the
full route-card audit as inventory hygiene, but do not let unselected route-card
failures drive the next routing decision.

## Current Frame

The old question was:

> What does the full field-tool audit still fail?

That is now too broad. The active downstream job is:

> Which selected field days can the runner execute, and which selected loops
> block that decision because their route-card source, cue text, GPX existence,
> or day-level handoff is not certified?

## Evidence Checked

Current field-day layer summary from `docs/field-packet/field-tool-data.json`:

- 31 field days.
- 50 loops.
- 14 multi-start days.
- 251 / 251 official segments covered.
- 7,684 total p75 minutes.
- 359 max p90 minutes.
- 11 certified route-card loops.
- 4 selected route-card audit-fix loops.
- 35 selected route-card promotion loops.

Field-day status counts:

| Status | Days | Total p75 | Official mi | On-foot mi |
| --- | ---: | ---: | ---: | ---: |
| `executable_route_card` | 5 | 1,241 | 32.61 | 52.10 |
| `needs_day_gpx_validation` | 2 | 485 | 12.43 | 16.64 |
| `needs_route_card_audit_fix` | 3 | 819 | 19.62 | 36.46 |
| `needs_route_card_promotion` | 21 | 5,139 | 99.78 | 209.98 |

Selected cue/card mileage blockers:

| Route card | Date | Candidate | p75 | Official mi | On-foot mi | Why it matters |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `12` | 2026-06-20 | `block-upper_8th_corrals_sidewinder` | 262 | 7.81 | 12.86 | Blocks one weekend field day. |
| `10B` | 2026-07-01 | `combo-currant-creek-bitterbrush-trail` | 134 | 2.45 | 4.45 | Selected loop has audit-fix blocker inside a broader promotion day. |
| `7` | 2026-07-10 | `block-westside_seaman_veterans` | 127 | 2.25 | 3.77 | Blocks one loop in a two-loop weekday. |
| `16A-2` | 2026-07-11 | `manual-16a-2` | 310 | 5.53 | 14.96 | Blocks one large weekend field day. |

Selected multi-loop days needing day-level GPX validation:

| Date | Loops | p75 | p90 | Official mi | On-foot mi |
| --- | --- | ---: | ---: | ---: | ---: |
| 2026-07-02 | `19` Cervidae + `4B` Scott's Trail | 236 | 268 | 3.30 | 6.52 |
| 2026-07-13 | `16B` Stack Rock Connector + `11` Hawkins | 249 | 283 | 9.13 | 10.12 |

Top selected route-card promotion gaps by p75:

| Candidate | Date | p75 | Official mi | On-foot mi |
| --- | --- | ---: | ---: | ---: |
| `harlows-hollows-connector-ricochet-shooting-range-twisted-spring` | 2026-07-12 | 315 | 2.61 | 13.62 |
| `around-the-mountain-trail` | 2026-06-28 | 279 | 6.64 | 10.17 |
| `harlows-hollows` | 2026-06-21 | 275 | 1.40 | 11.81 |
| `dry-creek-trail` | 2026-07-05 | 270 | 6.97 | 14.64 |
| `three-bears-trail-freestone-ridge` | 2026-07-15 | 255 | 6.72 | 13.10 |
| `polecat-loop-peggys-trail` | 2026-07-14 | 254 | 10.19 | 13.32 |
| `combo-who-now-loop-trail-harrison-ridge-harrison-hollow-kempers-ridge-trail-full-sail-trail-buena-vista-trail-bob-smylie-hippie-shake-trail` | 2026-07-06 | 242 | 7.85 | 10.61 |

Route distance authority guard:

- `jq '[.. | objects | select(has("field_track_miles") or has("source_on_foot_miles") or has("field_mileage_reconciled_from_gpx"))] | length' docs/field-packet/field-tool-data.json` returned `0`.
- `rg -n "gpx_miles|track_length_miles\(|Nav GPX mileage|nav_gpx_mileage_mismatch|route_gpx_mileage_mismatch" years/2026/scripts docs/BTC_FIELD_PACKET_REQUIREMENTS.md docs/field-packet years/2026/checkpoints/field-tool-completion-audit-2026-05-06.* years/2026/checkpoints/human-executable-field-day-layer-2026-05-10.*` returned no matches.

Full field-tool audit status after the GPX-distance removal:

- `python years/2026/scripts/field_tool_completion_audit.py` returned failed, 11 / 13 requirements passed.
- Remaining full-inventory failures are missing verified parked starts, cue/card mileage mismatches, and no-track GPX files for unselected route cards.
- This is useful backlog evidence, but it is not the right queue for deciding which selected field days to repair first.

## Adjacent Frames Checked

- Full route-card inventory cleanup: useful for eventual packet quality, but too
  broad for the next decision because it includes route cards that are not
  selected by the field-day layer.
- Selected audit-fix loops: best first repair queue because four cue/card fixes
  convert three blocked field days and one mixed promotion day without changing
  route distance authority.
- Day-level GPX validation: best second queue because both days already have
  certified route cards; they need multi-loop handoff proof.
- Route-card promotion queue: largest remaining work, but should be ranked by
  selected-day cost and schedule pressure rather than by the whole route-card
  inventory.
- New route redesign: lower priority until selected certification gaps are
  cleared, because redesign can add uncertified surface while the current plan
  already covers 251 / 251 segments.

## Frame Iteration

1. Original frame: inspect the full field-tool audit and fix every failure.
2. Stronger frame: scope certification work to selected field-day loops, because
   that is the current decision unit.
3. Challenge to stronger frame: make sure route totals are not contaminated by
   GPX-derived distance. The current generated payload scan found zero
   GPX-derived distance fields in the decision payload, so the selected-loop
   certification queue is the better next artifact.

Stop reason: a third frame around new route redesign would change the route set
before the selected field-day plan is certified, adding risk before removing the
current execution blockers.

## Adversarial Failure Stories

- The full audit fails on unselected route cards, so the planner spends time
  fixing inventory while selected field days remain blocked by four cue/card
  mismatches.
- A route card's GPX track length looks odd, so the packet blocks a selected day
  even though route totals still come from the route calculation and the GPX is
  only navigation geometry.
- A day with two individually certified route cards is treated as publishable,
  but the runner never gets a day-level handoff showing which GPX to open, where
  to re-park, and how the transfer fits the hard stop.
- Promotion work starts with the easiest card instead of the selected high-p75
  days, leaving the largest weekend and weekday schedule pressure unresolved.
- A new shortcut candidate is explored before the existing selected plan is
  certified, creating another route source that must be reconciled across card,
  GPX, cues, and field-day layer.

## Strategy Improvement

Create and maintain a field-day-scoped certification queue:

1. P0: fix selected cue/card mileage blockers for `12`, `10B`, `7`, and
   `16A-2`. These matter because cue text is a field decision surface, not
   because GPX distance matters.
2. P1: validate day-level GPX/handoff for the selected multi-loop certified
   route-card days on 2026-07-02 and 2026-07-13.
3. P2: promote selected route-card gaps by field-day pressure, starting with
   the highest p75 selected loops and weekend/long-day constraints.
4. Backlog: repair missing parking or no-track GPX failures for route cards that
   are not currently selected, unless one becomes a backup, replacement, or
   high-value redesign target.

## Required Artifact Change

The field-day layer should remain the primary execution artifact. Future
certification summaries should print a selected-field-day queue before the full
route-card inventory queue, so route decisions answer "what can I run on this
day?" before "is every route card in the repository clean?"

## Proof Gaps

- A generated queue/report would make this durable instead of relying on manual
  `jq` inspection.
- The four selected cue/card mismatches still need source repair and
  regeneration.
- The two selected multi-loop days still need day-level GPX validation.
- Promotion gaps still need route-card source, GPX, cues, p75/p90, access, and
  audit proof before publication.

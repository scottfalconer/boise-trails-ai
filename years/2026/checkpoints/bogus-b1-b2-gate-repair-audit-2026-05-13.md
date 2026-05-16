# Bogus B1/B2 Gate Repair Audit

Generated: 2026-05-13T13:17:01Z

Status: `first_pass_repair_stops_keep_current_bogus`

This is a gate-repair audit, not a route-card promotion. It tests whether the post-H1 Bogus B1/B2 template candidates can clear direct-gap, repeat, ownership, cue, current-signage, closure/date, and cost gates enough to justify active route-card work.

## Summary

- Candidates audited: 2
- Promotion candidates after first pass: 0
- Recommendation: `keep_current_bogus_cards_after_first_pass`
- Active packet mutated: `false`
- B3 remains deferred until B1 and B2 are individually clean.

## Candidate Results

| Candidate | Status | Direct Gap | Repeat / Ownership | Real GPX-Priced Cost | Mileage Breakdown | Recommendation |
|---|---|---|---|---:|---|---|
| `B1-simplot-side-bogus-day` | `blocked_keep_current_bogus` | `failed_source_cue_gpx_available_but_candidate_not_rebuilt`; 1.27 mi original, 2 source GPX cue legs priced | `classified_explicit_priced_repeat` | 14.17 mi / 484 p75 / 544 p90 | 0.09 repeat / 3.55 connector / 0.79 road-est. | `stop_first_pass_keep_current_bogus` |
| `B2-pioneer-mores-side-day` | `blocked_keep_current_bogus` | `failed_source_cue_gpx_available_but_candidate_not_rebuilt`; 0.64 mi original, 1 source GPX cue legs priced | `classified_explicit_priced_repeat` | 12.71 mi / 392 p75 / 441 p90 | 0.79 repeat / 5.07 connector / 4.9 road-est. | `stop_first_pass_keep_current_bogus` |

## Gate Notes

### B1-simplot-side-bogus-day

- Hard failures after first pass: continuous_gpx_not_rebuilt_from_named_connector_splice, cue_sheet_not_field_ready_until_gpx_rebuilt
- Post-gate requirements if ever promoted later: field_packet_recertification_not_run
- Direct gaps: 2 original gaps, 2 have source route-card GPX cue legs for pricing, but the candidate GPX is not rebuilt without direct_gap_fallback.
  - Gap to `1713`: 0.35 mi direct fallback -> 0.36 mi source GPX cue from `FD07A` (`source_cue_gpx_available_but_candidate_not_rebuilt`).
  - Gap to `return_to_car`: 0.92 mi direct fallback -> 1.8 mi source GPX cue from `FD25A` (`source_cue_gpx_available_but_candidate_not_rebuilt`).
- Repeat/ownership: `classified_explicit_priced_repeat`; declared owned elsewhere: 1655, 1703, 1721.
- Mileage: 14.17 real GPX-priced on-foot mi; 0.09 repeat mi; 3.55 connector mi; 0.79 road mi estimate (`upper_bound_from_mixed_connector_classes`).
- Cue sheet status: `draft_not_field_ready` (13 cues).
- Around the Mountain/current signage: source confirms counter-clockwise all-users direction; keep day-of signage check as operational gate.
- Closure/date gate: June 18/19 access remains operationally gated by Deer Point stewardship road/trail closure windows; this is not route truth.

### B2-pioneer-mores-side-day

- Hard failures after first pass: continuous_gpx_not_rebuilt_from_named_connector_splice, cue_sheet_not_field_ready_until_gpx_rebuilt
- Post-gate requirements if ever promoted later: field_packet_recertification_not_run
- Direct gaps: 1 original gaps, 1 have source route-card GPX cue legs for pricing, but the candidate GPX is not rebuilt without direct_gap_fallback.
  - Gap to `1732`: 0.64 mi direct fallback -> 2.07 mi source GPX cue from `18` (`source_cue_gpx_available_but_candidate_not_rebuilt`).
- Repeat/ownership: `classified_explicit_priced_repeat`; declared owned elsewhere: 1493, 1750.
- Mileage: 12.71 real GPX-priced on-foot mi; 0.79 repeat mi; 5.07 connector mi; 4.9 road mi estimate (`upper_bound_from_mixed_connector_classes`).
- Cue sheet status: `draft_not_field_ready` (16 cues).
- Around the Mountain/current signage: source confirms counter-clockwise all-users direction; keep day-of signage check as operational gate.
- Closure/date gate: June 18/19 access remains operationally gated by Deer Point stewardship road/trail closure windows; this is not route truth.

## Decision

Stop Bogus promotion after this first pass. B1 and B2 can price source route-card GPX cue legs for the direct gaps, but neither has a rebuilt candidate GPX that removes direct_gap_fallback geometry. Current Bogus route cards should remain active until a later generator can build continuous candidate GPX from the named connectors and pass recertification.

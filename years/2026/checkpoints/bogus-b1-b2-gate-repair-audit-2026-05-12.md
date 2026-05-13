# Bogus B1/B2 Gate Repair Audit

Generated: 2026-05-13T04:42:34Z

Status: `first_pass_repair_stops_keep_current_bogus`

This is a gate-repair audit, not a route-card promotion. It tests whether the post-H1 Bogus B1/B2 template candidates can clear direct-gap, repeat, ownership, cue, current-signage, closure/date, and cost gates enough to justify active route-card work.

## Summary

- Candidates audited: 2
- Promotion candidates after first pass: 0
- Recommendation: `keep_current_bogus_cards_after_first_pass`
- B3 remains deferred until B1 and B2 are individually clean.

## Candidate Results

| Candidate | Status | Direct Gap | Repeat / Ownership | Cost After Named Gap Substitution | Recommendation |
|---|---|---|---|---:|---|
| `B1-simplot-side-bogus-day` | `blocked_keep_current_bogus` | 1.27 mi, 2 named cue repairs but GPX not rebuilt | `classified_explicit_priced_repeat` | 14.27 mi / 488 p75 / 548 p90 | `stop_first_pass_keep_current_bogus` |
| `B2-pioneer-mores-side-day` | `blocked_keep_current_bogus` | 0.64 mi, 1 named cue repairs but GPX not rebuilt | `classified_explicit_priced_repeat` | 11.53 mi / 356 p75 / 400 p90 | `stop_first_pass_keep_current_bogus` |

## Gate Notes

### B1-simplot-side-bogus-day

- Hard failures after first pass: continuous_gpx_not_rebuilt_from_named_connector_splice, cue_sheet_not_field_ready_until_gpx_rebuilt, field_packet_recertification_not_run
- Direct gaps: 2 original gaps, 2 have reusable named route-card cues, but promotion still needs rebuilt continuous GPX.
- Repeat/ownership: `classified_explicit_priced_repeat`; declared owned elsewhere: 1655, 1703, 1721.
- Cue sheet status: `draft_not_field_ready` (13 cues).
- Around the Mountain/current signage: source confirms counter-clockwise all-users direction; keep day-of signage check as operational gate.
- Closure/date gate: June 18/19 access remains operationally gated by Deer Point stewardship road/trail closure windows; this is not route truth.

### B2-pioneer-mores-side-day

- Hard failures after first pass: continuous_gpx_not_rebuilt_from_named_connector_splice, cue_sheet_not_field_ready_until_gpx_rebuilt, field_packet_recertification_not_run
- Direct gaps: 1 original gaps, 1 have reusable named route-card cues, but promotion still needs rebuilt continuous GPX.
- Repeat/ownership: `classified_explicit_priced_repeat`; declared owned elsewhere: 1493, 1750.
- Cue sheet status: `draft_not_field_ready` (16 cues).
- Around the Mountain/current signage: source confirms counter-clockwise all-users direction; keep day-of signage check as operational gate.
- Closure/date gate: June 18/19 access remains operationally gated by Deer Point stewardship road/trail closure windows; this is not route truth.

## Decision

Stop Bogus promotion after this first pass. B1 and B2 still depend on named cue substitutions for direct gaps, but neither has a rebuilt continuous GPX using those connector geometries. Current Bogus route cards should remain active until a later generator can build continuous route-card GPX from the named connectors and pass recertification.

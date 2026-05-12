# Current-Calendar Skip-Ready Promotion Audit

Generated: 2026-05-12T14:00:11Z
Status: `blocked_needs_route_card_claim_promotion`

## Summary

- Skip-ready candidates: 2
- Ready for active menu deletion: 0
- Blocked candidates: 2
- Blocked on-foot savings: 4.39 mi

## Candidates

| Removed route | Status | Blockers | Savings |
|---|---|---|---:|
| 114-3: FD14C | blocked | field_day_layer_still_references_removed_route, source_route_claim_missing_for_promoted_segment, source_wayfinding_still_marks_segment_as_no_new_credit | 1.63 mi |
| 122-1: FD22A | blocked | field_day_layer_still_references_removed_route, source_route_claim_missing_for_promoted_segment, source_wayfinding_missing_promoted_segment_credit_cue, source_wayfinding_still_marks_segment_as_no_new_credit | 2.76 mi |

## Promotion Rule

- A skip-ready route is deletable only after the predecessor route card claims the segment, cues it as credit, removes the later card from the field-day layer, regenerates the packet, and passes recertification.
- A predecessor route that only lists the segment under `official_repeat_segment_ids` is physical evidence, not an executable ownership promotion.

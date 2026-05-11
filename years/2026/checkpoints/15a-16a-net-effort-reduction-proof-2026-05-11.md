# 15A / 16A net effort reduction proof

- Status: `proved_planning_net_effort_reduction`
- Proof level: `planning_menu_repricing`
- Full-menu on-foot miles: 263.98 -> 252.32 (11.66 mi saved)
- Full-menu p75 minutes: 6336 -> 6132 (204 min saved)
- Full-menu p90 minutes: 7111 -> 6882 (229 min saved)
- Official segment coverage: 251 -> 251

## Route Change

- `15A-1` keeps the same GPX/on-foot effort but claims Shingle Creek segment `1656`, which its current GPX already covers end-to-end in ascent direction.
- `16A-2` changes from Shingle Creek + Sheep Camp to the Sheep Camp-only probe for segment `1653`.
- `16A-2` local delta: -11.66 on-foot miles, -204 p75 minutes, -229 p90 minutes.

## Gates

- PASS: `active_packet_segment_count_matches_summary` - Current active menu has 251 unique segments; expected 251.
- PASS: `source_route_already_covers_latent_segment` - 15A-1 activity review lists 1656 as extra completed credit.
- PASS: `source_route_latent_segment_full_credit` - 15A-1 review must cover 1656 end-to-end and in the required direction.
- PASS: `source_route_has_no_missed_planned_segments` - 15A-1 review has no missed planned segments.
- PASS: `replacement_probe_track_valid` - Sheep-only retained segment 1653 has graph and track validation.
- PASS: `replacement_probe_has_timing_and_effort` - Sheep-only probe includes p75/p90 timing and DEM effort fields.
- PASS: `proposed_assignment_preserves_current_unique_coverage` - Moving Shingle to 15A-1 and retaining Sheep in 16A-2 preserves the current official segment set.
- PASS: `proposed_assignment_has_no_duplicates` - Proposed route ownership has no duplicate official segment assignments.
- PASS: `proposed_total_on_foot_lower` - Proposed full menu changes total on-foot miles by -11.66.
- PASS: `proposed_total_p75_lower` - Proposed full menu changes total p75 minutes by -204.

## Scope Boundary

- This proves a planning-level net human-effort reduction against the current full active menu.
- It does not prove official BTC credit until a real challenge-window BTC activity validates the segment.
- It does not promote the active field packet; promotion still needs a source route-card replacement, regeneration, human-validity review, and day-of condition/access checks.

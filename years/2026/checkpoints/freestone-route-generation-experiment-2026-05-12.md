# Freestone Cluster Route Generation Experiment

Generated: 2026-05-12T22:58:58Z

Status: `generated_candidates_not_direct_replacements`

## Summary

- Best generated variant: `nearest-segment-greedy`
- Best variant on-foot: 31.38 mi
- Best variant scaled p75: 649 min
- Current four matched cards: 39.54 mi / 818 p75
- Uncovered current segment IDs if replacing all four: 19

This is a route-generation experiment, not a route-card promotion. The generated GPX covers the template segment set, but the cluster bundle still has uncovered current IDs, so it is not a direct replacement.

## Generated Variants

| Variant | Status | GPX | On-foot | P75/P90 scaled | Connector | Repeat | Coverage | Cue load | Replacement readout |
|---|---|---|---:|---:|---:|---:|---|---:|---|
| `template-sequence-greedy` | `generated_continuous_graph_gpx` | years/2026/checkpoints/freestone-route-generation-experiment-2026-05-12/template-sequence-greedy.gpx | 32.52 | 673/756 | 10.45 | 1.65 | covers_template_segment_set | 20 | not_direct_replacement_needs_additional_loops_or_shrunk_cards |
| `nearest-segment-greedy` | `generated_continuous_gpx_with_direct_gap_fallback` | years/2026/checkpoints/freestone-route-generation-experiment-2026-05-12/nearest-segment-greedy.gpx | 31.38 | 649/729 | 10.29 | 1.33 | covers_template_segment_set | 16 | not_direct_replacement_needs_additional_loops_or_shrunk_cards |

## Current Bundle Gap

- Bundle status: `needs_additional_loops`
- Contained cards fully covered by template: ['FD19C', 'FD20A']
- Partial cards touched but not replaced: ['FD04A', '3']
- Uncovered IDs: 1522, 1529, 1530, 1531, 1548, 1549, 1550, 1551, 1552, 1558, 1574, 1575, 1627, 1628, 1629, 1630, 1631, 1720, 1748

## Gate Notes

- The nearest-segment variant is shorter, but less like the public/common route archetype.
- The template-sequence variant preserves the intended common-route order but is much longer because graph connector links are expensive.
- Next useful work is generating shrink/leftover loops for the 19 uncovered IDs, not promoting this template GPX by itself.

# Route-Specific Exception Audit - 2026-05-09

Purpose: inventory code-level route exceptions that could hide reusable BTC heuristics. This is public-safe and records code paths only, not raw private GPS, exact private parking coordinates, or activity payloads.

Status: open audit with active packet remediation started. The multi-start/re-park recalculation path has been promoted into the active pipeline. `exception_debt_001` through `exception_debt_005` have been removed from route-specific code branches in the active exporter/audit path and converted to data-backed hints or generic checks. The remaining items below are still exception debt or data/config extraction candidates.

## Findings

| ID | Code location | Current behavior | Why it is risky | Needed durable shape | Status | Priority |
| --- | --- | --- | --- | --- | --- | --- |
| `exception_debt_001` | `years/2026/scripts/export_mobile_field_packet.py` old `MANUAL_SIGNPOST_NOTES` | Added Harrison-specific field notes for repeated `#57` / `#51` / `#52` junctions and the `1B` overlap. | The same confusing-junction pattern can happen anywhere a route revisits a signed junction cluster. | Keep field-tested notes as data, but generate caution prompts from repeated junctions, dense signpost clusters, and overlap/exit topology. | Field-tested prose moved to `years/2026/inputs/personal/2026-field-route-hints.json`; generic overlap warnings remain generated from geometry. | Medium |
| `exception_debt_002` | `years/2026/scripts/export_mobile_field_packet.py` old `MANUAL_ACCESS_HINTS` | Hardcoded Harrison Hollow Trailhead -> Who Now access via `#57 Harrison Hollow (AWT)`. | Named non-credit access from car to first official segment is a general field-readiness requirement, not a Harrison-only fact. | Infer named start/return access from the connector graph and signpost labels; if inference is impossible, store route-access hints in data/config rather than code. | Route-specific code removed; fallback access hints now live in data/config and route-line matched names are still added generically. | High |
| `exception_debt_003` | `years/2026/scripts/export_mobile_field_packet.py` old one-route collapsed-package guard | Blocked one known collapsed Package 1 / `block-hillside_harrison_frontside` regression. | Accepted split and multi-start replacements can disappear anywhere after recalculation, not only Package 1. | Certification should generically assert that accepted replacement packages/candidates from the current manifest remain present or are explicitly superseded. | Replaced with accepted replacement manifest preservation keyed by package plus block identity; long single-card outings are allowed unless they supersede an accepted replacement. | High |
| `exception_debt_004` | `years/2026/scripts/export_mobile_field_packet.py` old `apply_route_specific_wayfinding_cautions` | Added a Harrison-specific double-back and exit warning for cue 7 -> 8. | The generic overlap detector now catches repeated-line legs, but overlap exit guidance is still tied to one route label and cue number. | Extend generic overlap analysis to emit exit-transition warnings when an overlap leg is followed by a distinct signed edge. | Replaced with `apply_overlap_exit_wayfinding_cautions`, which keys off `overlap_match` direction and the next cue. | High |
| `exception_debt_005` | `years/2026/scripts/field_tool_completion_audit.py` old Harrison access/return assertions | Checked that `1B` includes `#57 Harrison Hollow (AWT)` before Who Now and on return after Hippie Shake. | The audit proves one route but not the general rule that named non-credit access/return edges must appear in cues. | Compare matched non-credit named edges against cue/step text for every route and fail missing names generically. | Replaced with generic named start-access and return-access cue checks. | High |
| `exception_debt_006` | `years/2026/scripts/multi_start_alternative_audit.py` `west_climb_candidate_found` | Reports whether one West Climb candidate id is found after audit. | Candidate-specific summary metrics can hide whether all accepted replacement classes are preserved. | Replace with generic accepted-candidate / promising-candidate preservation metrics keyed by audit-selected alternatives. | Medium |
| `exception_debt_007` | `years/2026/scripts/export_example_map.py` `private_anchor_label` | Rewrites two private-anchor candidate ids to public-safe labels. | Public sanitization labels are data, not route logic; string matching candidate ids can silently miss new private anchors. | Store public-safe labels in parking-review or replacement metadata and make sanitizer consume that metadata. | Medium |
| `exception_debt_008` | `years/2026/scripts/multi_start_alternative_audit.py` Bogus Basin trailhead whitelist | Restricts Bogus-area components to known lodge/trailhead anchors. | This is a valid local-reality policy, but it currently lives as code constants rather than a sourced local-reality config. | Move place-specific access policies into local-reality data/config with source notes, then have the audit consume that. | Medium |
| `exception_debt_009` | `years/2026/scripts/day_of_preflight.py` named trail rules | Encodes Lower Hulls odd/even, Polecat/ATM/Bucktail signage checks, and low-foothills heat set in code. | These are real local rules, but they should be transparent data-backed local reality, not invisible code-only assumptions. | Extract to a current local-reality rule file or official-condition rule table; keep code as evaluator only. | Low |
| `exception_debt_010` | `years/2026/scripts/p90_repaired_candidate_universe_audit.py` and related Shingle scripts | Keeps a named Shingle exception scenario as a what-if diagnostic. | A named exception can leak into planning if not clearly isolated from active strict-profile routing. | Keep as diagnostic-only unless the user explicitly changes the p90 policy; active planners should require a policy/config gate. | Low |

## Immediate Rule

Route-specific branches are allowed only as temporary safety guards or data-backed local reality. When they encode route-quality behavior, access behavior, promotion behavior, progress behavior, or public artifact behavior, they must be logged here or in a successor audit and then converted into:

- a BTC heuristic or failure mode,
- a data/config rule with source notes,
- reusable generator logic,
- a generic audit or regression test,
- or an explicit diagnostic-only script that cannot mutate the active planner.

## Follow-Up Backlog

1. Move public-safe private-anchor labels to replacement or parking-review metadata.
2. Move Bogus Basin and day-of named-trail policies into local-reality config consumed by preflight/audit scripts.
3. Replace West Climb-specific candidate summary metrics with generic accepted/promising-candidate preservation metrics.
4. Keep Shingle what-if scripts diagnostic-only unless a policy/config gate explicitly promotes them.

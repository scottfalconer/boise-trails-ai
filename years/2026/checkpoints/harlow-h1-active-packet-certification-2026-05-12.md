# Harlow / Avimor H1 Active Packet Certification

Generated: 2026-05-13T03:40:04Z

Status: `certified_after_recertification`

This is the final active-packet checkpoint for H1. It supersedes the source-promotion state in `harlow-h1-route-card-promotion-2026-05-12`, which correctly reported `promoted_to_canonical_route_card_source_pending_recertification` before packet regeneration and gate reruns were complete.

## Result

- H1 route card exists and is certified.
- Removed active route cards: `FD27A`, `FD27B`, `FD27C`, `FD24A`, `FD30A`.
- Active route-card count: 44.
- Official segment accounting: 251/251 represented.
- H1 claimed segment set equals the removed-card union.
- H1 has no direct-gap fallback, hidden self-repeat, unreconciled latent credit, or unpriced repeat.
- H1 parking metadata is present in the public-safe field packet.
- H1 runner-facing cues use named field features rather than opaque OSM connector ids.
- H1 is assigned to `2026-07-04`; `2026-06-21` and `2026-07-12` are reserve/buffer days with `reusable_empty_field_day` status.

## Cost Change

| Scope | Old | New | Savings |
|---|---:|---:|---:|
| On-foot miles | 34.00 | 9.64 | 24.36 |
| p75 minutes | 991 | 289 | 702 |
| p90 minutes | 1117 | 324 | 793 |

## Final Gates

| Gate | Result |
|---|---|
| `harlow_h1_promotion_assertions.py` | passed 18/18 |
| `export_mobile_field_packet.py` | passed, 132 GPX files |
| `field_latent_credit_audit.py` | passed, 44 routes |
| `field_official_repeat_audit.py` | passed, 0 hidden repeat failures |
| `route_repeat_optimization_audit.py` | passed, 0 hidden / latent / unpriced hard failures |
| `field_progress_report.py` | passed, remaining coverage preserved |
| `field_recertification_report.py` | passed |
| `field_tool_completion_audit.py` | passed 15/15 |
| `field_route_walkthrough_audit.py` | passed 44/44 |
| `pytest -q` | passed 506 tests in 123.80s |

## Control-Plane Note

The source-promotion checkpoint remains useful as the record of the source mutation. This checkpoint is the active-packet certification record after regeneration and recertification.

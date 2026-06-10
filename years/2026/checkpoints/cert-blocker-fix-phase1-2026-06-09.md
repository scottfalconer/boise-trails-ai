# Certification Blocker Fixes — Phase 1 (durable hardening) — 2026-06-09

Branch: `cert-blocker-fixes-2026-06-09`. Follows the independent certification
review (`independent-field-packet-certification-review-2026-06-09.md`). Phase 1
is the set of fixes that harden the certification machinery and close the
privacy/test/governance blockers **without changing any route's planned anchor
or mileage**. Phase 2 (route re-anchors + full re-baseline) is held for Scott's
per-route decisions because it changes where he parks and runs.

## Phase 1 — done and verified

### Privacy (blocker)
- Home-address regex no longer ships the literal it guards: assembled at runtime
  from `0x38F` / `int('10010', 2)` in `export_mobile_field_packet.py` and
  `field_tool_completion_audit.py` (obfuscated by request; address is publicly
  listed). Equivalence to the old pattern proven on 11 samples.
- Personal Strava identifiers removed at the generator: dropped the dead
  `example_activity_id/name/date` + `example_strava_segment_id` fields from
  `personal_route_planner.summarize_effort_match` (no downstream consumer).
- Defense-in-depth sanitizer net `strip_private_time_source_fields` in
  `export_example_map.py`; public artifacts re-exported (Strava ids 33→0 in each
  of the 4 public files).
- Audit net: `PRIVATE_STRAVA_PATTERNS` (key-anchored, no coordinate false
  positives) + widened `scan_public_safety` to the public root/example artifacts
  in `field_tool_completion_audit.py`. Verified it catches a synthetic leak and
  passes clean files.
- Bonus real leak found and fixed: the widened scan surfaced a `/Users/scott`
  absolute path embedded in the public map HTML payload (`route_name_source`
  provenance). Root cause: `sanitize_map_html` re-embedded the DATA payload after
  the `remove_local_paths` pass; now scrubs the re-embedded payload too. 1→0.
- Remaining (follow-up, tangential to route cert): 3 committed experiment sim
  files still carry real activity ids/names/dates; referenced by the research
  article bundle. Scrub via surrogate or git-rm + .gitignore.

### Stale committed tests (blocker)
- Group A (4, real-packet label pins) re-pinned to the current 31-card packet:
  `test_special_management_rule_audit.py` FD18A/FD14A → route `5B` (the route
  that now carries the Polecat clockwise rule and absorbed seg 1541/1604);
  forward passes, reverse fails with the exact rule id, and the multi-directional
  exception (seg 1604) is asserted distinctly. `test_route_edge_cover_audit.py`
  depot-phase-reset pin `112-1` → outing `1-2` (Full Sail) with the current
  9-segment set (1579 moved to 1B). Functions renamed off the retired FD labels.
- Group B (8 exporter fixture tests) repaired by densifying degenerate synthetic
  geometry / clearing a spurious car pass so the fixtures are valid under the
  current (correct, un-weakened) exporter gates. [completed via subagent]

### Dual-claim audit guard (blocker: segment 1680)
- New hard check in `field_latent_credit_audit.py`: any official segment that is
  exact credit (in `segment_ids`) for >1 active route fails the audit, with a
  `dual_claimed_exact_credit_segments` report. Verified it flags seg 1680 (The
  Face Trail) claimed by both 17 and 18A. Regression test added. The ownership
  fix itself (which of 17/18A keeps 1680) is phase 2.

### Adversarial disproof registry rubber-stamp (blocker)
- Ran the real deterministic dominance gate against the current 31 routes:
  `build_route_review_pack.py` + `gate_route_reviews.py` → 25 PASS_NON_DOMINATED,
  **6 FAIL_DOMINATED**: `5A, 16A-2, 1A-2, 1A-1, 16C-1, 15B`.
- Rewrote `refresh_all_route_adversarial_disproof.py` to consume that review
  pack (keyed by route label), fail closed on a missing review or a stale review
  (segment-set mismatch) or an unwaived FAIL_* (reusing
  `gate_route_reviews.evaluate_reviews` + the waiver file), derive each route's
  dominance checks from the review instead of hardcoding `True`, and exit
  nonzero. Tests updated to the fail-closed contract.
- Verified end to end: registry now reports 6 failures /
  `route_efficiency_achieved=False`; the repeat-optimization audit correctly
  re-opened 6 warnings (57→51 closed) instead of rubber-stamping all 57.

## Phase 1 validation
- Targeted suites green together: special-management, route-edge-cover,
  latent-credit, all-route-adversarial-disproof, repeat-optimization, efficiency,
  official-repeat, export-example-map, field-tool-completion, gate-route-reviews,
  route-review-pack (96 + 10 tests), plus the 12 previously-red committed tests.
- No route anchor or mileage changed in phase 1.

## Phase 2 — held for Scott's decisions (changes the plan)
The dominance gate's 6 failures plus route 10A and the 1680 owner are genuine
re-anchor-vs-waiver decisions that depend on which trailheads have real parking
Scott will use. Each FAIL says "regenerate from <anchor> OR add a route/
source-hashed waiver". Executing any of them triggers a full canonical
re-baseline (`multi_start_field_menu_replacements → human_loop_plan → export →
reconcile_field_packet_menu_metrics → export_example_map → re-export`), which
also adopts the improved generator's mileage on ~27 routes. Also in phase 2:
real `start_justification` text, the impossible repeat-mileage cue cap,
live-map chevrons/GPS readout, water/heat/bailout annotations, and a parking-
honesty gate (manual_required parking must not ship as field_ready — flags 10A).

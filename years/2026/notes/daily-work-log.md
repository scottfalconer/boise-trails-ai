# 2026 Daily Work Log

This is the short daily log for what we are trying, what changed, and what still
needs proof. It complements the longer planning decision log and the public
field-test logs.

## 2026-06-22 - Route 2 progress-prune cost regression repair

- Objective: fix the route `2-1` regression introduced by pruning completed
  Owl's Roost / Gold Finch / 15th St. / Chickadee Ridge mileage out of the
  physical route. The pruned route was coverage-valid but longer in runnable
  cost than the original car-to-car route.
- Result:
  - Deactivated the route 2 `pruned_component_route` repair and restored the
    lower-cost Lower Hull's Gulch physical route: 18.8 on-foot miles, 332 min
    p75, 372 min p90.
  - Changed outing-menu metric generation so progress-filtered route summaries
    recompute claimed official miles and displayed trail names from remaining
    official segment ids, while completed segments may remain as repeat or
    connector context in the physical route.
  - Added a route-truth repair guard that rejects future `pruned_component_route`
    repairs when they materially increase runnable cost without an explicit
    waiver.
  - Fixed repeat-accounting gates exposed by the restored route: completed
    progress segments are reconciled as completed-at-export repeat context,
    repeat/no-credit text matching recognizes the packet's current wording, and
    long cues with existing repeat ids are reviewed for omitted self-repeat ids.
  - Regenerated canonical private map/menu, sanitized public map/menu, phone
    field packet, GPX bundle, route-review pack, adversarial disproof registry,
    and checkpoint audits from the corrected source.
- Validation:
  - `python3 years/2026/scripts/human_loop_plan.py` passed.
  - `python3 years/2026/scripts/export_example_map.py` passed.
  - `python3 years/2026/scripts/export_mobile_field_packet.py` passed and wrote
    84 GPX files plus the regenerated phone packet.
  - JSON validation passed for the route-truth repair file, public field-tool
    data, public/example/private map data, and current hard-gate checkpoint
    JSON files.
  - Field certification passed: latent-credit audit, progress report,
    recertification report, same-anchor spur-split audit, route-edge cover
    audit, official-repeat audit, route-repeat optimization audit, field-tool
    completion audit, field-route walkthrough audit, and post-credit connector
    audit.
  - Advisory/dependency audits refreshed: route-bridge duplication, latent
    repricing, ownership reassignment, and simulated-progress sweep.
  - `python3 years/2026/scripts/build_route_review_pack.py --all-field-packet-routes --basename route-review-all-dev`
    reviewed 28 routes; `python3 years/2026/scripts/gate_route_reviews.py years/2026/outputs/private/route-reviews/route-review-all-dev.review.json --today 2026-06-22`
    passed with the existing `1A-1` waiver.
  - `python3 years/2026/scripts/refresh_all_route_adversarial_disproof.py`
    passed with 28 / 28 routes accepted and 0 deterministic same-credit
    failures.
  - `python3 -m pytest -q --durations=10` passed with 746 tests passed and 1
    skipped in 2542.79s.
- Current blocker:
  - No known packet/source certification blocker remains. Standard same-day
    trail condition, closure, signage, heat, and water checks still apply before
    running any route.

## 2026-06-21 - Route 2 stale live-map source repair

- Objective: remove the completed Owl's Roost / Gold Finch / 15th St. /
  Chickadee Ridge source legs from live-map route `2-1` after dashboard
  progress made those trails completed credit, while keeping the remaining
  Kestral / Red Cliffs / Crestline / Lower Hulls / Hulls Interpretive route
  certified and publishable.
- Result:
  - Added a generic active `pruned_component_route` route-truth repair path to
    `human_loop_plan.py` and applied it to
    `block-camels_lower_hulls_even_day`.
  - Regenerated the canonical private outing menu, sanitized public map/menu,
    phone field packet, GPX bundle, route-review pack, adversarial disproof
    registry, and certification checkpoints from one route source.
  - Route `2-1` now reads as `Camels Back / Hulls Gulch: Kestrel` and its
    public/private/phone payloads list only Kestral Trail, Red Cliffs,
    Crestline Trail, Lower Hull's Gulch Trail, and Hull's Gulch Interpretive.
    The stale Owl's Roost / Gold Finch / 15th / Chickadee names have zero hits
    in the route `2-1` payloads.
  - Fixed two exporter regressions found during full-suite verification:
    special-management GPX ordering now searches the directed group globally
    instead of greedily, and wayfinding display-mileage sync preserves access
    cue mileage/source snapshots.
- Validation:
  - `python3 years/2026/scripts/human_loop_plan.py` passed and regenerated the
    canonical private outing map/menu.
  - `python3 years/2026/scripts/export_mobile_field_packet.py` passed and
    wrote 84 GPX files plus the regenerated phone packet.
  - Full field-packet chain passed: latent-credit audit, progress report,
    recertification report, same-anchor spur-split audit, route-edge cover
    audit, field-tool completion audit, field-route walkthrough audit, and
    post-credit connector audit.
  - `python3 years/2026/scripts/build_route_review_pack.py --all-field-packet-routes --basename route-review-all-dev`
    reviewed 28 routes; `python3 years/2026/scripts/gate_route_reviews.py years/2026/outputs/private/route-reviews/route-review-all-dev.review.json --today 2026-06-21`
    passed with the existing valid `1A-1` waiver.
  - `python3 years/2026/scripts/refresh_all_route_adversarial_disproof.py`
    passed with 28 / 28 routes accepted and 0 deterministic same-credit
    failures.
  - JSON validation passed for the route-truth repair file, public field-tool
    data, manifest, and public/example map-data JSON files.
  - `git diff --check` passed.
  - `python3 -m pytest -q --durations=10` passed with 741 tests passed and 1
    skipped in 1941.96s.
- Current blocker:
  - No known packet/source certification blocker remains in the generated field
    packet or public map. Standard same-day condition, closure, signage, and
    heat checks still apply before running any route.

## 2026-06-21 - Final same-anchor spur proof hardening

- Objective: tighten the proof that no active field-packet route still has the
  Sheep Camp / Peace Valley failure class: a materially cheaper same-anchor
  spur left as a separate outing.
- Result:
  - Hardened `same_anchor_spur_split_audit.py` so disconnected same-anchor
    bundles only become manual-review advisories when they have material
    estimated savings and host-route contact evidence. The audit checkpoint now
    records both blocking findings and manual-review advisories explicitly.
  - Added regression tests for disconnected same-anchor candidates with and
    without material savings.
  - Fixed repeat-credit label ordering in `export_mobile_field_packet.py` so
    generated route text is deterministic and compact (`segments 1-4`, `6-8`)
    instead of flipping segment order between exports.
- Validation:
  - `python3 years/2026/scripts/same_anchor_spur_split_audit.py` passed with 18
    routes, 0 blocking findings, and 0 manual-review advisories.
  - `python3 years/2026/scripts/export_mobile_field_packet.py` passed and wrote
    54 GPX files plus the regenerated phone packet.
  - Full field-packet chain passed: latent-credit audit, progress report,
    recertification report, same-anchor spur-split audit, route-edge cover
    audit, field-tool completion audit, field-route walkthrough audit, and
    post-credit connector audit.
  - `python3 years/2026/scripts/build_route_review_pack.py --all-field-packet-routes --basename route-review-all-dev`
    reviewed 18 routes; `python3 years/2026/scripts/gate_route_reviews.py years/2026/outputs/private/route-reviews/route-review-all-dev.review.json`
    passed with the existing waived `1A-1` dominance finding.
  - `python3 -m pytest -q years/2026/tests/test_same_anchor_spur_split_audit.py years/2026/tests/test_export_mobile_field_packet.py::test_official_segment_credit_label_orders_same_trail_segment_numbers`
    passed 6 tests.
- Current blocker:
  - No known same-anchor spur-split blocker or advisory remains in the active
    18-route field packet.

## 2026-06-20 - BTC dashboard progress sync and map update

- Objective: pull the current BTC dashboard progress after Owl's Roost, 15th
  St., Gold Finch, and Chickadee Ridge were completed, then update the active
  maps without publishing stale route-source directions.
- Result:
  - Captured a fresh read-only BTC dashboard snapshot under ignored private
    inputs. The dashboard reported 19 completed official segments, 6.889
    official miles, and 4.3329% complete.
  - Joined the dashboard `CompletedSegmentIds` against the June 13 official
    foot segment GeoJSON. All 19 ids matched current official segments.
  - Applied only the six new dashboard ids beyond the existing private ledger:
    `1481`, `1517`, `1518`, `1567`, `1568`, and `1596`.
  - The first packet export correctly refused to publish because the
    post-progress Camel's Back / Hulls Gulch route `2` had stale cue/GPX
    anchors after the now-completed Owl's Roost / Gold Finch / 15th St. /
    Chickadee Ridge cluster was removed from new-credit planning.
  - Added `block-camels_lower_hulls_even_day` to the source route-truth
    mismatch hold area, regenerated the canonical outing menu, and reapplied
    progress. Route `2` is now a manual repair hold instead of a runnable
    field card.
  - Regenerated the phone field packet and GPX set with 17 field-ready routes,
    11 manual holds, 0 omitted non-field-ready routes, and the 19 completed
    segment ids embedded in `docs/field-packet/field-tool-data.json`.
- Validation:
  - `python3 -m json.tool years/2026/inputs/personal/2026-manual-route-designs-v1.json >/dev/null`
    passed.
  - `python3 years/2026/scripts/human_loop_plan.py` passed and regenerated the
    canonical private outing map/menu.
  - `python3 years/2026/scripts/field_progress_versions.py apply-day --epoch challenge-2026 --day-id 2026-06-20-dashboard-sync --review-json years/2026/outputs/private/progress/dashboard-review-2026-06-20.json`
    passed, regenerated reports, and wrote 51 GPX files plus the phone packet.
  - `python3 years/2026/scripts/field_latent_credit_audit.py` passed with 17
    routes and 0 routes needing repair.
  - `python3 years/2026/scripts/field_progress_report.py` passed with 19
    completed segments, 231 remaining, and 250 / 250 official segments
    accounted.
  - `python3 years/2026/scripts/field_recertification_report.py --skip-heavy-optimizer`
    ran and reported remaining coverage preserved, original target still
    possible from the menu, and remaining full completion not currently
    field-ready because manual holds remain.
  - `python3 years/2026/scripts/same_anchor_spur_split_audit.py` passed with
    17 routes and 0 findings.
  - `python3 years/2026/scripts/route_edge_cover_audit.py` passed with 17
    routes and 0 failed routes.
  - `python3 years/2026/scripts/field_tool_completion_audit.py` failed 20 / 21
    requirements at the adaptive completion-feasibility gate because
    `remaining_full_completion_feasible=false` while 114 remaining segments are
    held. The generated packet itself had 17 field-ready routes, 0 GPX failures,
    0 cue/map mismatches, and 0 omitted non-field-ready routes.
  - `python3 years/2026/scripts/field_route_walkthrough_audit.py` passed with
    17 / 17 routes.
  - `python3 years/2026/scripts/post_credit_connector_audit.py` passed with 17
    routes, 0 findings, and 2 non-blocking hidden-exit warnings.
  - `python3 -m pytest -q years/2026/tests/test_field_progress_report.py years/2026/tests/test_field_recertification_report.py years/2026/tests/test_export_mobile_field_packet.py years/2026/tests/test_field_tool_completion_audit.py`
    passed 167 tests in 1721.72s.
- Current blocker:
  - Remaining coverage is preserved, but field-ready remaining coverage is
    still incomplete because manual route-source holds remain. The specific
    post-progress blocker for route `2` is now explicit and must be repaired as
    a new canonical Hulls / Red Cliffs route before it returns to the runnable
    field menu.

## 2026-06-20 - Same-anchor spur-split audit and package 8 repair

- Objective: prove whether the Sheep Camp route-selection failure class existed
  elsewhere in the active field packet, specifically small same-parking spurs
  preserved as separate outings after another route already reaches their
  endpoint.
- Result:
  - Added `same_anchor_spur_split_audit.py` with regression tests for
    same-anchor spur splits, different-anchor exclusions, and through-route
    exclusions.
  - The audit found one real remaining case: package `8` had separate Homestead
    cards for Harris Ridge (`8A`) and Peace Valley (`8B`) even though the base
    graph-validated combined route `block-oregon_trail_harris_peace_valley`
    covered `1724`, `1722`, and `1723` in one 4.05 mi card.
  - Removed the stale package-8 split override from the tracked public override
    source and the active private replacement source, then regenerated the
    human-loop plan and phone packet. Package `8` is now one route, and the
    runnable field-packet route count is 18.
  - Fixed exporter repeat-credit cue anchoring so official segments completed
    before export can remain visible as structured repeat mileage without
    being reintroduced as new remaining credit or blocking the route as a
    navigation-source mismatch.
- Validation:
  - `python3 years/2026/scripts/same_anchor_spur_split_audit.py` passed with 18
    routes and 0 findings.
  - `python3 years/2026/scripts/export_mobile_field_packet.py` passed and wrote
    54 GPX files plus the regenerated phone packet.
  - Full field-packet chain passed: latent-credit audit, progress report,
    recertification report, same-anchor spur-split audit, route-edge cover
    audit, field-tool completion audit, field-route walkthrough audit, and
    post-credit connector audit.
  - `python3 years/2026/scripts/build_route_review_pack.py --all-field-packet-routes --basename route-review-all-dev`
    reviewed 18 routes; `python3 years/2026/scripts/gate_route_reviews.py years/2026/outputs/private/route-reviews/route-review-all-dev.review.json`
    passed with the existing waived `1A-1` dominance finding; refreshed
    all-route adversarial disproof reports 18 / 18 current routes accepted.
  - `python3 -m pytest -q years/2026/tests/test_same_anchor_spur_split_audit.py years/2026/tests/test_export_mobile_field_packet.py::test_field_packet_treats_completed_official_geometry_as_repeat_not_new_credit`
    passed 4 tests.
- Current blocker:
  - No known same-anchor spur-split or packet certification blocker remains in
    the generated 18-route field packet. Standard day-of condition, heat,
    closure, and signage checks still apply.

## 2026-06-20 - Repair Sheep Camp / Dry Creek route selection

- Objective: answer whether Sheep Camp Trail 1 should stay as its own package
  16 outing after the Dry Creek / Shingle route-truth repair.
- Result:
  - Confirmed the generated `16A-D1` GPX only touched Sheep Camp `1653` at the
    Dry Creek junction before repair, so the map was showing the selected route
    accurately but the route selection was stale.
  - Repaired the package-16 route source so `16A-D1` clears Sheep Camp as a
    mid-route spur from the Dry Creek junction, then continues to Shingle and
    returns down Dry Creek.
  - Superseded the standalone Sheep Camp `16A-2` manual card. The field packet
    now exposes `16A-D1` as outing `16-3`, with segment ids `1542`, `1543`,
    `1544`, `1545`, `1546`, `1653`, and `1656`.
  - Hardened the route-truth lollipop generator so mid-route spur repeats are
    not misclassified as final return-to-car repeats, and made lollipop handling
    data-driven with `route_truth_lollipop`.
- Validation:
  - `python3 -m pytest -q years/2026/tests/test_human_loop_plan.py years/2026/tests/test_export_mobile_field_packet.py::test_route_truth_lollipop_skips_avoidable_repeat_repair years/2026/tests/test_field_tool_completion_audit.py`
    passed 50 tests.
  - `python3 years/2026/scripts/human_loop_plan.py` regenerated the canonical
    outing menu.
  - `python3 years/2026/scripts/export_mobile_field_packet.py` regenerated 57
    GPX files and the phone packet.
  - `python3 years/2026/scripts/field_activity_review.py --activity docs/field-packet/gpx/audit/dry-creek-16a-d1.gpx --planned-outing-id 16-3 --planned-segment-ids 1542,1543,1544,1545,1546,1653,1656 --output-json /tmp/16a-d1-sheep-repaired-check.json`
    completed all 7 planned segments with 0 missed and 0 partial.
  - Full field-packet chain passed: latent-credit audit, progress report,
    recertification report, route-edge cover audit, field-tool completion
    audit, field-route walkthrough audit, and post-credit connector audit.
  - `python3 years/2026/scripts/build_route_review_pack.py --all-field-packet-routes --basename route-review-all-dev`
    reviewed 19 routes; `python3 years/2026/scripts/gate_route_reviews.py years/2026/outputs/private/route-reviews/route-review-all-dev.review.json`
    passed with the existing waived `1A-1` dominance finding; refreshed
    all-route adversarial disproof reports 19 / 19 current routes accepted.
  - `python3 -m pytest -q years/2026/tests/test_all_route_adversarial_disproof.py years/2026/tests/test_route_review_pack.py years/2026/tests/test_gate_route_reviews.py`
    passed 15 tests.
  - Full `python3 -m pytest -q` was interrupted after 19:10 with 200 passed and
    3 failures observed before interruption. The two stale disproof failures
    were fixed by refreshing route review/disproof artifacts; the remaining
    isolated failure is
    `years/2026/tests/test_export_execution_gpx.py::test_candidate_segments_for_track_reorders_special_management_loop_to_legal_flow`,
    an unrelated Polecat special-management loop-order test.
- Current blocker:
  - No route-selection blocker remains for Sheep Camp / Dry Creek / Shingle.
    Standard day-of condition, heat, closure, and signage checks still apply.
  - The unrelated Polecat special-management route-order unit test remains open
    and needs a focused topology fix; it did not affect the regenerated 19-route
    field packet certification.

## 2026-06-20 - Apply 1B challenge progress

- Objective: review the completed challenge-window `1B` activity and remove
  those official segments from remaining 2026 route planning.
- Result:
  - Pulled a fresh ignored Strava API snapshot for 2026-06-18 through
    2026-06-20 under
    `years/2026/inputs/strava/api-pulls/2026-06-20-challenge-1b/`.
  - Ran the activity matcher against the June 13 official foot segment data.
    The 2026-06-19 evening run completed all 12 planned `1B` segments plus
    extra segment `1755` (`Buena Vista Trail 5`). Segment `1507` was a
    crossing/near-touch only and remains in planning.
  - Applied `challenge-2026 / 2026-06-19-1b` through
    `field_progress_versions.py apply-day`, preserving the locked challenge
    original and updating the ignored private planner state/ledger.
  - Added `years/2026/notes/challenge-progress.md` as the public-safe progress
    tracking document and added a challenge field log at
    `years/2026/field-tests/challenge/2026-06-19-1b/`.
  - Regenerated the phone field packet with completed segments applied. `1B`
    is no longer in manual holds, and `1A-2` no longer claims `1755` as new
    credit.
- Validation:
  - `python3 years/2026/scripts/field_activity_review.py --activity years/2026/inputs/strava/api-pulls/2026-06-20-challenge-1b/activity_details/18991721205.json --planned-outing-id 1-3 --planned-segment-ids 1697,1698,1699,1700,1717,1716,1714,1715,1579,1581,1582,1578 --output-json years/2026/outputs/private/progress/activity-review-2026-06-19-1b.json`
    wrote 13 completed, 1 extra, 0 missed, 0 partial, and 1 near-touch.
  - `python3 years/2026/scripts/field_progress_versions.py apply-day --epoch challenge-2026 --day-id 2026-06-19-1b --review-json years/2026/outputs/private/progress/activity-review-2026-06-19-1b.json`
    updated the versioned day snapshot and regenerated the field packet.
- Current blocker:
  - No fresh BTC dashboard snapshot is present in the repo for this event. The
    local progress ledger is based on Strava geometry and local official-segment
    matching; official BTC dashboard proof should be refreshed separately when
    the dashboard handle or export is available.

## 2026-06-13 - Official map refresh and packet recertification

- Objective: refresh against the latest public Boise Trails Challenge trail API
  before the challenge window and recertify the active field packet after route
  list/geometry drift.
- Result:
  - Added a reusable official puller and captured the June 13 public pull under
    `years/2026/inputs/official/api-pull-2026-06-13/`.
  - Official on-foot challenge data is now 250 segments / 159.0 mi. The May 4
    list had 251 segments / 164.43 mi.
  - Repaired official drift: removed old Stack Rock Connector 1 (`1663`),
    remapped old Stack Rock Connector 2 (`1664`) to current `1762`, synchronized
    Sweet Connie `1667`, and flipped Polecat special-management direction
    overrides for exact official-geometry reversals (`1601`, `1603`).
  - Regenerated public map/menu artifacts and the mobile field packet. Route
    `16C-1` now claims only official segment `1762` and includes an explicit
    Freddy's Stack Rock service-connector checkpoint for the walkthrough gate.
- Validation:
  - `python3 years/2026/scripts/export_mobile_field_packet.py` passed and wrote
    93 GPX files plus regenerated packet HTML/manifest.
  - `python3 years/2026/scripts/field_latent_credit_audit.py` passed.
  - `python3 years/2026/scripts/field_progress_report.py` passed with 250
    remaining available official segments.
  - `python3 years/2026/scripts/field_recertification_report.py` passed with
    remaining full completion feasible.
  - `python3 years/2026/scripts/route_edge_cover_audit.py` passed 31 / 31
    routes.
  - `python3 years/2026/scripts/field_tool_completion_audit.py` passed 20 / 20
    requirements, 31 field-ready routes, 250 / 250 accounted official segments,
    and special-management passed.
  - `python3 years/2026/scripts/field_route_walkthrough_audit.py` passed 31 /
    31 routes.
  - `python3 years/2026/scripts/post_credit_connector_audit.py` passed with 0
    findings and 0 source-gap proof blockers.
  - `pytest -q years/2026/tests/test_pull_official_challenge_data.py years/2026/tests/test_repair_official_data_drift.py years/2026/tests/test_reconcile_field_packet_menu_metrics.py years/2026/tests/test_field_route_walkthrough_audit.py years/2026/tests/test_special_management_rule_audit.py years/2026/tests/test_field_tool_completion_audit.py years/2026/tests/test_route_edge_cover_audit.py years/2026/tests/test_field_progress_report.py years/2026/tests/test_field_recertification_report.py`
    passed 72 tests.
  - `git diff --check` passed; changed JSON/GeoJSON artifacts validated with
    `python3 -m json.tool`.
- Current blocker:
  - No known packet/source certification blocker remains after the June 13
    official-data refresh. Standard day-of condition, closure, heat, and signage
    checks still apply before running any route.

## 2026-06-09

### Phase 3 - apply the human-judgment queue wins + exporter quality

- Objective: "do all" remaining - restore 10A->H1 (the ~12 mi win), exporter
  quality fixes, experiment-file privacy scrub, full re-baseline + verify.
- Result:
  - 10A -> H1 Avimor RESTORED. Reworked the stale promotion infra
    (promote_harlow_h1_route_card.py) to match the replaced route by SEGMENT SET
    instead of the retired field_menu_label, always append the H1 package, and
    parameterize the route-count assertion by the actual removed count. Route
    10A (Harlow's west-access probe, 21.84 mi / manual_required parking) is now
    H1 "Avimor / Harlow: Twisted Spring" at 9.64 mi - ~12 on-foot mi and ~702
    p75 min saved on that route, full 251/251 coverage preserved.
  - 1A-1 FD14D: NOT applied - the accepted FD14D replacement is a policy
    declaration with no captured route geometry (unlike H1's repaired GPX), so
    the shorter shape can't be promoted without regenerating the connector path.
    The 1A-1 waiver stands (reason updated; hash refreshed after re-baseline).
  - Exporter quality: repeat-official cue mileage now capped to the displayed
    leg (the 54 impossible "0.21-mi leg includes 4.56 mi repeat" notes);
    live-map overlap-leg chevrons + Start-GPS distance-to-route readout +
    malformed off-map SVG triangle + stuck "Loading GPX..." fetch guard; honest
    water / heat-exposure / bailout annotations on every card; start_justification
    fallback now emits a machine-detectable placeholder flag (real per-route
    justification TEXT remains a content follow-up).
  - Privacy follow-up: scrubbed real Strava activity ids/names from the 3
    committed experiment sim files (47 ids + 12 names -> surrogates).
  - Re-baselined the full chain (human_loop_plan -> promote_harlow_h1 -> export
    -> reconcile -> export -> export_example).
- Validation:
  - Full 8-command certification chain passes on the H1-promoted packet:
    completion 20/20 + special-management gate passed, walkthrough 31/31,
    latent-credit 0 dual claims, post-credit 0 findings, 251/251 coverage.
  - Dominance gate passes (1A-1 waived); registry 0 unwaived failures, 31 routes.
  - Full pytest suite: see commit.
- Current blocker:
  - 1A-1 FD14D shorter shape needs connector-path geometry generation (waiver
    stands meanwhile). Real per-route start_justification text is a content
    follow-up (placeholders are now flagged). The 8 manual-map-review areas and
    parking-judgment re-anchors remain in the Human-judgment queue.

### Ralph route-optimization loop - iteration 1 (converged)

- Objective: one iteration of the guarded route-optimization loop
  (`years/2026/prompts/ralph-route-optimization.md`): find the highest-value SAFE
  route improvement, or confirm the menu is already gate-optimal.
- Measured (all audits re-run this iteration, all exit 0):
  - Dominance gate: PASSED (0 unwaived failures; 1A-1 waived to 2026-07-18).
  - Efficiency: global optimizer beats nothing (0 dominant solutions); all 47
    route proofs accepted_current; 0 manual holds; human_loop_plan ratio 1.69.
  - Repeat optimization: 0 open warnings (58 proofed-closed), 0 avoidable
    post-credit repeats, 0 hidden self-repeats, 0 unpriced repeats.
  - Latent-credit: 0 dual claims. Edge-cover: 0 hard failures, 0 advisories.
  - Full certification chain re-verified: 251/251 coverage, special-management
    gate passed, completion 20/20, walkthrough 31/31, post-credit 0 findings.
- Result: NO safe, auto-applicable route change exists this iteration. The plan
  is already optimal at the deterministic-gate level (the phase-1/phase-2 work
  closed the real findings). Every remaining improvement needs human judgment or
  the blocked promotion infra (below). Loop converged; no source change made.

### Human-judgment queue (route optimization — needs Scott or an infra fix)

These are the remaining improvements the loop will NOT auto-apply (they need
ground knowledge or a code change, not a route re-selection):

1. **10A -> H1 Avimor** (largest win, ~12 mi / ~71 p75 min). Blocked: the H1
   promotion path matches by `field_menu_label`, which manual routes no longer
   carry. Needs the promotion infra reworked to match by candidate_id/segment
   set. Also fixes 10A/10B missing pace-model p75 (efficiency
   time_estimate_quality problem_count=2). Infra fix, then re-run the loop.
2. **1A-1 -> FD14D shorter shape** (~1.7 mi). Same blocked infra; currently a
   documented waiver. Drop the waiver once the FD14D shape can be applied.
3. **8 areas flagged for manual map review** by the efficiency
   `alternative_challenge` (no better EXACT/superset candidate auto-found, so a
   human must judge whether a better real route exists): block-freestone_three_
   bears_curlew, connector-highlands-trail-dry-creek-trail, block-cartwright_
   peggy_interface, block-bogus_mores_lodge_tempest, block-polecat_core,
   block-upper_8th_corrals_sidewinder, the Table Rock combo, block-cervidae_peak.
4. **2 boundary recombinations** with a better single metric but that do NOT
   dominate current (generated_combo_beats_current_count=0) - human call on
   whether the recombination is worth it.
5. **High non-credit-ratio / same-trailhead routes** (12 / 7 advisory): mostly
   geography-locked necessary grinders; any re-anchor needs parking Scott can
   confirm is real (the route-10A lesson). Not auto-applied.

### Certification blocker fixes - phase 1 (durable hardening)

- Objective: fix the certification blockers from the independent review that do
  NOT change any route's anchor or mileage (privacy, stale tests, dual-claim
  guard, registry rubber-stamp). Branch `cert-blocker-fixes-2026-06-09`.
- Result:
  - Privacy: home-address regex assembled at runtime (obfuscated by request);
    dropped dead Strava `example_*` identifier fields at the generator
    (`personal_route_planner.summarize_effort_match`); added a sanitizer net
    (`export_example_map.strip_private_time_source_fields`) and an audit net
    (`field_tool_completion_audit` `PRIVATE_STRAVA_PATTERNS` + widened
    `scan_public_safety` to the public root/example artifacts). Re-exported the
    public artifacts: Strava ids 33->0 in each of the 4 files. The widened scan
    also caught a real `/Users/scott` path leak in the public map HTML payload
    (re-embed-after-sanitize bug in `sanitize_map_html`); fixed, 1->0.
  - Stale tests: Group A (4) re-pinned to the current 31-card packet
    (FD18A/FD14A -> route 5B with the multidirectional Polecat exception
    asserted; depot-reset 112-1 -> outing 1-2 Full Sail), functions renamed off
    the retired FD labels. Group B (8 exporter fixture tests) repaired by
    densifying degenerate synthetic geometry / clearing a spurious car pass; no
    exporter gate weakened.
  - Dual-claim guard: new hard check in `field_latent_credit_audit` fails when
    one official segment is exact credit for >1 active route; flags seg 1680
    (claimed by 17 and 18A); regression test added.
  - Registry rubber-stamp: ran the real dominance gate
    (`build_route_review_pack` + `gate_route_reviews`) -> 6 FAIL_DOMINATED
    (5A, 16A-2, 1A-2, 1A-1, 16C-1, 15B). Rewrote
    `refresh_all_route_adversarial_disproof` to consume that review pack and
    fail closed (missing/stale review or unwaived FAIL_*), deriving dominance
    checks from the review instead of hardcoding True. The repeat-optimization
    audit now correctly re-opens 6 warnings (57->51 closed) instead of
    rubber-stamping all 57.
- Validation:
  - All 12 previously-red committed tests pass (`12 passed in 260s`); full
    exporter test file `107 passed`. Targeted phase-1 suites green together
    (96 + 10 tests).
  - `years/2026/checkpoints/cert-blocker-fix-phase1-2026-06-09.md`.
  - This commit also lands the in-progress 2026-05-26..28 route-audit working
    session (block/export/walkthrough/planner scripts + tests + checkpoints) so
    HEAD is self-reproducing, per the review's "depends on uncommitted code"
    finding.
- Current blocker:
  - Phase 2 (route re-anchors + full re-baseline) is held for per-route
    decisions: the 6 dominated routes, route 10A (restore accepted H1 Avimor),
    and the seg-1680 owner each need a re-anchor-vs-waiver call that depends on
    real trailhead parking. The certification chain is intentionally red until
    phase 2 (the new guards correctly flag 1680 and the 6 dominated routes).

### Independent field packet certification review

- Objective: independently re-certify the 31-route field packet and current
  certified maps (multi-agent review: 7 audit dimensions, full chain re-run,
  adversarial verification of every blocker/major finding).
- Result:
  - The binding 8-command certification chain passed 8 / 8 on the current
    worktree: 31 / 31 field-ready routes, 251 / 251 official segments, 20 / 20
    completion requirements, 31 / 31 walkthroughs, 96 post-credit connector
    proofs with 0 findings; regeneration drift vs HEAD is timestamp-only.
  - One-route-truth passed with 0 metric diffs across all seven artifacts and an
    intact SHA chain; GPX layer exact at 31/31/31 files.
  - Verdict: runnable but NOT certifiable as accepted. 20 findings confirmed by
    adversarial verifiers, including 5 blockers: a privacy leak in committed
    sanitizer code (details kept out of this log), the adversarial disproof
    registry refresh hardcoding all proofs true with the dominance gate never
    run against the current 31 cards, route 10A shipping a dominated
    parking-uncertified probe start in place of the accepted H1 Avimor card,
    official segment 1680 double-claimed by routes 17 and 18A (165.59 vs 164.43
    official mi), and 12 committed tests failing at the certified HEAD with no
    full-suite run recorded since 2026-05-24.
  - Majors include: 1A-1 recreating the canonical FD14D regression for segment
    1482, all 31 start_justifications being exporter boilerplate, zero verified
    water/heat/bailout annotations for a mid-summer challenge, 54 / 260 cues
    with physically impossible repeat-official mileage notes, and missing
    overlap-leg direction arrows on the live map.
  - Two findings were refuted by verifiers: dated chain checkpoints DO exist
    inside 5e1582f (under stale-dated filenames), and the chain passes even
    under strict HEAD audit code without the uncommitted relaxations.
- Validation:
  - Full checkpoint:
    `years/2026/checkpoints/independent-field-packet-certification-review-2026-06-09.md`.
  - `pytest -q` ran in full: 12 failed, 692 passed, 1 skipped in 2006.85s; all
    12 failures reproduce in a clean worktree at HEAD `5e1582f`.
- Current blocker:
  - Dominance/acceptance layer (registry, start justifications, 10A, 1A-1) and
    the privacy remediation must be resolved before the packet is promoted as
    accepted for the 2026-06-18 window.

## 2026-06-05

### Bob's 4A field-menu drift gate

- Objective: explain and harden the field-packet failure behind `4A` /
  `bobs-trail-urban-connector`, where field-facing mileage and connector
  savings metadata disagreed across the canonical outing menu, phone packet,
  and connector proof audit.
- Result:
  - Added a field-tool completion gate requiring every field-packet route
    record to match the canonical outing-menu component by `candidate_id`,
    official miles, on-foot miles, p75 minutes, and official segment ids.
  - Added a post-credit connector audit failure for stale
    `shortest_repair_savings_miles` metadata when the current shortest legal
    connector proof reports materially different savings.
  - Added a field-tool completion gate requiring the live map to open on the
    first field-visible cue. This catches zero-length start cues that cause the
    live map to default to cue 2 or later.
  - Reconciled the canonical outing-menu route metrics from the generated
    field-packet route truth, regenerated the phone packet / live map / GPX
    bundle, and refreshed the public sanitized outing menu.
  - Bob's 4A is now treated as one example of a systemic drift class, not an
    isolated route-line issue.
- Validation:
  - `pytest -q
    years/2026/tests/test_field_tool_completion_audit.py::test_completion_audit_fails_when_live_map_would_default_to_second_cue`
    passed.
  - `pytest -q years/2026/tests/test_field_tool_completion_audit.py` passed 26
    tests.
  - `pytest -q years/2026/tests/test_field_tool_completion_audit.py
    years/2026/tests/test_post_credit_connector_audit.py` passed 43 tests.
  - `python years/2026/scripts/field_tool_completion_audit.py --output-json
    /tmp/field-tool-completion-audit-bobs-check.json --output-md
    /tmp/field-tool-completion-audit-bobs-check.md` exited nonzero as expected:
    18 / 19 requirements passed, with 27 canonical route metric mismatches.
    Bob's 4A reported `on_foot_miles` field packet 4.58 versus canonical 4.72.
  - `python years/2026/scripts/field_tool_completion_audit.py --output-json
    /tmp/field-tool-completion-audit-live-map-cues.json --output-md
    /tmp/field-tool-completion-audit-live-map-cues.md` exited nonzero as
    expected: 18 / 20 requirements passed, with 27 canonical route metric
    mismatches and 14 live-map default-cue failures.
  - `python years/2026/scripts/post_credit_connector_audit.py --output-json
    /tmp/post-credit-connector-audit-bobs-check.json --output-md
    /tmp/post-credit-connector-audit-bobs-check.md --manifest-json
    /tmp/post-credit-connector-audit-bobs-check-manifest.json` exited nonzero
    as expected: 89 stale connector-savings metadata findings across 30 routes.
    Bob's 4A cue 3 reported 0.25 mi saved while proof found 0.0016 mi; cue 5
    reported 0.22 mi saved while proof found 0.0016 mi.
  - `python years/2026/scripts/export_mobile_field_packet.py` passed and
    regenerated 93 GPX files plus the phone packet and manifest.
  - The full field-packet certification chain passed on the regenerated packet:
    latent-credit audit, progress report, recertification report, route-edge
    cover audit, field-tool completion audit, field-route walkthrough audit,
    and post-credit connector audit.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed with 31 /
    31 field-ready routes, 251 / 251 official segments accounted, 0 canonical
    route metric failures, and 0 live-map default-cue failures.
  - `python years/2026/scripts/post_credit_connector_audit.py` passed with 31
    routes, 96 post-credit connector proofs, 0 shorter connector findings, 0
    stale connector-savings findings, and 0 route-card / GPX mileage
    mismatches.
  - Browser verification on
    `http://127.0.0.1:8765/live-map.html?outing=4-1` showed Bob's 4A opening on
    `Cue 01 -> 02` with the Bob's Trail to Urban Connector active-leg banner
    and no console errors.
- Current blocker:
  - No known packet/source certification blocker remains for the generated
    field packet. Standard same-day condition, closure, and signage checks still
    apply before running any route.

## 2026-05-28

### Stale Proof Registry Cleanup

- Objective: clean up the stale all-route adversarial proof-registry test after
  the field packet moved from the old 49-card public copy to the current
  31-route regenerated packet.
- Result:
  - Refreshed the public/example outing-menu artifacts from the private
    canonical `2026-outing-menu-map-data.json`, so the public sanitized map and
    the phone packet both represent the same 31-route menu.
  - Added `refresh_all_route_adversarial_disproof.py` to regenerate the
    public-safe proof registry from `docs/field-packet/field-tool-data.json`.
  - Refreshed `all-route-adversarial-disproof-2026-05-16` to 31 current route
    proofs keyed by field-packet route code and candidate id.
  - Updated the proof-registry and public-map consistency tests so they check
    current packet synchronization instead of historical FD03A/FD09A/FD14D
    rows.
- Validation:
  - `python years/2026/scripts/export_example_map.py` completed and rewrote the
    public/example map, menu, data, and PNG artifacts.
  - `python years/2026/scripts/refresh_all_route_adversarial_disproof.py`
    completed with 31 routes, 31 proofs, and 0 deterministic same-credit
    failures.
  - `pytest -q years/2026/tests/test_all_route_adversarial_disproof.py` passed
    4 tests.
  - `pytest -q years/2026/tests/test_all_route_adversarial_disproof.py
    years/2026/tests/test_public_source_route_reevaluation.py
    years/2026/tests/test_public_map_artifact_consistency.py
    years/2026/tests/test_export_example_map.py` passed 17 tests.
  - `pytest -q years/2026/tests/test_route_efficiency_audit.py
    years/2026/tests/test_field_official_repeat_audit.py` passed 20 tests.
  - `python years/2026/scripts/route_efficiency_audit.py`,
    `python years/2026/scripts/route_repeat_optimization_audit.py`, and
    `python years/2026/scripts/field_official_repeat_audit.py` completed. The
    repeat optimization audit passed with 31 routes and 57 / 57 optimization
    warnings closed by current proofs; the official-repeat audit passed with
    31 routes.
  - JSON parsing passed for the refreshed proof registry, public map data,
    route-efficiency audit, route-repeat audit, and official-repeat audit.
- Current blocker:
  - No known stale proof-registry blocker remains.

## 2026-05-27

### Post-credit Connector Proof After Kemper/Buena Vista Miss

- Objective: treat the missed ~70 ft Kemper connector saving as a planner/audit
  failure class, not as a one-off mileage issue.
- Result:
  - Reworked route generation so stale inter-segment links are rechecked against
    the currently oriented next segment before they survive into GPX/cue output.
  - Regenerated the field packet. Route `4A` now remains at 2.84 official miles
    but drops to 4.58 on-foot miles, with Bob's Trail -> Bob's repeat overlap ->
    Urban Connector -> Highlands return instead of the earlier longer car-pass
    shape.
  - `post_credit_connector_audit.py` now reports 0 shorter legal post-credit
    connector findings and 0 route-card/GPX mileage mismatches across 96
    post-credit connector proofs.
  - Added route-order normalization so connector/non-credit cues that physically
    complete required official segments are promoted to credit, while later
    duplicate official movement is demoted to explicit repeat/no-new-credit
    mileage. Initial start-access cues are not promoted before any route credit
    has happened.
- Validation:
  - `python -m py_compile years/2026/scripts/export_execution_gpx.py
    years/2026/scripts/block_day_packager.py` passed.
  - `python -m py_compile years/2026/scripts/export_mobile_field_packet.py
    years/2026/scripts/route_repeat_optimization_audit.py` passed.
  - `pytest -q
    years/2026/tests/test_export_execution_gpx.py::test_candidate_track_coordinates_reorients_segment_before_stale_inter_link
    years/2026/tests/test_block_day_packager.py::test_route_cue_reorients_next_segment_before_stale_inter_link`
    passed 2 tests.
  - `pytest -q
    years/2026/tests/test_export_mobile_field_packet.py::test_final_non_credit_cue_extends_to_actual_route_end_even_with_short_source_path
    years/2026/tests/test_export_mobile_field_packet.py::test_non_credit_claimed_repeat_declarations_add_hidden_self_repeat
    years/2026/tests/test_export_mobile_field_packet.py::test_apply_shortest_repairs_to_wayfinding_cues_uses_route_interval_endpoints`
    passed 3 tests.
  - `pytest -q
    years/2026/tests/test_route_repeat_optimization_audit.py::test_hidden_self_repeat_fails_when_non_credit_leg_reuses_claimed_segment
    years/2026/tests/test_route_repeat_optimization_audit.py::test_hidden_self_repeat_review_uses_explicit_source_path_when_present
    years/2026/tests/test_route_repeat_optimization_audit.py::test_unpriced_declared_repeat_fails
    years/2026/tests/test_export_mobile_field_packet.py::test_apply_shortest_repairs_to_wayfinding_cues_uses_route_interval_endpoints`
    passed 4 tests.
  - `python years/2026/scripts/route_edge_cover_audit.py` passed with 31 routes,
    0 hard failures, and 0 phase-reset advisories.
  - `python years/2026/scripts/export_mobile_field_packet.py` passed and wrote
    93 GPX files plus the regenerated phone packet.
  - `python years/2026/scripts/post_credit_connector_audit.py` passed with 31
    routes, 96 post-credit connector proofs, 0 shorter-connector findings, 0
    unproved connector findings, 0 source-gap blockers, and 0 route-card/GPX
    mismatches.
  - `python years/2026/scripts/route_repeat_optimization_audit.py` passed with
    31 routes, 0 hidden self-repeat blockers, 0 latent-credit blockers, 0
    unpriced-repeat blockers, and 0 avoidable post-credit repeat instances.
  - `python years/2026/scripts/field_official_repeat_audit.py` passed with 29
    public repeat cues and 0 repeat cues missing segment IDs or repeat text.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 31 / 31
    routes.
  - `python years/2026/scripts/field_latent_credit_audit.py`,
    `python years/2026/scripts/field_progress_report.py`,
    `python years/2026/scripts/field_recertification_report.py`, and
    `python years/2026/scripts/field_tool_completion_audit.py` passed. The
    completion audit passed 18 / 18 requirements with 31 field-ready routes and
    all 251 official segments accounted.
- Current blocker:
  - No known packet/source certification blocker remains for the generated
    field packet. Standard same-day condition, closure, and signage checks still
    apply before running any route.

## 2026-05-21

### FD12A West Climb Partial Field Check

- Objective: compare today's shortened Strava run against the current `FD12A`
  West Climb / Harrison Hollow field-packet route before collecting separate
  map UI feedback.
- Result: the dense Strava stream stayed on the `FD12A` route line within
  project tolerance through the early planned corridor. The activity matched
  the start access, Who Now Loop, Harrison Ridge, and return-to-start portion
  before stopping instead of continuing into the remaining Harrison Hollow /
  Kemper's Ridge / Full Sail / Buena Vista / Bob Smylie / Hippie Shake work.
- Segment-level review: 8 of 21 planned official segments completed, 13 missed,
  and 5 partial. Completed planned segments were all four Who Now Loop
  segments, both Harrison Ridge segments, Harrison Hollow 1, and Kemper's Ridge
  4. Harrison Hollow 2 and Kemper's Ridge 1/3 were not fully completed.
- Evidence: ignored Strava API pull under
  `years/2026/inputs/strava/api-pulls/2026-05-21-west-climb-field-test/`;
  private review output under
  `years/2026/outputs/private/progress/activity-review-2026-05-21-west-climb.json`.
- Current blocker: resolved into the live-map overlap repair below.

### FD12A Route Efficiency Review

- Objective: re-check whether the full `FD12A` West Climb / Harrison / Full
  Sail route still makes sense after the partial field run made the repeated
  Who Now / Harrison corridor feel wrong.
- Result: keep `FD12A` as currently proven, but treat it as a high-repeat
  optimization candidate. Current route review still reports
  `PASS_NON_DOMINATED`, no same-credit alternatives found, 7.85 official miles,
  10.61 on-foot miles, 2.76 miles of non-credit/repeat burden, p75/p90
  242/272 minutes, and Harrison Hollow Trailhead as the parking/start anchor
  for the exact segment set.
- Repeat review: the repeat audit passes while flagging closed
  `high_declared_repeat_miles` pressure at 4.69 declared repeat miles. The
  repeat-productivity audit currently assigns `FD12A` 0.00 dead-repeat
  candidate miles and 4.69 necessary repeat miles under known legal/start/order
  evidence.
- Planning decision: do not demote the route based on feel alone. A replacement
  needs the same official segment set or a documented same-day split, current
  parking/access proof, complete GPX/cues/DEM timing, and at least 0.25 miles
  or 10 p75 minutes saved.
- Public field-test note:
  `years/2026/field-tests/pre-challenge/2026-05-21-west-climb/`.

### FD12A Live Map Overlap Repair

- Objective: preserve the red / yellow / green route-order signal while making
  overlapping or repeated route corridors readable on the phone map.
- Result: reverted the temporary gray-context experiment, then changed the live
  map renderer to draw repeated physical corridors as display-only schematic
  lanes, following the same separation principle used by transit maps. The
  active-leg and GPS math still uses the true projected GPX geometry; only the
  full-route context and cue-leg backdrop use offset lanes. Follow-up field UI
  feedback tightened the lanes into a more subway-like treatment: adjacent
  9px colored strokes, screen-stable lane spacing while zooming, smoother
  context paths, route-order color chunks instead of many tiny fragments, and a
  shared corridor baseline so repeated passes on the same trail are clipped
  together before visual lane offsets are applied. A second follow-up aligned
  current/next cue markers and cue-leg color breaks to the same `route_miles`
  anchors used by the active blue leg, instead of scaled cue-card mileage. A
  third follow-up moved the active blue display itself onto the schematic lane
  geometry, with arrows and current/next cue anchors sampling that same displayed
  blue geometry; GPS and progress math still use the true GPX. A fourth
  follow-up made the active blue line use the same smooth snapped rendering as
  the context lanes by sampling the smoothed path into route-aware points, so
  the direction arrows and highlighted blue line share one display path. A fifth
  follow-up added a touch-sized close control to the top cue banner so a long
  cue can be hidden when it covers map context; the footer cue card remains
  visible, and the banner reappears for a different active cue.
- Field lesson: route-order color is a functional proximity cue in the field,
  not decoration. When the same trail is reused, the map needs separated lanes
  before it needs a different palette.
- Validation:
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated the
    field packet, GPX bundle, manifest, and private artifact manifest.
  - `node --check` on the generated `docs/field-packet/live-map.html` script
    passed.
  - `python3 -m json.tool docs/field-packet/field-tool-data.json` passed.
  - `python3 -m json.tool docs/field-packet/manifest.json` passed.
  - `python3 -m py_compile years/2026/scripts/export_mobile_field_packet.py`
    passed.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py::test_live_gps_map_is_active_cue_leg_navigation_artifact years/2026/tests/test_export_mobile_field_packet.py::test_live_gps_map_uses_consistent_active_leg_direction_arrows`
    passed 2 tests in 6.54s after the smoothed active-path cleanup.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py::test_live_gps_map_top_cue_banner_can_be_hidden years/2026/tests/test_export_mobile_field_packet.py::test_live_gps_map_default_viewport_is_single_screen_follow_surface`
    passed 2 tests in 5.33s after the cue-banner close cleanup.
  - `pytest -q` passed 561 tests in 119.25s after the cue-banner close cleanup.
  - Playwright loaded
    `http://127.0.0.1:8765/live-map.html?outing=112-1&cue=9&v=cue-banner-close-20260521`
    with one visible `Hide cue banner` button. Clicking it hid the top cue
    banner while preserving the footer cue, route context, active line, and
    arrows; stepping to the next cue showed the banner again.
  - `git diff --check` passed.

## 2026-05-11

### 16A-2 Optimization Deep Dive

- Objective: dig into `16A-2` from the runner/optimization frame after the
  public-route evidence pointed at the Dry Creek / Shingle loop pattern.
- Finding: the optimization is not a better standalone `16A-2` parking anchor.
  The current `15A-1` Dry Creek GPX already covers Shingle Creek segment `1656`
  in the required ascent direction while only claiming Dry Creek credit.
- Evidence:
  `years/2026/checkpoints/15a-1-latent-shingle-credit-review-2026-05-11.json`
  shows `1656` as an extra completed segment for the current `15A-1` GPX, with
  match fraction `1.000`, endpoints covered, and ascent direction passed.
- The current `16A-2` GPX also re-covers Dry Creek `1542`, `1543`, and `1544`
  as extra completed segments, confirming that the current Shingle/Sheep card is
  paying repeat mileage across adjacent cards.
- Follow-up access check: live OSM data shows an `amenity=parking` way at the
  Dry Creek / Sweet Connie start area, OSM way `1328228551`, centered around
  `43.6916536, -116.182042`. This supports the current roadside start; it does
  not change the larger conclusion that `16A-2` should likely become
  Sheep Camp-only after `15A-1` validates Shingle.
- Recommendation: schedule or package `15A-1` before `16A-2`; if the actual BTC
  activity validates `1656`, reduce `16A-2` to the Sheep Camp-only probe
  (`1653`, 3.30 on-foot miles, 106 p75 / 119 p90) instead of running the full
  14.96-mile current card.
- Added checkpoint:
  `years/2026/checkpoints/16a-2-optimization-deep-dive-2026-05-11.md`.
- Validation commands:
  - `python3 years/2026/scripts/field_activity_review.py --activity docs/field-packet/gpx/audit/15a-1-dry-creek-sweet-connie-roadside-parking-dry-creek-trail.gpx --planned-outing-id 15-1 --planned-segment-ids 1542,1543,1544,1545,1546 --output-json years/2026/checkpoints/15a-1-latent-shingle-credit-review-2026-05-11.json` wrote 6 completed, 1 extra, 0 missed, 2 partial.
  - `python3 years/2026/scripts/field_activity_review.py --activity docs/field-packet/gpx/audit/16a-2-dry-creek-sweet-connie-roadside-parking-sheep-camp-trail-shingle-creek-trail.gpx --planned-outing-id 16-2 --planned-segment-ids 1656,1653 --output-json years/2026/checkpoints/16a-2-activity-review-current-route-2026-05-11.json` wrote 5 completed, 3 extra, 0 missed, 3 partial.

## 2026-05-10

### What We Are Attempting

- Look for a high-value route-mapping optimization from the current field
  packet and recent multi-start work, with the emphasis on real human field
  cost rather than map aesthetics.
- Challenge the assumption that every high-ratio route should be shortened as
  a standalone route card.

### Proof Work

- Reviewed current BTC heuristics, local-reality requirements, field-packet
  requirements, the accepted multi-start replacement history, the current
  field-packet route metrics, the rerun multi-start audit, and relaxed-drive
  field-day proof artifacts.
- Confirmed that after already-promoted split/re-park replacements, the active
  route-card-level opportunity is mostly `10A`: the current audit retains two
  Harlow / Hidden Springs alternatives saving 2.48 to 2.86 on-foot miles and
  19 to 27 p75 minutes, but both still need parking/access verification.
- Identified the higher-value system approach: make same-day field-day bundles
  a first-class route-mapping artifact, because the relaxed-drive proof covers
  251/251 official segments in 31 field days with 14 multi-start days and only
  76 total between-start drive minutes.
- Added checkpoint:
  `years/2026/checkpoints/high-value-route-mapping-optimization-2026-05-10.md`.
- Targeted validation for the current planner guard and multi-start audit
  behavior passed:
  `python -m pytest years/2026/tests/test_personal_route_planner.py years/2026/tests/test_multi_start_alternative_audit.py` -
  54 passed in 0.51s.

### Current Blocker

- The `10A` route-card replacement remains blocked on parking/access proof.
- The field-day bundle idea still needs implementation: a generated source
  artifact, phone-packet day mode, and day-level certification that preserves
  per-loop official segment proof.

### Follow-up Implementation

- Added the `Field-day layer over route cards` heuristic to
  `docs/BTC_HEURISTICS.md`, plus supporting failure-mode, case, contrastive
  case, and behavior-eval entries.
- Added `years/2026/scripts/export_field_day_layer.py`, which overlays the
  dated relaxed-drive calendar assignment onto the certified phone route cards.
  Each loop now either links to a certified route card/GPX or is explicitly
  flagged as needing route-card promotion.
- Generated the first field-day layer artifacts:
  `years/2026/checkpoints/human-executable-field-day-layer-2026-05-10.json`,
  `.md`, and `-manifest.json`.
- Current layer result: 31 field days, 50 loops, 14 multi-start days,
  251/251 official segments covered, 76 total between-start drive minutes,
  15 loops matched to certified route cards, and 35 loops still needing
  route-card promotion before publication.
- The manifest verification passed for the generated layer artifacts.
- Wired the field-day layer into the current phone field guide generated by
  `years/2026/scripts/export_mobile_field_packet.py`. `docs/field-packet/index.html`
  now has a `Field Days` tab that lists dated field-day bundles, hides the
  route-card filter UI while active, and links certified loops back to their
  existing route cards and official GPX files.
- Regenerated `docs/field-packet/field-tool-data.json` with a sanitized
  `field_day_layer` object. The embedded layer keeps day/loop metadata,
  segment ids, route-card references, and GPX hrefs, but no field-day-specific
  coordinates.
- UI smoke validation passed against `http://127.0.0.1:8765/index.html`:
  tapping `Field Days` showed the day layer, hid route cards, and retained
  certified route-card plus GPX links.
- Follow-up audit status: `field_route_walkthrough_audit.py` passed 30/30
  routes, but `field_tool_completion_audit.py` failed 12/13 because eight
  active route cards are missing verified parked-start evidence. That is a
  route-certification blocker, not a Field Days UI blocker.

### Connector / Map Repair

- Fixed connector routing so mapped access, between-trail links, and return legs
  avoid the candidate's own required official segments as hidden
  `official_repeat` shortcuts. Explicit out-and-back returns remain visible
  instead of being hidden as mapped connector wins.
- Rebuilt the accepted multi-start replacements for `1A`, `4C`, `5`, and `15A`
  from the fixed planner, then regenerated the human-loop map and phone field
  packet. The rebuilt `1A-2` West Climb / Bob Smylie / Buena Vista / Full Sail
  route has no self-repeat connector IDs against its active required segment
  set.
- Validation: `pytest -q years/2026/tests/test_personal_route_planner.py` passed
  with 33 tests; field-packet JSON validation passed; `export_mobile_field_packet.py`
  regenerated 90 GPX files with zero GPX validation failures.

### Route Pain Index

- Added `years/2026/scripts/route_pain_index.py` to rank current field-guide
  route cards by human pain and actionable route-mapping leverage, using the
  live phone packet, the rerun multi-start audit, and the field-tool completion
  audit.
- Generated `years/2026/checkpoints/route-pain-index-2026-05-10.json`, `.md`,
  and `-manifest.json`.
- Current result after structured access review: `13` is the raw highest-pain
  route card, but `10A` is still the top route-mapping target as
  `certifiable_anchor_redesign`. The retained `10A-MS-08` paper alternative
  saves 3.38 on-foot miles and 43 p75 minutes, but those are now blocked paper
  savings, not actionable unpromoted savings.
- The accepted split routes for `1A`, `4C`, `5`, and `15A` are treated as
  already-promoted savings, not rediscovered route-shortening work. Their
  current blocker is parked-start certification where the field-tool audit still
  reports missing verified parked-start evidence.
- Validation: `python -m pytest years/2026/tests/test_route_pain_index.py`
  passed with 3 tests.
- Followed up on the highest-leverage `10A` question by checking Avimor public
  trail/parking sources, the 2024 Avimor trail map, MTB Project Harlow Hollows
  notes, prior Street View artifacts, and Google Earth Pro imagery for
  `10A-MS-08`. Result: do not promote `10A-MS-08`; North Burnt Car Place remains
  a physically plausible but uncertified residential road-parking probe, and the
  Harlow's / Hidden Springs west probe is not a certified car start. Checkpoint:
  `years/2026/checkpoints/10a-ms-08-access-verification-2026-05-10.md`.
- Codified the broader lesson as `Certifiable parking before closest road`.
  The planner should not stop at the nearest mapped road/residential edge; it
  should also search outward for a park, official lot, amenity parking, event
  meeting point, or other certifiable anchor, then recompute connector mileage
  and p75/p90. This came from reframing `10A` around Foothills Heritage Park /
  Spring Creek parking rather than trying to force the North Burnt / Harlow west
  residential probes to pass.
- Recomputed the concrete Foothills Heritage Park / Avimor Spring Valley Creek
  reframing for `10A`. The heuristic is correct, but the simple mechanical
  collapse does not currently beat the active `10A` card: all `10A` from Spring
  Valley Creek prices at 16.45 on-foot miles, 416 p75 minutes, and 466 p90
  minutes; the `10A-MS-08` partition reanchored to Spring Valley Creek sums to
  about 15.14 on-foot miles and roughly 418 p75 / 469 p90 minutes once duplicate
  home-drive is removed. Checkpoint:
  `years/2026/checkpoints/certifiable-parking-expansion-audit-2026-05-10.md`.
- The resulting route-mapping opportunity is now sharper: add
  certifiable-anchor expansion plus waypoint-constrained connector corridors.
  For `10A`, the remaining possible win is a hand-shaped FHP/Spring Valley
  Creek to Harlow/Burnt Car tie-in GPX, not a blind trailhead swap.
- Implemented that first audit step in
  `years/2026/scripts/multi_start_alternative_audit.py`: the anchor search now
  preserves a nearby certifiable parking anchor even when a closer road probe
  ranks first, and blocked split alternatives get review-only waypoint-corridor
  repair candidates with connector tax, adjusted savings, and promotion gates.
  Generated the focused `10A` checkpoint
  `years/2026/checkpoints/certifiable-anchor-repair-audit-2026-05-10.md`.
  Result: five review-only `10A` redesign candidates survive the connector
  budget/math screen, led by `10A-MS-13` reanchored through Avimor Spring Valley
  Creek parking with 0.73 round-trip connector tax, 2.13 adjusted on-foot miles
  saved, and -6 adjusted p75 minutes. None are field-packet replacements until
  regenerated route source, GPX, cues, p75/p90, and certification audits pass.
  Validation:
  `python -m pytest years/2026/tests/test_multi_start_alternative_audit.py years/2026/tests/test_route_pain_index.py years/2026/tests/test_personal_route_planner.py`
  passed with 59 tests.
- Ran a separate frame-shift pass after confirming the field-day layer already
  exists. New strategy checkpoint:
  `years/2026/checkpoints/route-card-mileage-truth-frame-shift-2026-05-10.md`.
  Correction after user feedback: GPX distance is not a decision source, so it
  should not be a route-readiness blocker. The real invariant is route distance
  authority: route/card `on_foot_miles`, p75/p90, official miles, repeat miles,
  connector miles, and road miles must come from the route distance calculation,
  not GPX-derived track length. Current follow-up is to remove GPX-distance
  mismatch checks while preserving GPX existence, continuity, official endpoint
  coverage, gap honesty, cue/card mileage checks, and route-card-source
  freshness.
- Follow-up frame-shift after the route-distance correction: the next high-value
  strategy is a field-day scoped certification queue, not another full-inventory
  route-card audit pass. Current selected-field-day evidence shows 5 executable
  route-card days, 2 days needing day-level GPX validation, 3 days needing
  route-card audit fixes, and 21 days needing route-card promotion. The P0 fixes
  are the selected cue/card blockers on `12`, `10B`, `7`, and `16A-2`; P1 is
  day-level handoff validation for 2026-07-02 and 2026-07-13; P2 is selected
  route-card promotion ranked by p75/schedule pressure. Checkpoint:
  `years/2026/checkpoints/field-day-scoped-certification-frame-shift-2026-05-10.md`.

## 2026-05-08

### What We Are Attempting

- Review every current field-menu outing for within-outing route-efficiency
  research opportunities: alternate parked starts, trail order, legal
  direction, split/re-park options, personal Strava evidence, public
  trail-report context, and current R2R/Bogus condition notes.
- Keep this as a research agenda, not a field-packet promotion. Any alternative
  still needs parking/legal-access proof, continuous GPX, p75/p90 timing,
  certified cue text, and walkthrough audits before it can replace a runnable
  outing.

### Proof Work

- Multi-start alternative audit: generated a review-only one-transfer split
  audit across the current field menu. It evaluated 24 multi-segment outing
  components, retained 50 alternatives, and found 9 promising alternatives plus
  3 parking-check alternatives in the current checkpoint after correcting the
  Strava-derived parking-anchor policy. Evidence:
  `years/2026/checkpoints/multi-start-alternative-audit-2026-05-08.md`.
- Public-safe per-outing research agenda: created
  `years/2026/checkpoints/outing-efficiency-research-agenda-2026-05-08.md`.
  It covers all 27 current field-menu outings, including single-segment/small
  outings not evaluated by the multi-start split audit.
- Public sources checked for the agenda included Ridge to Rivers condition
  reports, interactive-map entrypoint, wet-weather guidance, heat/best-time
  guidance, area pages for Hillside/Hulls/Military/Polecat/Hawkins/Oregon
  Trail/Table Rock/Bogus, the R2R map PDF, Bogus Basin trail report,
  Recreation.gov 8th Street Trailhead, and BoiseTrails local report pages.
- Local/private evidence stayed privacy-safe in the public checkpoint:
  imported Strava effort counts and parking-anchor counts were used as summary
  evidence, but private exact coordinates and raw activity ids were not written
  to the agenda.
- Corrected the agenda's parking interpretation: private Strava-derived
  parking anchors are evidence the user has actually parked there before. The
  remaining publication work is public-safe naming unless there is specific
  evidence of changed access, ambiguous/private access, or user uncertainty.
  User-reviewed `yes` parking anchors follow the same pattern: treat them as
  accepted unless a concrete new concern appears.
- Reviewed the local `btc-2026-integrated-outing-efficiency-response.docx`
  and verified the key current-source claims against the BTC site, USDA Forest
  Service Deer Point order, R2R area pages, Bogus status, and local closure
  reporting. Result: the DOCX helps by adding gates and rejection criteria,
  but it does not change the core ranking. Integrated updates into
  `years/2026/checkpoints/outing-efficiency-research-agenda-2026-05-08.md`:
  challenge-window sequencing, Deer Point first-two-day closure handling,
  Bogus amenities constraints, BTC app/GPS-upload proof posture, Polecat
  clockwise-through-2026 handling, Cartwright construction fallback, and
  explicit split rejection criteria.
- Resolved the remaining user-decision questions and codified them in
  `AGENTS.md` plus the outing-efficiency agenda: slower splits are acceptable
  when they provide bailout/logistics value; legal residential road starts are
  acceptable after verification; Bogus lodge/facilities are not needed and
  should not block otherwise-open Bogus routes; the user will use the tested
  BTC app workflow for official recording.
- Redid the high-yield analysis with those settled answers. Result: primary
  design candidates are `17`, `5`, `13`, and `15A`; `10A` becomes a
  high-upside access-validation candidate for verified legal road starts;
  `19` remains a parking/access verification item;
  `1A` remains a valid slower bailout/heat/foot-mile variant; and `4C` remains
  low priority unless public-safe labeling and cues are easy.
- Official map update recommendation: do not replace canonical official map
  route lines yet. The analysis supports a clear map-update backlog, but the
  alternatives still need regenerated route source, GPX, cue text, p75/p90
  timing, direction evidence, and certification audits before they replace
  `years/2026/outputs/private/2026-outing-menu-map-data.json`. Evidence:
  `years/2026/checkpoints/official-map-update-recommendation-2026-05-08.md`.

### Current Blocker

- The agenda identifies research targets; it does not make route changes. The
  highest-priority unresolved work is now explicit candidate design for `17`,
  `5`, `13`, and `15A`, plus access verification for `10A` and `19`. If a
  `10A` residential/road anchor verifies as public, legal, repeatable, and
  cue-able, it should move into the design queue instead of being replaced by a
  worse designated-trailhead option by default.

## 2026-05-06

### What We Are Attempting

- Tighten the route proof from "coverage exists" to "this can be run as real
  home-to-home field days with one car, legal parked starts, continuous GPX, and
  p90 door-to-door bounds."
- Keep the proof honest when it fails. The first route-efficiency proof was
  useful, but it was too abstract: it could say the route menu was proven while
  still ignoring whether oversized outings fit the user's actual day.
- Use today's one-hour availability as a field-packet/navigation test, not as a
  full official completion attempt.

### Proof Work

- Field-day feasibility proof: current field menu still covers 251/251 official
  segments, but strict p90 field-day feasibility fails because 14 runnable loops
  exceed the largest configured daily bound. Evidence:
  `years/2026/checkpoints/field-day-completion-plan-2026-05-06.md`.
- Existing-candidate p90 gap analysis: all known usable candidates cover
  251/251 overall, but only 222/251 official segments are covered by candidates
  under the current 260-minute p90 bound. Evidence:
  `years/2026/checkpoints/p90-completion-gap-analysis-2026-05-06.md`.
- Single-segment split probe: after fixing a reversed-ascent GPX continuity bug,
  14 of the 29 p90-missing segments became graph-validated, continuous, and
  under the 260-minute p90 bound. Fifteen remain unresolved. Evidence:
  `years/2026/checkpoints/p90-segment-split-probe-2026-05-06.md`.
- Manual Harlow west access probe: resolves 13 Harlow/Spring/Twisted/Whistling
  Pig/Ricochet/Shooting segments under the 260-minute p90 bound, but it remains
  conditional until parking/access is field verified. Evidence:
  `years/2026/checkpoints/manual-access-anchor-probe-harlow-west-2026-05-06.md`.
- Manual Sweet Connie lower access probe: resolves the three Sweet Connie
  segments under the 260-minute p90 bound, but it remains conditional until
  parking/access is field verified. Evidence:
  `years/2026/checkpoints/manual-access-anchor-probe-sweet-connie-lower-2026-05-06.md`.
- Manual Shingle lower access probe: resolves Sheep Camp under the 260-minute
  p90 bound, but Shingle Creek still exceeds the current p90 bound. Evidence:
  `years/2026/checkpoints/manual-access-anchor-probe-shingle-lower-2026-05-06.md`.
- USFS Shingle Creek Trailhead probe: graph/track-valid, but worse for the
  target blockers: Shingle Creek p90 428, Dry Creek p90 314, Sweet Connie p90
  461. It does not solve the strict p90 proof. Evidence:
  `years/2026/checkpoints/usfs-shingle-trailhead-probe-2026-05-06.md`.
- Forced-anchor p90 probe: tested the 15 remaining p90-missing segments against
  the nearest known public, manual, and private Strava-derived parking anchors.
  Evidence:
  `years/2026/checkpoints/p90-forced-anchor-probe-2026-05-06.md`.
- Forced-anchor result: Dry Creek segment `1545` and Sweet Connie segment `1667`
  now have strict field-ready probes under 260 minutes p90.
- Parking/access verification: Avimor Spring Valley Creek / Twisted Spring
  parking is source-verified for the Harlow/Spring cluster, and Dry Creek /
  Sweet Connie roadside parking is source-verified for planning. Evidence:
  `years/2026/checkpoints/parking-access-verification-2026-05-06.md`.
- Updated forced-anchor result: 14 of the 15 remaining target segments now have
  strict field-ready rows under 260 minutes p90. Shingle Creek segment `1656`
  remains the only missing segment.
- Shingle lower-end access check: two closer OSM parking features near the
  Shingle lower endpoint were tested, but the graph-valid routes were worse
  than the lower Dry Creek / Sweet Connie roadside start. The issue is graph
  access: the OSM parking features are physically closer but require 3.91-4.06
  graph-valid access miles to the official lower endpoint. Evidence:
  `years/2026/checkpoints/parking-access-verification-2026-05-06.md`.
- Shingle time audit: fixed a Strava segment-history distance conversion bug,
  regenerated the history, and confirmed the Shingle official segment has a
  prior forward effort at 11.12 min/mi. The p90 failure is not bad segment pace;
  it is access/return burden from legal same-car parking. Evidence:
  `years/2026/checkpoints/shingle-1656-time-audit-2026-05-06.md`.
- Shingle connector-gap audit: inspected OSM/R2R ways and connector graph nodes
  around the closer OSM parking features. No legal short connector was promoted;
  future work should not add a synthetic shortcut without field/source proof.
  Evidence:
  `years/2026/checkpoints/shingle-1656-connector-gap-audit-2026-05-06.md`.
- Repaired candidate-universe audit: merged existing usable candidates,
  segment-split probe rows, and strict field-ready forced-anchor rows. Strict
  p90 coverage is now 250/251, with Shingle Creek `1656` as the only missing
  segment. If the 292-minute Shingle p90 exception is accepted, candidate
  coverage becomes 251/251, but the exact set cover still selects 80 loop
  candidates, so that is not a finished field-day schedule.
  Evidence:
  `years/2026/checkpoints/p90-repaired-candidate-universe-audit-2026-05-06.md`.
- Active completion audit: the current strict field-day proof is explicitly
  `not_complete`, and supersedes the older route-efficiency proof for this
  objective. Evidence:
  `years/2026/checkpoints/field-day-p90-completion-audit-2026-05-06.md`.
- Completion-audit pass: added a prompt-to-artifact checklist mapping every
  active-goal requirement to current evidence. Result is still not complete:
  Shingle `1656` fails the p90 bound, the repaired probe universe is not yet a
  dated field-day plan, and p75 optimization is blocked until feasibility is
  restored.
- Repaired field-day packing audit: even if Shingle gets a non-compliant
  292-minute weekday override, the selected 80-loop coverage set still does not
  pack into the current 31-day availability profile. The selected set has 27
  loops over the 180-minute weekend bound and only 22 weekdays available. This
  means Shingle is the first blocker, but route consolidation / weekday-pressure
  reduction is the next blocker. Evidence:
  `years/2026/checkpoints/p90-repaired-field-day-pack-audit-max4-2026-05-06.md`.
- Joint field-day optimizer: tested route selection and field-day packing
  together instead of fixing the p75 set cover first. Strict current bounds
  still miss Shingle `1656`. After fixing the max-combo-size bug and adding
  bounded connected-drive-chain combos, the wide run generated 31,962 field-day
  candidates for the non-compliant 292-minute Shingle weekday bound but still
  found no feasible schedule; relaxed diagnostics need at least 43 field days
  or 37 weekdays with the current 9-weekend limit. Max-coverage mode covers
  219/251 segments under strict current bounds and 231/251 with the
  non-compliant Shingle 292-minute what-if. Evidence:
  `years/2026/checkpoints/p90-joint-field-day-optimizer-wide-2026-05-06.md`.
- Availability sensitivity audit: tested 8 weekday/weekend p90-bound scenarios.
  Only 360/360 was feasible for 251/251 in the current generated route universe.
  Current bounds remain 217/251 max coverage in the default sensitivity grid
  and 219/251 in the wider direct optimizer; the Shingle 292 weekday floor is
  228/251 in the default grid and 231/251 in the wider direct optimizer;
  292/360 reaches 249/251; 360/360 reaches 251/251 but uses all 31
  days and 7,571 total p75 minutes. Evidence:
  `years/2026/checkpoints/p90-availability-sensitivity-audit-2026-05-06.md`.
- Sensitivity gap target audit: turned the near-miss scenarios into redesign
  targets. The closest non-full scenario, 292-minute weekdays / 360-minute
  weekends, misses only Deer Point Trail segment `1540` and Central Ridge Spur
  segment `1558`, and both have individually generated field-day options. That
  means the near-miss problem is schedule packing / opportunity cost, not that
  those two segments are impossible. Evidence:
  `years/2026/checkpoints/p90-sensitivity-gap-targets-2026-05-06.md`.
- Near-miss pressure audit: diagnosed the 292/360 near-miss as a day-count
  pressure problem. Full coverage with actual 22 weekday / 9 weekend day counts
  is infeasible. With 9 weekends fixed, the generated universe needs 24
  weekdays; with 22 weekdays fixed, it needs 10 weekends. Evidence:
  `years/2026/checkpoints/p90-near-miss-pressure-audit-2026-05-06.md`.
- Near-miss consolidation probe: looked for concrete ways to save those field
  days. It found three under-292 weekday pair consolidations, but all share the
  Shane's Trail singleton, so simple pair consolidation saves at most one
  weekday. The closest weekend-only day to convert into a weekday is the Upper
  8th / Corrals / Sidewinder block at p90 294, only two minutes over the
  weekday bound. Evidence:
  `years/2026/checkpoints/p90-near-miss-consolidation-probe-2026-05-06.md`.
- Relaxed inter-trailhead-drive sensitivity: widened the same 292/360 scenario
  from the default inter-trailhead-drive heuristic to 45 minutes and neighbor
  search 40. That generated a full 251/251, 31-field-day cover with 22 weekdays
  / 9 weekends and total p75 7,684 minutes. This is not the current personal
  bound proof, and it may reintroduce more car-hopping, but it proves the
  near-miss was partly caused by an overly narrow combo-generation heuristic.
  Evidence:
  `years/2026/checkpoints/p90-near-miss-pressure-audit-drive45-n40-2026-05-06.md`.
- Relaxed-drive solution quality audit: summarized the car-hop cost of that
  sensitivity plan. It has 31 field days, 14 multi-start days, 76 total
  between-start drive minutes, only one day over 20 minutes between starts, and
  four days over p90 340 minutes. Evidence:
  `years/2026/checkpoints/p90-relaxed-drive-solution-quality-2026-05-06.md`.
- Relaxed-drive draft field-day plan: exported the p75-min full-cover solution
  into a reviewable 31-field-day list. It covers 251/251 official segments,
  has zero days over the 292/360 p90 bounds, and all selected loop metadata
  reports validation passed. It is still not field-ready because dates and
  day-level multi-loop GPX are not assigned/exported. Evidence:
  `years/2026/checkpoints/p90-relaxed-drive-draft-field-day-plan-2026-05-06.md`.
- Relaxed-drive calendar assignment: assigned the 31 draft field days to the
  2026-06-18 through 2026-07-18 challenge dates, preserving weekday/weekend
  types and placing Lower Hulls on an even day. The assignment audit covers
  251/251, has 0 day-type violations, 0 Lower Hulls even-day violations, and 0
  p90 violations. Evidence:
  `years/2026/checkpoints/p90-relaxed-drive-calendar-assignment-2026-05-06.md`.
- Relaxed-drive GPX readiness audit: checked the 50 selected loop rows for
  stored export geometry. 35 loops are exportable from stored personal/hybrid
  candidate geometry; 12 canonical field-menu loops need explicit phone-packet
  GPX lookup; 3 forced-anchor loops need probe regeneration for coordinates.
  Day-level GPX is not ready. Evidence:
  `years/2026/checkpoints/p90-relaxed-drive-gpx-readiness-audit-2026-05-06.md`.
- Forced-anchor GPX export: regenerated navigation GPX for the 3 forced-anchor
  loops, and those individual GPX tracks pass continuity checks. Evidence:
  `years/2026/checkpoints/p90-forced-anchor-gpx-export-2026-05-06.json`.
- Updated relaxed-drive GPX readiness: after field-packet lookup and
  forced-anchor regeneration, all 50 selected loop rows now have either stored
  geometry or a navigation GPX source. Loop-level GPX availability is no longer
  the blocker; day-level multi-loop GPX validation is.
- Relaxed-drive day-level GPX export: exported 31 dated day GPX files, but the
  validation failed on 5 days because several hybrid combo tracks contain large
  internal geometry gaps. The failed rows are Table Rock / Rock Garden / Tram,
  Two Point / Femrite's / Shane's, Currant Creek / Bitterbrush, the combined
  Hillside-Harrison-Hollow route, and Owl's Roost / Chickadee / 15th / Gold
  Finch. This is useful proof work because it caught a field-reality issue: a
  combo can be schedule-valid while its stitched GPX is not yet safe to follow.
  Evidence:
  `years/2026/checkpoints/p90-relaxed-drive-day-gpx-export-2026-05-06.json`.
- Day-level GPX repair: added a post-build graph-stitch pass for remaining
  candidate-track gaps. Re-exporting the relaxed-drive day GPX now reports 31
  dated GPX files, loop validation passed, day-track validation passed, and
  0 failed days. Evidence:
  `years/2026/checkpoints/p90-relaxed-drive-day-gpx-export-2026-05-06.json`.
- P90 profile acceptance audit: compared the now-validated relaxed-drive draft
  against the active private personal availability profile. Result: the draft
  is not accepted as the active personal plan because it uses 292/360 p90 bounds
  and a 45-minute inter-trailhead drive search, while the active profile is
  260/180 and 20 minutes. Evidence:
  `years/2026/checkpoints/p90-profile-acceptance-audit-2026-05-06.md`.
- Strict-profile max-coverage fallback: extracted the best current 260/180
  field-day plan from the wide joint optimizer. It schedules all 31 challenge
  dates under the active bounds, but covers only 219/251 segments and 122.45
  official miles. Evidence:
  `years/2026/checkpoints/p90-strict-profile-max-coverage-plan-2026-05-06.md`.
- Strict-profile gap recovery targets: classified the 32 segments missing from
  the strict fallback. Only Shingle Creek `1656` has no strict field-day
  candidate under current bounds. The other 31 missing segments have generated
  strict candidates and are missing because of schedule tradeoffs. Evidence:
  `years/2026/checkpoints/p90-strict-profile-gap-recovery-targets-2026-05-06.md`.
- Strict-profile swap audit: forced each missing segment into the 31-day strict
  max-coverage schedule. Result: 10 one-for-one swaps, 21 coverage-loss swaps,
  and 1 no-candidate row. This confirms the strict fallback cannot be improved
  above 219/251 by simply choosing a different existing candidate for one
  missing segment. Evidence:
  `years/2026/checkpoints/p90-strict-profile-swap-audit-2026-05-06.md`.
- Direct optimizer bug fix: `p90_joint_field_day_optimizer.py` accepted
  `--max-combo-size 4` but only generated one-, two-, and three-loop days.
  Added a regression test and fixed combo generation so max4 is real. This is
  why the strict fallback moved from 217/251 to 219/251.
- Connected-chain combo fix: the direct optimizer also treated multi-start days
  as tight cliques, requiring every parked start in a day to be near every other
  start. Real field days only need a reasonable A-to-B-to-C drive chain. Added
  connected nearby-drive-chain generation as an additive path on top of the old
  clique candidates and capped expansion so it stays runnable. This improved
  the non-compliant Shingle-292 pressure from 46 to 43 minimum field days, but
  it did not improve strict current-bounds max coverage beyond 219/251.
- Runtime note: the exact strict swap audit and full availability sensitivity
  audit become too slow on the larger connected-chain universe in an interactive
  pass. The current strict proof boundary should therefore rely on the
  regenerated wide optimizer, strict max-coverage plan, and strict gap-recovery
  audit; the older swap audit remains useful context but was not refreshed
  against the larger connected-chain candidate universe.
- Shingle anchor exhaustive probe: tested Shingle `1656` against all 74 known
  public/manual/private parking anchors, not just the nearest-by-straight-line
  anchors. Result: no under-260-minute graph/track-valid route. The best
  field-ready row is still Dry Creek / Sweet Connie roadside parking at
  292 min p90 / 260 min p75 / 11.88 on-foot mi. Evidence:
  `years/2026/checkpoints/p90-shingle-1656-anchor-exhaustive-probe-2026-05-06.md`.
- Completion decision gate: summarized the current branch point. Keeping strict
  260/180 bounds leaves the best schedule at 219/251. Accepting only the
  Shingle 292-minute exception is not sufficient because the scenario still
  needs 43 field days or 37 weekdays. The only current generated full-clear
  profile is the relaxed 292 weekday / 360 weekend / 45-minute inter-start-drive
  draft, which covers 251/251 but is not accepted as active because it violates
  current strict bounds on 22 days. Evidence:
  `years/2026/checkpoints/p90-completion-decision-gate-2026-05-06.md`.
- Responsible-relaxed profile definition: user converged on a more realistic
  completion envelope where all official segments are still mandatory, partial
  segment credit is still disallowed, but 18 on-foot miles in a day is a fair
  cap. We recorded that as the private
  `responsible_relaxed_18mi_v1` certificate profile: 292-minute weekday p90,
  360-minute weekend p90, 45-minute maximum between parked starts, and no
  unsourced shortcuts/private/no-foot connectors.
- Responsible-relaxed certificate: verified the existing relaxed-drive dated
  plan against that new profile. Result: certificate passed for full required
  segment feasibility. The selected plan covers 251/251 official segments, has
  31 field days, 7,684 total p75 minutes, 315.18 on-foot miles, max day 15.9
  on-foot miles, max p90 359 minutes, max between-start drive 27 minutes, and
  validated day-level GPX continuity. Evidence:
  `years/2026/checkpoints/p90-responsible-relaxed-certificate-2026-05-06.md`.
- Proof-scope clarification: this is now a real feasibility certificate for
  the named responsible-relaxed profile, not a proof that the older strict
  260/180 profile works. It also does not prove global optimality over every
  possible continuous legal-access route. It proves the p75-min selected plan
  over the current generated 91,949-field-day candidate universe, with a
  connector-graph rural-postman lower bound as context only.
- Attachment note: the newly mentioned Strava JSON attachment filenames were
  not visible in the repo, Desktop, Downloads, or `/mnt/data` during this pass.
  The certificate therefore used the repo's existing imported Strava/personal
  state and current generated artifacts. If those uploaded files materialize in
  the workspace later, rerun the calibration/parking-anchor derivations before
  treating this certificate as the final pre-event calibration state.
- Certificate hardening: the first responsible-relaxed certificate still had a
  weak parking proof because 33 of 50 selected loops lacked embedded
  `parking_confidence` even though their trailhead names were familiar. The
  verifier now resolves every selected parked start against the city trailhead
  layer, private planner trailheads, parking-access checkpoint, and private
  Strava-derived anchors. Result: 25/25 unique parked starts verified.
- Same-car loop hardening: the certificate now checks actual GPX loop endpoint
  gaps, not just day-level track continuity. Result: 50 selected loops have max
  endpoint gap 0.0 miles, so the run loops return to their parked starts in the
  exported GPX.
- Objective completion audit: added
  `years/2026/checkpoints/p90-objective-completion-audit-2026-05-06.md`.
  Result: achieved for the named `responsible_relaxed_18mi_v1` proof profile,
  not for the older 260/180 strict profile and not as a global optimum over
  every physically possible continuous route.

### Funny / Important Lesson

The first "proof" was not wrong for what it measured, but it was not based in
the real field definition. It proved there was a mathematically defensible route
menu, not that the user could actually run the full challenge inside real
door-to-door windows. Going forward, a proof is only allowed to graduate if it
also accounts for parking starts, return-to-car loops, GPX continuity, p90 time,
and field-day packing.

The specific regression risk is naming: a file called "completion audit" can
sound final even when it only proves a lower-bound or route-quality subproblem.
The active proof needs to say which field definition it is proving and whether
older proof files are superseded for the current goal.

### Today's Field Test

- Available window: about one hour door-to-door.
- Do not treat this as an official completion attempt. The shortest full outing
  in the current menu is still longer than one hour.
- Recommended test: Harrison Hollow cue micro-test from Harrison Hollow
  Trailhead. Use the phone field packet and Nav GPX only far enough to retest
  the Who Now / Harrison Ridge / Kemper's Ridge decision area, then turn around
  in time to stay inside the hard stop.
- Backup if the window expands to about 90 minutes: run `Scott's Trail` from
  Upper Interpretive Trailhead as the shortest full end-to-end route-card test.

### Current Blocker

The strict p90 proof is still not complete. Shingle Creek segment `1656` remains
over the current 260-minute p90 bound with the tested anchors. The current best
source-verified Shingle route is 292 minutes p90 / 260 minutes p75, so the
remaining blocker is the time bound, not parking verification.

There is also a schedule-shape blocker behind Shingle: the repaired selected
loop set and the direct field-day optimizer both create too much weekday
pressure for the current weekday/weekend availability split. A final proof needs
different route consolidation or different availability bounds, not just a
one-segment Shingle exception.

For a realistic adaptive attempt, the current generated universe suggests the
honest target under strict bounds is closer to 219/251 segments unless routes or
availability change. That gives the field tests a useful job: improve timing,
navigation, and route consolidation enough to move the max-coverage schedule up.

The first tested full-clear sensitivity is 360 minutes on both weekdays and
weekends. Treat that as a feasibility stress test, not the current personal
plan.

The closest tested near-miss is 292-minute weekdays / 360-minute weekends:
249/251 segments, missing Deer Point `1540` and Central Ridge Spur `1558`. Because
both have short individual field-day options, the next proof work should test
whether better grouping can include them without losing higher-value coverage.
The first concrete pressure result is that this near-miss needs either two more
weekday field days, one more weekend-length field day, or route grouping that
removes one to two field days from the generated full-cover schedule.
The first concrete route-design target is: add one generated combo around
Shane's Trail, then review whether Upper 8th / Corrals / Sidewinder is really a
294-minute p90 day or can be made weekday-safe with better routing/timing.
The next planning question is qualitative: whether the 45-minute inter-trailhead
drive sensitivity is acceptable under the user's "avoid random car hops unless
time demands it" preference, or whether we should keep it as proof-only and
continue improving coherent loop grouping.
The first quality read is less bad than feared on driving, but still not a
finished plan: 14 multi-start days means the output needs field-menu review so
it does not become another random-errand schedule.
The current best proof artifact is now a draft, reviewable 251/251 field-day
plan under relaxed 292/360 assumptions, not a completed active-goal proof.
It now also has a deterministic date assignment, but day-level multi-loop GPX
and current-condition checks remain blockers before field use.
The GPX blocker has moved again: selected loop GPX sources are available for all
50 selected loop rows, and the relaxed-drive draft now has validated day-level
GPX. This removes GPX as the blocker for that relaxed draft. The blocker is now
whether the relaxed 292/360 + 45-minute inter-trailhead-drive profile is
acceptable, plus current conditions/signage before field use. The profile audit
quantifies the mismatch as 22 p90 day violations and 1 inter-trailhead-drive
violation against the active 260/180 + 20-minute profile.

If the relaxed profile is not accepted, the honest strict-profile fallback is
219/251 segments. That is useful for pre-challenge testing and adaptive route
progress, but it should not be described as a completion plan.

The strict-profile recovery target list tells us what to work on next:
Shingle `1656` is the only true no-candidate route/access/time redesign target,
while the other 31 missing strict-profile segments need better grouping,
different availability choices, or a different completion target.
The swap audit narrows that further: ten of those 31 can be preference swaps
without lowering total segment count, but none increases strict coverage above
219/251. To move the strict baseline upward, we need multi-segment grouping
improvements, a real Shingle time/access breakthrough, or different bounds.

### Field Tool Update

- The user clarified that "I have 2 hours today" is a valid first-class
  constraint. The phone packet now needs to act as an adaptive field tool, not
  only a static list of route cards.
- Added a public-safe field-tool data contract at
  `docs/field-packet/field-tool-data.json`, generated by
  `years/2026/scripts/export_mobile_field_packet.py` from the canonical outing
  menu payload and the responsible-relaxed certificate summary.
- Wrote the explicit field-tool objective and acceptance gates at
  `years/2026/checkpoints/field-tool-objective-2026-05-06.md`.
- The phone packet now exposes 60, 90, 120, 180, 240, and 360 minute
  door-to-door filters, tracks completed route cards by official segment ids in
  local storage, updates the remaining-segment count on the phone, and surfaces
  a "Best today" recommendation inside the selected time window.
- Route cards and field-tool route rows now surface total climb derived from
  the route cue segment effort fields, so effort is visible at the same point as
  time and distance.
- Route cards and field-tool route rows now include p90 door-to-door time in
  addition to p75, using route cue estimates where present and a conservative
  fallback otherwise.
- Added `years/2026/scripts/field_progress_report.py`. The phone packet can now
  export `boise-trails-progress.json`, and the planner-side report converts
  completed outing ids into official segment ids, subtracts any
  `missed_segment_ids`, writes a private-state patch, and reports whether the
  current field menu still covers every remaining official segment.
- Baseline zero-progress report:
  `years/2026/outputs/private/progress/field-progress-latest.json` reports
  251 official segments, 0 completed, 251 remaining, 251 available from the
  current field menu, 0 missing, and `original_target_still_possible_from_menu =
  true`.
- `export_mobile_field_packet.py` now accepts `--progress-json`, so reviewed
  phone progress can generate a fresh remaining-only phone packet before the
  private planner state is manually edited.
- Added `years/2026/scripts/field_recertification_report.py`. At zero progress,
  it writes `years/2026/outputs/private/progress/field-recertification-latest.json`
  with status `passed`: the certified baseline is loaded, remaining field-menu
  coverage is preserved, and remaining full completion is feasible under the
  selected profile.
- The recertifier now also checks remaining certified-calendar capacity. At
  zero progress it reports 31 scheduled remaining field days and 31 available
  challenge dates.
- The default recertifier is intentionally fast for field use: it checks the
  existing certified baseline plus remaining field-menu coverage. The slower
  `--run-heavy-optimizer` path is available for deeper audits, but the first
  interactive attempt was too slow to use as the default phone-progress command.
- Added source-hash stamping to `docs/field-packet/field-tool-data.json` so the
  phone packet can prove which canonical outing-menu payload it was generated
  from. This directly addresses the earlier stale-artifact regression class
  where one interface could quietly point at different route data than another.
- Fixed a field-readiness gap where manually designed Harrison/West Climb route
  cards showed `0 ft` climb because segment-level DEM fields were missing even
  though route-level DEM effort existed. The phone packet now falls back to
  route-level DEM effort and moving-effort estimates.
- Added `years/2026/scripts/field_tool_completion_audit.py`; the current audit
  passes 10/10 field-tool requirements with 26 runnable route cards and 251/251
  official segment ids represented.
- Added per-outing completion-safety metadata to the phone packet. All 26
  current route cards preserve remaining field-menu coverage after normal
  completion, and the phone "Best today" ranking now prefers completion-safe
  candidates inside the selected time window.

### Route Map Cue Numbering Regression

- Field testing and screenshot review exposed a low-level route-map bug: the
  renderer was treating GPX waypoint/segment labels as primary map marker
  numbers, which can make challenge segment order look like field navigation
  order.
- Fixed the renderer contract so the primary field-map bubbles are ordered
  `NavCue` markers: `1 = start/car`, then route-order decisions, then the final
  return/finish. Segment metadata such as `SEG 7 Harrison Hollow 1` remains
  snap/audit data, not the default visible marker number.
- Added cue-sheet sidecars `nav-cues.json`, `nav-cues.csv`, and `nav-cues.md`
  while preserving legacy `cue-sheet.*` filenames as aliases.
- Added metadata fields for primary marker mode, visible marker numbers,
  snapped/ambiguous waypoint counts, segment waypoint counts, omitted segment
  label counts, and cue-sheet output paths.
- Added a regression test with intentionally misleading `SEG 99` and `SEG 42`
  waypoints to prevent segment numbers from returning as default primary map
  bubbles.
- Fresh Harrison Hollow sample output under `/tmp/route-render-navcue-1b/`
  shows route-order markers `1..7`, with start and finish side-by-side at the
  parked-car location instead of one marker hiding the other.
- Tested image-generation workflows against the Harrison Hollow GPX. The image
  model can make attractive style mockups, but helper-image and coordinate-only
  prompts both drifted on cue placement, cue numbering, or labels. Decision:
  image generation is design inspiration only; deterministic GPX rendering owns
  field navigation.
- Added an `imagegen-helper` renderer profile that deliberately produces a
  clean style-reference image from the true GPX: no segment labels, no cue text,
  no detail boxes, no repeated-pass bubbles, sparse arrows, true route-order cue
  anchors, and a parking marker. Sample:
  `/tmp/route-imagegen-helper-profile/route-overview.png`.

### Field Decision Guide Renderer

- Reframed the map problem from "make the GPX overlay prettier" to "answer the
  field question at a confusing junction." The artifact needed in the field is:
  where am I, what signed trail number/name do I take next, and is this a
  repeated place?
- Added a `napkin` renderer profile as a schematic support map, but kept the
  deterministic renderer responsible for cue numbers, labels, and route-order
  semantics. The image model is still only a style/reference tool.
- Added deterministic field-decision sidecars for napkin/profile renders:
  `field-decisions.html`, `field-decisions.md`, and `field-decisions.json`.
  These are cue-card outputs, not audit maps.
- Changed the visible cue action language to signpost-target first. For
  Harrison Hollow, the real sample now says `TAKE #51`, `TAKE #58`, `TAKE #57`,
  `TAKE #52`, and `TAKE #50` instead of unsafe geometry guesses such as
  `TURN AROUND`.
- Kept geometry-derived turn actions as review context only when confidence is
  low. This is deliberate: a false left/right/turn-around instruction is more
  dangerous than a signpost-oriented `TAKE #58` instruction.
- The Harrison Hollow sample still flags ambiguous waypoint snaps and missing
  elevation because the navigation GPX cue waypoints are not perfect field
  decision anchors and the GPX has no elevation samples. That is an honest
  review state, not a ready-to-trust final field map.
- Sample outputs: `/tmp/route-napkin-1b/napkin-map.png` and
  `/tmp/route-napkin-1b/field-decisions.html`.
- Updated the actual phone packet so the route card section is now `What to do
  next` instead of a generic turn-by-turn heading. Each cue is a numbered,
  tappable decision card; tapping a cue highlights it as the current step.
- Regenerated `docs/field-packet/index.html`, the GPX zip, and service-worker
  cache metadata from the canonical field packet source. The 1B card now keeps
  the key checkpoint warning inline and shows the signed-trail sequence as the
  primary field artifact.
- Added the field-decision cue-card behavior to
  `field_tool_completion_audit.py` so future packet regressions fail the audit
  if the phone page falls back to a generic turn-by-turn section. Current audit:
  11/11 requirements passed, 26 route cards, 251/251 official segments in the
  field menu.
- Fixed a hard-reload usability regression where the phone packet defaulted to
  the `<=2h` filter and hid 24/26 outings, including 1B. The generated packet
  now defaults to `All`; time filters narrow the list only after the user taps
  them.
- Identified another field-readiness miss in 1B: the cue generator jumped from
  the car directly to `#51 Who Now Loop`, even though the real route starts on
  named access trail from Harrison Hollow Trailhead. The phone packet now uses
  trailhead access metadata and explicitly says to start on `#57 Harrison
  Hollow (AWT)` and then use the signed access/connector toward `#51 Who Now`.
  This rule is broader than trails: named roads, paths, and connectors used in
  the GPX should appear as route steps even when they do not count as official
  challenge credit.
- Found the matching return-side miss in 1B: the card implied that finishing
  `#50 Hippie Shake` meant the user was already back at the car. That is not
  field-safe. The packet now computes non-credit start and return gaps from
  the ordered route geometry and official segment endpoints. When the final
  official segment does not end at the parked car, the return card must describe
  a connector/access/road leg instead of saying the user is already back at the
  parking point. For 1B this produces an explicit return step back toward `#57
  Harrison Hollow (AWT)` / Harrison Hollow Trailhead.
- While investigating, noticed the underlying 1B source geometry still has a
  `source_gap_warning_count` and emits multiple GPX track segments. Treat this
  as a remaining route-design issue: the cue wording is safer, but 1B should be
  rebuilt as a clean single car-to-car navigation route before challenge use.
- Follow-up test audit: the first return fix was too route-specific. Added
  generic unit coverage for both start and return non-credit gaps computed from
  geometry, plus generic completion-audit failures when any route hides a
  required start-access or return-access leg. The 1B-specific assertions now
  remain as known regression guards, not as the only protection.

### Field-Executable Contract Hardening

- Reviewed the broader test-coverage critique and agreed with the core failure
  mode: existing checks often proved segment-id accounting or rendered GPX
  shape, not whether a person can follow a continuous, legal, car-to-car route.
- Added generic regression coverage for provisional progress, blocked-route
  suppression, p90 hard-stop filtering, official MultiLineString rejection,
  OSM access restriction filtering, multi-segment route reversal, unstitched
  source gaps, inter-`trkseg` gaps, source-gap audit failures, and Nav
  GPX-vs-official endpoint coverage.
- Changed phone progress accounting so `completed_outing_ids` are provisional
  UX state. The private state patch no longer promotes an outing to completed
  official segments unless validated segment ids are supplied from GPS/activity
  evidence.
- Tightened connector snap defaults from 0.2 mi to 0.02 mi for field-published
  routing, and preserved raw OSM access fields on connector edges so private /
  no-foot connectors can be rejected at graph-build time instead of only by
  exporter string checks.
- Changed rendered map validation so splitting a route into valid-looking
  rendered parts can no longer hide a source gap. A `source_gap_warning` now
  fails the field-executable contract unless the gap is explicitly represented
  as a named connector, road connector, official repeat, re-park boundary, or
  manual hold.
- Regenerated the phone field packet with the stricter validators. The first
  pass correctly failed instead of pretending readiness: the audit exposed one
  invalid Homestead / Harris Ridge / Peace Valley route whose rendered
  `MultiLineString` split hid missing continuous field navigation.
- Replaced that Package 8 block with two explicit Homestead outings from the
  graph-validated personal route menu: `8A. Harris Ridge Trail` at 118 minutes
  door-to-door / 4.44 mi on foot, and `8B. Peace Valley Overlook` at 101
  minutes door-to-door / 2.70 mi on foot. This preserves the 251/251 segment
  menu while making the two real 2-hour-ish choices visible instead of one
  broken 3h39m stitched card.
- Regenerated the canonical private menu, public example map/menu, phone field
  packet, progress report, recertification report, and completion audit from
  the same canonical source. Current result: `docs/field-packet/manifest.json`
  has 27 runnable cards, 81 GPX files, 27 navigation GPX files, and
  `gpx_validation_passed = true`.
- Completion audit is now certifiable for the field packet:
  `years/2026/checkpoints/field-tool-completion-audit-2026-05-06.json` reports
  `status = passed`, 13/13 requirements passed, 27 route cards, and 251/251
  official segment ids represented. The audit now states source-gap handling
  honestly: 14 source-gap warnings are represented by explicit field
  connector/re-park/manual metadata; 0 are hidden.

### Field Cue Sheet / Trail Roadbook Notation

- Added a text-first `Field Cue Sheet` / `wayfinding_cues` layer to the phone
  packet. The cue rows now follow a roadbook-style pattern: sequence number,
  cumulative miles, leg miles, cue type, action, signed trail/road to follow,
  target, and an `UNTIL` anchor describing the observable junction/landmark.
- This is specifically meant to prevent the Harrison/Who Now failure mode:
  a target-only instruction like `Leave car toward #51 Who Now Loop` is not
  enough when the runner must first follow another signed access trail. The
  field cue must say what to follow and until what sign/junction.
- The completion audit now rejects movement cues that lack `until`, `target`,
  or a signed trail/road/landmark source. That makes this a certifiable field
  contract rather than a wording preference.
- Updated `AGENTS.md` so future agents preserve this distinction: official
  segment ids stay as completion metadata, while visible phone cue numbers are
  field-decision order from the parked car back to the parked car.

### Headless Field-Runner Walkthrough Audit

- Added `years/2026/scripts/field_route_walkthrough_audit.py`, a deterministic
  "blind walker" check for the exported phone packet and Nav GPX. It uses only
  the runner-facing artifacts plus public/signed trail graph labels and
  official segment geometry, then checks whether named access trails,
  connectors, road legs, hidden GPX gaps, official coverage, and direction
  rules are discoverable from the cue sheet.
- Added synthetic regression coverage for the original bug class: a route from
  a parked car over `#57 Harrison Hollow` to first official segment `#51 Who
  Now` fails if the cue only says `Leave car toward #51`; the same route passes
  when the cue names `#57 Harrison Hollow` and the `#51` junction.
- Fixed walker-layer false positives found during validation: generic synthetic
  connector labels like `OSM footway connector 72484` no longer count as
  field-visible sign names, repeated-pass direction checks now use matched
  official traversal groups instead of the nearest repeated endpoint, and the
  coverage matcher uses a spatial index/resampled route line so the full audit
  is fast enough to run during normal validation.
- Follow-up pass made the full packet walkthrough-certifiable. The exporter now
  enriches `wayfinding_cues` from the same route-line-matched trail graph used
  by the walker, so named non-credit roads/trails such as East Sunset Peak
  Road, South Council Spring Road, Bogus Creek access roads, Hidden Springs
  connectors, and Eagle Bike Park connector trails appear in the phone cue text
  instead of being hidden behind `follow GPX`.
- Added per-segment `segment_direction_evidence` to the public field-tool data
  so ascent validation does not assume GeoJSON line order is always the allowed
  uphill direction. This fixed the Polecat Loop 2 case where the valid ascent
  evidence says the route is opposite official geometry order.
- Current result: `python years/2026/scripts/field_route_walkthrough_audit.py`
  passes 27/27 routes with no remaining failure counts, with a current
  May 7 checkpoint written to
  `years/2026/checkpoints/field-route-walkthrough-audit-2026-05-07.json`.
  The older completion audit also still passes 13/13 requirements and the
  field menu still covers 251/251 official segment ids.

#### May 7 follow-up: walkthrough certification cleanup

- Experiment: reran the headless walker against all 27 exported routes. The
  first May 7 pass reproduced the remaining blocker pattern: 16/27 routes
  passed, with missing named start-access edges, missing connector cues, and
  one Polecat direction-rule failure.
- Finding: most failures were not route coverage failures. The Nav GPX was
  continuous enough, but the phone cue text only named the planner's internal
  start/connector metadata. It did not always name extra road/trail edges that
  the exported route line actually traversed and the walker could map-match.
- Experiment: added failing synthetic exporter tests for two generic cases:
  a start-access edge matched from the route line but absent from the cue text,
  and a between-official connector edge matched from the route line but absent
  from the cue text. Then added the exporter enrichment layer and made the
  tests pass.
- Finding: the first enrichment pass still failed production because the
  helper read top-level `route.segment_ids`, while the exporter route object
  still held claimed segments under `route.outing.segment_ids`. Fixing the
  helper to read either location let the enrichment apply to production routes.
- Experiment: added a failing walker test for ascent direction where the
  valid route traversal is opposite official GeoJSON coordinate order. This
  matched the Polecat Loop 2 failure. Added exported `segment_direction_evidence`
  so the walker can validate ascent rules from explicit planner evidence
  instead of assuming GeoJSON order means uphill.
- Decision: the correct fix is not to suppress walker failures. If the walker
  finds a route-line-matched named non-credit edge, the phone packet should
  name it in `wayfinding_cues` / `turn_by_turn_steps`, unless the walker is
  demonstrably treating a synthetic implementation label as a real sign.
- Decision: generated field artifacts should continue to come from
  `export_mobile_field_packet.py` and the canonical map data. Do not hand-edit
  `docs/field-packet/index.html`, `field-tool-data.json`, or GPX files to make
  the audit pass.
- Validation:
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated the
    phone packet and 81 GPX files.
  - `python years/2026/scripts/field_route_walkthrough_audit.py --output-json years/2026/checkpoints/field-route-walkthrough-audit-2026-05-07.json --output-md years/2026/checkpoints/field-route-walkthrough-audit-2026-05-07.md`
    passed 27/27 routes with zero failures.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 13/13
    requirements and still represented 251/251 official segments.
  - `pytest -q years/2026/tests` passed 326 tests.

#### May 7 follow-up: live GPS field map prototype

- Objective: test whether the phone field packet can become more than a GPX
  download/cue sheet by showing the runner's current GPS position on a simple,
  route-first map. The target is field decision support, not high-fidelity
  street/topo cartography.
- Decision: build the first pass as a deterministic PWA page, not an
  image-generated artifact and not an external tile-map dependency. The page
  should use the same public field-tool data and selected Nav GPX as the rest
  of the packet, then render a controllable SVG route ribbon with cue markers,
  sparse chevrons, and the live GPS dot.
- Implementation: added generated `docs/field-packet/live-map.html`. It reads
  `field-tool-data.json`, loads the selected outing's `gpx_href`, parses GPX
  with `DOMParser`, renders ribbon / cue-leg / napkin styles, stores the active
  outing in the same local-storage key as the phone packet, and uses
  `navigator.geolocation.watchPosition()` for live position updates.
- Finding: direct `file://` loading cannot fetch `field-tool-data.json` or GPX
  in the browser, so live-map validation needs the GitHub Pages HTTPS URL or a
  local HTTP server. This is acceptable for the iPhone PWA target because
  geolocation also requires a secure context in normal field use.
- Validation so far:
  - Added failing exporter tests first for live-map generation, per-outing
    links, service-worker precache inclusion, route data loading, geolocation,
    style controls, and active-outing selection.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py -k "live_gps_map or phone_first"`
    passed after implementation.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py` passed
    28 tests.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated the
    phone packet, including `live-map.html` and service-worker cache metadata.
  - Local browser validation through `python -m http.server 8765 --directory docs/field-packet`
    loaded `http://127.0.0.1:8765/live-map.html?outing=1-2&v=2` with no console
    errors or warnings after adding the missing mobile PWA meta/icon tags.

#### May 7 follow-up: live GPS map rendering cleanup

- Objective: make the Harrison live map readable on a phone instead of repeating
  the old clutter pattern: dense arrows, raw GPX-point drawing, waypoint/order
  mismatch, and a solid-blue route that did not show useful progress.
- Finding: the 1B Nav GPX currently has two GPX track segments totaling about
  9.29 rendered miles, while the route card says 5.69 miles. The live map must
  not hide that by drawing a fake connector or pretending the gap is runnable.
- Implementation: the live map now preserves GPX `trkseg` parts, builds route
  distance only within real track segments, simplifies the display polyline,
  draws a haloed route ribbon with sparse distance-sampled chevrons, and uses
  `wayfinding_cues` as the primary numbered marker layer instead of raw GPX
  waypoint names.
- Implementation: fixed the progress-gradient rendering so each displayed
  segment keeps `routeM` and the SVG stroke is not overridden by the base route
  class. Browser verification on 1B showed first/middle/final route strokes of
  blue/green/red (`hsl(214.6 ...)`, `hsl(126.5 ...)`, `hsl(5.4 ...)`) with 17
  chevrons and no console errors.
- Decision: when the live map detects inter-track gaps, it shows a visible
  route-review warning. GPX length mismatch was previously included here, but
  that was superseded by the route-distance-authority correction: GPX track
  length is not a decision metric.
- Validation:
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py` passed
    30 tests.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated the
    phone packet and 81 GPX files.
  - Browser validation at
    `http://127.0.0.1:8776/live-map.html?outing=1-2&v=routem-...` showed the
    route-review warning, eight wayfinding cue markers, 17 chevrons, gradient
    route strokes, and zero console errors/warnings.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 13/13
    requirements.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 27/27
    routes.

#### May 7 follow-up: live-map follow surface and arrows

- Objective: stop treating the Harrison live map as a whole-route puzzle. The
  field view must be visually followable from the screenshot: current cue,
  next cue, active route leg, and direction arrows should make the next movement
  obvious.
- Finding: the whole-route overview remained too ambiguous for dense overlap.
  The right primary UI is the active cue-to-cue leg. A later pass also found
  that the blue ribbon was simplified for display while arrow direction was
  sampled from raw dense GPX points, which could make arrows look inconsistent
  around curves and overlaps.
- Implementation: changed the live map layout to a single-screen follow
  surface, added a map-embedded `FOLLOW xx -> yy` banner, emphasized FROM/NEXT
  cue markers, muted inactive cue markers, and fitted the initial view to the
  active cue-to-cue leg. Replaced the old chevron rendering with active-leg
  direction arrows and then moved arrow placement/tangent sampling onto the
  same displayed geometry used by the highlighted blue ribbon.
- Implementation: updated the service worker to treat dynamic field-map/data/GPX
  resources as network-first and to normalize cache keys without query strings,
  so cache-busted local/browser validation does not keep showing stale
  `live-map.html`.
- Decision: the live map can still offer full-route overview controls, but the
  default field artifact should answer "what do I follow next?" rather than ask
  the runner to solve the whole route order from overlapping colored lines.
- Validation:
  - Added failing regressions for single-screen follow-surface behavior and
    consistent active-leg direction arrows, then made them pass.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py::test_live_gps_map_default_viewport_is_single_screen_follow_surface years/2026/tests/test_export_mobile_field_packet.py::test_live_gps_map_uses_consistent_active_leg_direction_arrows`
    passed.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated the
    phone packet and 81 GPX files.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py` passed 36
    tests.
  - Extracting the generated `live-map.html` script with
    `perl -0ne 'if (m#<script>(.*)</script>#s) { print $1 }'` and running
    `node --check /tmp/boise-live-map.js` passed.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 13/13
    requirements for 27 routes and 251/251 official segments.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 27/27
    routes with zero failures.
  - Browser validation at
    `http://127.0.0.1:8780/live-map.html?outing=1-2&v=display-geometry-arrows`
    showed the active `02 -> 03` leg in one viewport with FROM/NEXT markers,
    consistent arrows on the blue leg, muted context, and no console errors or
    warnings. Stepping to `03 -> 04` also rendered a followable active leg with
    arrows from `FROM 03` to `NEXT 04`.

#### May 7 follow-up: live map must be field-followable

- Objective: make `docs/field-packet/live-map.html` behave as a field-navigation
  artifact instead of a whole-route overview that requires the runner to
  visually solve dense overlaps.
- Finding: even after the source GPX was repaired, a full-route ribbon is still
  the wrong primary field UI for Harrison Hollow-style overlap. The runner
  needs to know the active cue-to-cue leg and the next observable cue, not infer
  the full route order from color crossings.
- Decision: the live map's default behavior should be roadbook-like: active
  wayfinding leg highlighted, rest of the route muted, current/next cue markers
  emphasized, sparse chevrons only on the active leg, and manual cue stepping
  available when GPS is not active or when the runner wants to preview.
- Product invariant: the field map must provide an unambiguous route-following
  surface: "I am here, this is the active cue-to-cue leg, this is the next
  cue/junction, and this is what to follow until then." If the runner has to
  mentally solve a full overlapping overview, the artifact is failing.
- Implementation: added active-cue state to the generated live map, active leg
  range calculation from `wayfinding_cues`, manual previous/next cue controls,
  active-leg fit, GPS-driven active-cue updates, and demoted full-route context.
  The existing full-route fit remains available as a secondary overview. Also
  fixed car-to-car marker layering so an overlapping finish dot cannot hide the
  start/current cue marker; overlapping endpoints now render as `START/FINISH`
  context below the cue markers. The initial active cue uses the same
  route-distance cue selection as GPS, so zero-distance start cues do not make
  the field panel skip past the actual first movement leg.
- Validation:
  - Added a failing regression test that requires the generated live map to
    expose active cue-leg navigation behavior instead of only whole-route
    drawing, plus a failing regression test for overlapping start/finish marker
    visibility.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py` passed 34
    tests.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated the
    phone packet and 81 GPX files.
  - Extracting the generated `live-map.html` script and running
    `node --check /tmp/boise-live-map.js` passed.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 13/13
    requirements.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 27/27
    routes.
  - Browser validation at
    `http://127.0.0.1:8776/live-map.html?outing=1-2&v=active-leg` showed the
    default view on the active blue `Cue 02 -> 03` movement leg with muted
    surrounding route context, cue stepping controls, no route-review gap
    warning for 1B, and no browser console errors/warnings.
  - `python years/2026/scripts/field_progress_report.py` and
    `python years/2026/scripts/field_recertification_report.py` both passed the
    clean challenge-start state with 251/251 remaining segments preserved.

#### May 7 follow-up: route-wide gradient refinement

- Objective: make the live-map ribbon read as one route-wide progress gradient,
  not as a set of hard color bands at trail/cue boundaries.
- Finding: the first gradient pass still colored each SVG slice with one solid
  midpoint color and used a wide rainbow hue sweep. On a simplified route this
  could make individual trail stretches read like separate color categories.
- Implementation: changed ribbon/napkin mode to generate per-slice SVG
  `linearGradient` definitions from each slice's start route distance to end
  route distance. Replaced the rainbow hue sweep with a single blue-to-violet-
  to-red ramp so the route reads as one continuous progression.
- Validation:
  - Added a failing regression assertion that rejected midpoint-only slice
    strokes and the old hue-ramp renderer, then made it pass.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py` passed
    30 tests.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated the
    phone packet and 81 GPX files.
  - Browser validation at
    `http://127.0.0.1:8776/live-map.html?outing=1-2&v=violet-gradient-...`
    showed 426 route slices backed by 426 SVG gradients, with start/mid/end
    colors `rgb(37 99 235)`, `rgb(103 68 237)`, and `rgb(220 38 38)`, plus no
    console errors or warnings.

#### May 7 follow-up: cue order versus overlap color

- Objective: make the live map answer the field question "do I go from cue 02
  to cue 03?" even when the physical route overlaps itself and later passes
  overpaint earlier passes.
- Finding: the 1B Nav GPX still contains extra geometry beyond the 5.69-mile
  route card. The map was correctly warning about that, but the gradient and
  finish/progress display were still using the longer 9.29-mile GPX span, which
  made cue colors look inconsistent with the cue sheet.
- Decision correction: the card-span cap was the wrong fix because it made the
  renderer compensate for a bad source artifact. The invariant is that Nav GPX,
  route card mileage, source-gap flags, and cue order must describe the same
  car-to-car route. A renderer may warn about a mismatch, but it must not hide
  one by cropping or reinterpreting the GPX.
- Root cause: the Package 1 manual field-menu override had copied disconnected
  rendered parts from the old collapsed Harrison/Hillside candidate. That
  produced stale `source_gap_warning` state plus a 1B Nav GPX that included an
  extra disconnected track part.
- Implementation: rebuilt the Package 1 override route geometries from official
  segment geometry plus graph-routed connector paths. `1B. Harrison Hollow` now
  exports as one continuous 6.36-mile car-to-car Nav GPX with no inter-`trkseg`
  gap or route-card length mismatch; `1A. West Climb` now exports as one
  continuous 7.93-mile car-to-car Nav GPX. Removed the live-map card-span cap
  and colored cue-dot workaround so the map displays the actual Nav GPX.
- Implementation: tightened validation so named connector metadata cannot
  excuse a hidden GPX track break. A hidden track break now needs an explicit
  re-park, multi-start boundary, or manual day-of hold; otherwise the GPX
  validation and completion audit fail.
- Implementation: for non-Package-1 routes that still have source split
  warnings, the exporter now graph-stitches inter-track gaps into explicit Nav
  GPX connector geometry when a connector path exists, and records
  `source_gap_repair` metadata. The completion audit can therefore distinguish
  "hidden source gap" from "source split repaired in the exported GPX."
- Validation:
  - Added failing regressions for: live map must warn but not mask hidden track
    gaps; `validate_outing_export()` must not treat a named connector cue as a
    hidden track-gap explanation; and `field_tool_completion_audit.py` must fail
    `source_gap_warning` even when generic named connector metadata exists.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py::test_validate_outing_export_does_not_treat_named_connector_as_hidden_track_gap years/2026/tests/test_export_mobile_field_packet.py::test_live_gps_map_warns_but_does_not_mask_nav_gpx_card_mismatch years/2026/tests/test_field_tool_completion_audit.py::test_completion_audit_fails_source_gap_even_when_named_connector_is_declared`
    passed.
  - `python years/2026/scripts/human_loop_plan.py` regenerated the canonical
    private field-menu map data from the repaired override source.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated the
    phone packet and GPX files.
  - A GPX check showed `1B` has one track segment, 6.36 miles, max trackpoint gap
    0.029 mi, starts at Harrison Hollow Trailhead, and ends at Harrison Hollow
    Trailhead. `1A` has one track segment, 7.93 miles, max trackpoint gap
    0.0365 mi, starts at West Climb, and ends at West Climb.
  - `python years/2026/scripts/field_progress_report.py` passed with 251/251
    remaining segments available and coverage preserved.
  - `python years/2026/scripts/field_recertification_report.py` passed with
    remaining full completion feasible.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 13/13
    requirements.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 27/27
    routes.

#### May 7 follow-up: passive GPS dot and map gestures

- Objective: make the live map behave like a map on iPhone: pinch and zoom
  should work directly, and GPS should simply show the runner's position dot
  instead of introducing a separate Follow mode.
- Decision: removed the `Follow` toggle and `state.follow` model. GPS updates
  now update the user dot, accuracy ring, distance-to-route, and progress
  estimate, but they do not auto-recenter the map or auto-step the active cue.
  Manual cue stepping and Fit/Fit leg remain available controls.
- Implementation: added SVG pointer gesture handling for one-finger pan,
  two-finger pinch zoom, and wheel zoom. The zoom buttons now use the same
  `zoomAt()` path as pinch/wheel so map controls stay consistent.
- Validation:
  - Added a failing regression that rejects the old Follow button/state and
    requires pointer pan, pinch zoom, wheel zoom, and passive GPS behavior.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py::test_live_gps_map_is_gesture_map_with_passive_gps_dot`
    failed before the implementation and passed after it.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py::test_live_gps_map_is_active_cue_leg_navigation_artifact years/2026/tests/test_export_mobile_field_packet.py::test_live_gps_map_default_viewport_is_single_screen_follow_surface years/2026/tests/test_export_mobile_field_packet.py::test_live_gps_map_uses_consistent_active_leg_direction_arrows years/2026/tests/test_export_mobile_field_packet.py::test_live_gps_map_is_gesture_map_with_passive_gps_dot`
    passed 4 tests.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py` passed 37
    tests after regeneration.
  - Extracting `docs/field-packet/live-map.html` script and running
    `node --check /tmp/boise-live-map.js` passed.
  - Browser validation on
    `http://127.0.0.1:8781/live-map.html?outing=1-2&v=passive-gps-gestures`
    showed no `Follow` button, no console errors/warnings, progress
    `0.00 / 6.34 mi`, and one-finger drag changed the SVG viewBox.
  - `python years/2026/scripts/field_progress_report.py` passed with 251/251
    remaining coverage preserved.
  - `python years/2026/scripts/field_recertification_report.py` passed with
    remaining full completion feasible.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 13/13
    requirements.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 27/27
    routes.

#### May 7 follow-up: live-map cue marker callouts

- Objective: keep the exact start/end/junction point visible on the live map.
  The prior large blue/green active/next cue bubbles were useful, but could
  cover the precise route point at confusing junctions.
- Implementation: changed active/next cue markers into callouts offset from the
  route, with a leader line back to a small anchor at the exact route point.
  Reduced the active/next bubble radius and label tag size so the route geometry
  remains visible underneath.
- Validation:
  - Added a failing regression for offset active/next cue markers and exact-point
    anchors, then made it pass.
  - Browser DOM validation on
    `http://127.0.0.1:8782/live-map.html?outing=1-2&v=marker-callouts` showed
    two exact cue anchors, active/next callouts, progress `0.00 / 6.34 mi`, and
    no console errors/warnings. The browser screenshot command timed out, so the
    visual confirmation here is DOM/state-based.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py` passed 38
    tests.
  - Extracting `docs/field-packet/live-map.html` script and running
    `node --check /tmp/boise-live-map.js` passed.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 13/13
    requirements.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 27/27
    routes.
  - `python years/2026/scripts/field_progress_report.py` passed with 251/251
    remaining coverage preserved.
  - `python years/2026/scripts/field_recertification_report.py` passed with
    remaining full completion feasible.

#### May 7 follow-up: zoom-stable endpoint markers

- Objective: fix the zoomed-in live-map start/finish marker problem where the
  large green/red endpoint dots and `START/FINISH` label covered the exact
  trailhead/junction point.
- Finding: the active/next cue callouts were screen-stable, but the start and
  finish markers still used fixed SVG map-unit radii (`r=17`, `r=15`) and
  inline label offsets. At high zoom that made the endpoint artwork scale over
  the route point.
- Implementation: replaced direct endpoint dots/labels with an
  `endpointCallout()` renderer. The exact endpoint now keeps a small anchor on
  the true route coordinate, while the visible start/finish dots and label are
  offset with a leader line.
- Validation:
  - Added a failing regression for zoom-stable endpoint callouts and removal of
    the fixed endpoint radii, then made it pass.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py::test_live_gps_map_does_not_hide_start_when_start_and_finish_overlap`
    passed 1 test.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated 81 GPX
    files and the field-packet HTML/manifest.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py` passed 38
    tests.
  - Extracting `docs/field-packet/live-map.html` script and running
    `node --check /tmp/boise-live-map.js` passed.
  - Browser DOM validation on
    `http://127.0.0.1:8783/live-map.html?outing=1-2&v=endpoint-callouts` showed
    one endpoint anchor, one endpoint callout line, one start dot, one finish
    dot, zero `r=17`/`r=15` circles, and no console errors/warnings. The browser
    screenshot command timed out, so visual confirmation is DOM/state-based.
  - `python years/2026/scripts/field_progress_report.py` passed with 251/251
    remaining coverage preserved.
  - `python years/2026/scripts/field_recertification_report.py` passed with
    remaining full completion feasible.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 13/13
    requirements.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 27/27
    routes.

#### May 7 follow-up: off-route GPS visibility

- Objective: make `Start GPS` visibly useful when testing the PWA away from the
  selected outing, such as from home, without reintroducing automatic recentering
  or follow mode.
- Finding: the GPS dot used a fixed route/map-unit radius (`r=10`), so it could
  become effectively invisible after zooming far out. If the GPS point was
  outside the current viewport, the passive GPS behavior also had no visible
  offscreen indicator.
- Implementation: made the user dot and heading marker screen-stable, added a
  `GPS off map` edge indicator when the GPS fix is outside the current view,
  and relabeled the explicit Fit control to `Fit GPS` after a fix is acquired.
  The map still does not recenter or auto-follow when GPS updates arrive.
- Validation:
  - Added a failing regression for offscreen GPS visibility and no-autofollow
    behavior, then made it pass.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py::test_live_gps_map_surfaces_offscreen_gps_without_autofollow`
    passed 1 test.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated 81 GPX
    files and the field-packet HTML/manifest.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py` passed 39
    tests.
  - Extracting `docs/field-packet/live-map.html` script and running
    `node --check /tmp/boise-live-map.js` passed.
  - Local Playwright geolocation smoke against
    `http://127.0.0.1:8784/live-map.html?outing=1-2&v=gps-offscreen` with an
    off-route test coordinate rendered one `.user-offscreen` marker, changed
    the Fit button to `Fit GPS`, and showed the `GPS acquired; tap Fit GPS to
    include your dot.` status without recentering.
  - Local Playwright geolocation smoke with a 1B route coordinate rendered one
    `.user-dot`, no `.user-offscreen`, and kept the Fit button as `Fit GPS`.
  - `python years/2026/scripts/field_progress_report.py` passed with 251/251
    remaining coverage preserved.
  - `python years/2026/scripts/field_recertification_report.py` passed with
    remaining full completion feasible.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 13/13
    requirements.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 27/27
    routes.

#### May 7 follow-up: optional live-map basemap tiles

- Objective: add contextual tiles behind the route-first live map without
  turning the field PWA into a tile-map dependency.
- Finding: the Ridge to Rivers interactive map embeds an Ada County ArcGIS app.
  The app exposes a public `FoothillsMosaic2025` ArcGIS tile layer, while its
  richer trail information is an ArcGIS feature layer rather than a simple XYZ
  raster tile source. For the first implementation, OSM is the default readable
  basemap and R2R/Ada County imagery is an optional cycle state.
- Implementation: added an SVG tile layer behind the grid/route, a `OSM -> R2R
  -> No map` basemap button, OSM attribution, R2R imagery attribution, and tile
  projection functions that preserve the existing GPX-derived route projection.
  The route ribbon, wayfinding cues, and GPS marker remain the primary field
  navigation surface.
- Validation:
  - Added a failing regression for optional basemap tiles without Leaflet, then
    made it pass.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated 81 GPX
    files and the field-packet HTML/manifest.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py` passed 40
    tests.
  - Extracting `docs/field-packet/live-map.html` script and running
    `node --check /tmp/boise-live-map.js` passed.
  - Local Playwright smoke against
    `http://127.0.0.1:8785/live-map.html?outing=1-2&v=tiles-local` rendered 20
    OSM tiles with OSM attribution, cycled to 20 R2R imagery tiles with R2R/Ada
    County attribution, then cycled to `No map` with zero tile images and hidden
    attribution. No console errors or warnings were observed.
  - `python years/2026/scripts/field_progress_report.py` passed with 251/251
    remaining coverage preserved.
  - `python years/2026/scripts/field_recertification_report.py` passed with
    remaining full completion feasible.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 13/13
    requirements.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 27/27
    routes.

#### May 8 follow-up: certified multi-start map replacement

- Objective: redo the outing-efficiency analysis with the settled user answers
  and replace the canonical field packet only for alternatives that can pass the
  full source/GPX/cue/timing/direction certification chain.
- Finding: the first multi-start audit over-promoted `13` and `17`. The reverse
  order heuristic for components with more than three trails was dropping
  non-reversible ascent-only trails. After fixing that, `13` and `17` were no
  longer better than baseline.
- Implementation: added a private merged override generator,
  `years/2026/scripts/multi_start_field_menu_override.py`, and regenerated the
  canonical map/menu/phone artifacts with certified replacements for `1A`,
  `4C`, `5`, and `15A`. Exact private Strava-derived anchors remain under
  ignored private files; public outputs use safe labels.
- Validation:
  - `python years/2026/scripts/multi_start_alternative_audit.py` regenerated the
    corrected audit.
  - `python years/2026/scripts/multi_start_field_menu_override.py` wrote the
    ignored private override source.
  - `python years/2026/scripts/human_loop_plan.py --field-menu-overrides-json years/2026/inputs/personal/private/2026-field-menu-overrides-v2-multi-start.private.json`
    regenerated the private canonical map/menu.
  - `python years/2026/scripts/export_example_map.py` regenerated public
    sanitized map/menu artifacts.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated 93 GPX
    files and the phone packet.
  - `python years/2026/scripts/field_progress_report.py` passed with 251/251
    remaining coverage preserved.
  - `python years/2026/scripts/field_recertification_report.py` passed.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 13/13
    requirements.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 31/31
    routes.

#### May 10 proof pass: runner-perspective frame-shift route audit

- Objective: pressure-test every current field-packet route from the runner's
  on-trail perspective, not only from route-card, GPX, or coverage validity.
- Result: added a generated checkpoint under
  `years/2026/checkpoints/runner-perspective-frame-shift-2026-05-10/` with an
  `index.md`, `manifest.json`, and one route-audit markdown file for each of
  the 30 current field-packet routes. The pass covers 271 start, cue/junction,
  and return-to-car checkpoints.
- Frame shift: the model frame is "the route artifact exists and validates";
  the runner frame is "can I choose the right signed trail, road edge, overlap,
  connector, or return leg while moving?" The output keeps literal sightline
  claims proof-gated because it used local route/overlay data, not field photos,
  Street View, or current day-of signage.
- Clarification/pivot: the user clarified that `what do you see?` was meant as
  an internal step-back mechanism for finding unexpected optimization surfaces,
  not as the final prose artifact. Added
  `years/2026/scripts/runner_perspective_optimization_audit.py`,
  `optimization-index.md`, per-route `optimization-audits/`, a public Strava
  behavior evidence addendum, and
  `unexpected-optimization-shortlist.md`.
- Optimization result: the corrected pass found 430 route optimization leads
  across 30 routes, including 62 high-priority leads. The highest-value next
  experiment is the `16A-2` Dry Creek / Shingle / Sweet Connie cluster, using
  public Shingle-up / Dry-down loop behavior as a source-backed hypothesis.
- Current blocker: this is not a day-of readiness signoff. Routes still need
  current Ridge to Rivers condition/signage checks before running, and eventual
  BTC activity geometry before credit claims.
- Validation:
  - `python years/2026/scripts/runner_perspective_frame_shift_audit.py` wrote
    30 route audits and 271 checkpoints.
  - `python years/2026/scripts/runner_perspective_frame_shift_audit.py
    --skip-frame-log` regenerated the audits after cue anchoring was corrected
    to use route-mile positions along the GPX track.
  - `python -m py_compile
    years/2026/scripts/runner_perspective_frame_shift_audit.py` passed.
  - `python years/2026/scripts/runner_perspective_optimization_audit.py` wrote
    30 route optimization audits, 430 optimization leads, and 62 high-priority
    leads.
  - `python years/2026/scripts/runner_perspective_optimization_audit.py
    --skip-frame-log` regenerated the optimization audits after linking the
    public-route evidence lane.
  - `python -m py_compile
    years/2026/scripts/runner_perspective_optimization_audit.py` passed.
  - `jq '.route_count == 30 and .checkpoint_count == 271 and (.routes|length==30)'
    years/2026/checkpoints/runner-perspective-frame-shift-2026-05-10/manifest.json`
    returned `true`.
  - `python -m pytest years/2026/tests/test_multi_start_alternative_audit.py years/2026/tests/test_export_mobile_field_packet.py years/2026/tests/test_field_tool_completion_audit.py`
    passed 72 tests.

#### May 10 implementation: field-day layer as the primary execution artifact

- Objective: move beyond single-card optimization and make the day-level field
  plan the phone-first execution artifact, while keeping route cards as the
  underlying proof/navigation unit.
- Implementation:
  - `export_field_day_layer.py` now writes an explicit execution model:
    `field_day_layer` is primary, `certified_route_card` is the proof unit, and
    the phone default view is `field-days`.
  - The field-day layer now demotes route cards with parking, cue/card mileage,
    missing GPX, or field-navigation audit blockers to
    `needs_route_card_audit_fix` instead of counting them as certified proof
    units.
  - `export_mobile_field_packet.py` now embeds the execution model, opens the
    packet on Field Days when the layer exists, keeps Route Cards as the
    subordinate tab, and displays route-card promotion/audit blockers in the
    day view.
- Result: the current layer covers 251/251 official segments across 31 field
  days and 50 loops, including 14 multi-start days. After the stricter audit
  gate, only 1 loop is audit-clean certified, 14 need route-card audit fixes,
  and 35 still need route-card promotion, so the packet remains explicitly
  non-publication-ready.
- Validation:
  - `python -m pytest years/2026/tests/test_export_field_day_layer.py
    years/2026/tests/test_export_mobile_field_packet.py` passed 54 tests.
  - `python -m pytest years/2026/tests/test_field_tool_completion_audit.py -q`
    passed 14 tests.
  - `python years/2026/scripts/export_field_day_layer.py` regenerated the
    field-day checkpoint.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated the
    phone packet and GPX zip.
  - `python -m json.tool years/2026/checkpoints/human-executable-field-day-layer-2026-05-10.json`
    and `python -m json.tool docs/field-packet/field-tool-data.json` validated
    the generated JSON.
  - Local HTTP smoke at `http://127.0.0.1:8788/index.html` confirmed the
    generated page serves with `view-field-days`, `DEFAULT_VIEW = "field-days"`,
    an active Field Days tab, and visible audit blockers.
  - `python years/2026/scripts/field_tool_completion_audit.py` failed 12/13 on
    the existing route-card quality requirement; that failure is now surfaced in
    the field-day layer as audit-fix gaps instead of hidden behind certified
    route-card wording.

#### May 10 route-mapping optimization stop-check

- Objective: identify the next high-value route-mapping optimization by real
  on-foot effort/time savings, then run a coverage/frame stop-check before
  treating the recommendation as done.
- Finding: `10A` remains the best single route-card redesign target, but
  `10A-MS-08` is not promotable as-is because the access-verification artifact
  keeps both proposed starts parking-gated or not certified car starts.
- Material stop-check result: the current field-tool audit checked parking,
  cue presence, GPX continuity, official endpoint coverage, source gaps, and
  public-safety leakage, but it did not compare field-card on-foot mileage
  against displayed cue mileage. A direct packet scan also inspected Nav GPX
  length, but the later route-distance-authority correction removed GPX length
  as a decision or certification source.
- Implementation: added a generic mileage-truth guard to
  `field_tool_completion_audit.py` so field-packet certification fails when
  card mileage and cue mileage drift beyond tolerance. The initial GPX-length
  portion of this guard was later removed because route totals come from route
  distance calculation, not GPX track length.
- Validation:
  - `pytest -q years/2026/tests/test_field_tool_completion_audit.py` passed 14
    tests.
  - `python years/2026/scripts/field_tool_completion_audit.py --output-json
    /tmp/field-tool-completion-audit-mileage-check.json --output-md
    /tmp/field-tool-completion-audit-mileage-check.md` failed as expected with
    12/13 requirements passing; the failing requirement now exposes
    mileage-truth drift instead of letting the packet certify from continuity
    and coverage alone.
  - `pytest -q years/2026/tests/test_route_pain_index.py` passed 3 tests after
    the access-decision input was added.
  - `python years/2026/scripts/route_pain_index.py` regenerated
    `route-pain-index-2026-05-10.{json,md}` and its manifest with the current
    field-tool audit input hash.
  - `pytest -q years/2026/tests/test_multi_start_alternative_audit.py years/2026/tests/test_export_mobile_field_packet.py years/2026/tests/test_field_tool_completion_audit.py`
    passed 86 tests.
  - `python years/2026/scripts/field_tool_completion_audit.py` failed with
    12/13 requirements passing, as expected for the current packet; the failing
    requirement lists missing verified parked starts plus cue/card mileage drift
    on `4B`, `4A`, `5A`, `4C-1`, and `15A-2`.
- Current blocker: do not promote a `10A` replacement or call the field packet
  ready until the route source/generator is repaired so the card and cue sheet
  describe the same route distance. GPX remains navigation geometry, not a
  mileage source.

#### May 9 audit: route-specific exception debt

- Objective: find remaining code-level overrides or route-specific branches
  that encode a reusable planner, exporter, audit, or local-reality heuristic.
- Finding: the multi-start/re-park path is now promoted through active
  recalculation, but several named-route or named-place branches remain:
  Harrison-specific signpost/access hints, a Package 1 collapsed-route guard,
  a Harrison-specific overlap-exit warning, a Harrison-only completion-audit
  assertion, a West Climb candidate summary metric, public-safe private-anchor
  label rewrites, Bogus Basin anchor whitelisting, day-of named-trail rules, and
  Shingle what-if exception scripts.
- Documentation:
  - Added `btc_exception_001` to `docs/BTC_HEURISTICS.md`.
  - Added `btc_failure_exception_001` to `docs/BTC_FAILURE_MODES.md`.
  - Added concrete and contrastive cases to `docs/BTC_CASES.md`.
  - Added `btc_eval_exception_001` to `docs/BTC_BEHAVIOR_EVALS.md`.
  - Wrote the open audit checkpoint at
    `years/2026/checkpoints/route-specific-exception-audit-2026-05-09.md` and
    `.json`.
- Boundary: this pass inventories and codifies the debt; it does not yet
  replace every route-specific branch with generic generator/audit/config
  behavior.

#### May 9 remediation: active field-packet exception removal

- Objective: remove the active exporter/audit branches that fixed Harrison or
  Package 1 by name instead of enforcing the reusable rule.
- Changes:
  - Moved Harrison field-tested signpost/access prose into
    `years/2026/inputs/personal/2026-field-route-hints.json`, keeping code as a
    data consumer rather than a route-name branch.
  - Replaced the Package 1 collapsed-route guard with accepted replacement
    manifest preservation keyed by package plus block identity. Long single-card
    outings are allowed unless they supersede an accepted replacement/split.
  - Replaced Harrison cue-number overlap logic with generic overlap-exit
    warnings driven by `overlap_match` and the next signed cue.
  - Replaced Harrison-only completion-audit assertions with generic named
    start-access and return-access cue checks.
  - Updated `docs/BTC_FIELD_PACKET_REQUIREMENTS.md` so the standing guards are
    phrased as generic named-access and accepted-replacement requirements, with
    concrete Harrison/West Climb cases living in the heuristic/failure/case
    docs.
- Remaining exception debt: private-anchor label sanitization, Bogus/day-of
  local-reality constants, West Climb-specific summary metrics, and Shingle
  diagnostic scripts remain logged in
  `years/2026/checkpoints/route-specific-exception-audit-2026-05-09.*`.

#### May 9 validation: cleanup pipeline remains field-certifiable

- Objective: verify that the cleanup still leaves the main route generation,
  browser map, phone field packet, live map, GPX exports, progress accounting,
  and recertification path working from the canonical source.
- Fixes found during validation:
  - Made `multi_start_field_menu_replacements.py` idempotent when rerun against
    an already-replaced active map.
  - Made progress and recertification commands default to validated segment
    state embedded in the exported field packet when no separate
    `--progress-json` is supplied.
  - Fixed public example-map sanitization so absolute private paths are removed
    from exported map-data JSON and private Strava-anchor labels are removed
    from public menu markdown.
  - Guarded the public Leaflet map against invalid bounds when private-anchor
    geometry is redacted from public examples.
- Validation:
  - `python years/2026/scripts/multi_start_field_menu_replacements.py` passed
    and wrote 4 multi-start replacement packages.
  - `python years/2026/scripts/human_loop_plan.py` regenerated the canonical
    private map/data/menu.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated the
    phone packet and 90 GPX files.
  - `python years/2026/scripts/field_progress_report.py` reported 13 completed
    validated segments, 238 remaining, 0 missing, and preserved remaining
    coverage.
  - `python years/2026/scripts/field_recertification_report.py` passed with
    remaining full completion feasible.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 13/13
    requirements, accounting for 251/251 official segment ids.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 30/30
    routes.
  - `python years/2026/scripts/export_example_map.py` regenerated public-safe
    map/menu artifacts.
  - Local Playwright smoke via `http://127.0.0.1:8127/` loaded the public map,
    phone packet, and `docs/field-packet/live-map.html?outing=1-2` with no
    console or page errors.
  - `python -m pytest -q` passed 394 tests.

#### May 8 historical correction, superseded: GPX/card mismatch was briefly a certification failure

- Objective: ensure the live map, field guide link, route card mileage, and GPX
  all describe the same car-to-car artifact instead of letting the map mask or
  compensate for source mismatches.
- Superseded decision: GPX track length is no longer a route-distance or
  certification source. Keep the history below as context for why the team
  checked source drift, but do not restore GPX/card distance comparison.
- Finding: the live map and field guide already used the same user-facing GPX
  href, but the refreshed route artifacts still disagreed with the route cards.
  Example: `1A-2. West Climb` shows 4.11 mi on the card while the field GPX
  measures about 11.33 mi. This is a source/export certifiability problem, not a
  map-display problem.
- Implementation at the time: added a now-removed GPX-length validation check,
  changed the field-guide link copy to `Open Field GPX`, and changed the
  live-map warning copy from `Official GPX` to `Route GPX`.
- Current result: the refreshed packet is not certifiable. Export now marks
  25/31 runnable outings as failed GPX validation due to route-card/GPX mileage
  mismatch, so the failure is visible in the manifest and completion audit
  instead of only in the live map.
- Validation:
  - Added a failing regression for route-card/GPX mileage mismatch, then made it
    pass.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py` passed 41
    tests.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated the
    packet and marked `gpx_validation_passed: false` with 25 failed GPX routes.
  - Extracting `docs/field-packet/live-map.html` script and running
    `node --check /tmp/boise-live-map.js` passed.
  - Local browser check against
    `http://127.0.0.1:8786/live-map.html?outing=1-2&v=gpx-guard` showed the
    route review banner with `Route GPX length 11.30 mi differs from route card
    4.11 mi`.
  - `python years/2026/scripts/field_tool_completion_audit.py` failed as
    expected: 10/13 requirements passed, with GPX validation failures and hidden
    source-gap evidence.

#### May 8 historical correction follow-up, superseded: no placeholder field packet

- Objective: restore the phone packet as a usable field artifact before field
  use, while preserving the rule that the route card and GPX must describe the
  same route.
- Superseded decision: do not derive displayed mileage or time buckets from GPX
  track length. Current route totals come from route distance calculation; GPX is
  navigation geometry for continuity, coverage, and field use.
- Decision: do not publish a placeholder and do not show per-route
  `GPX validation failed` warnings in the field guide. If a stale upstream
  mileage estimate disagrees with the actual generated field GPX, the field
  packet briefly derived the displayed on-foot mileage and time bucket from the
  GPX track. That behavior is now superseded; displayed route totals come from
  route distance calculation.
- Result: `1A-2. West Climb` now shows the actual field-track mileage, 11.33
  mi, instead of the stale 4.11 mi source estimate. The field guide has real
  `Open Field GPX` and `Open Live Map` links and no validation-failed route
  cards.
- Validation:
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py` passed 43
    tests.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated the
    real field packet with `gpx_validation_passed: true` and 0 failed GPX
    routes.
  - Extracting `docs/field-packet/live-map.html` script and running
    `node --check /tmp/boise-live-map.js` passed.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 13/13
    requirements.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 31/31
    routes.
  - `python years/2026/scripts/field_progress_report.py` preserved 251/251
    remaining official segments.
  - `python years/2026/scripts/field_recertification_report.py` passed with
    remaining full completion feasible.

#### May 8 correction follow-up: route card owns distance

- Objective: correct the previous follow-up before field use. The GPX/live map
  is a field-navigation outline, not the authoritative distance model. The
  route card owns planned mileage, p75/p90 time, and effort.
- Decision: do not publish placeholders, do not show per-route
  `GPX validation failed` warnings, and do not overwrite route-card mileage or
  time from GPX geometry length. The GPX and live map must still match the same
  route topology, cue order, parking endpoints, and source-gap evidence, but
  their geometric line length is allowed to be schematic.
- Implementation: removed the route-card/GPX mileage failure and the temporary
  field-track mileage reconciliation. The live map now scales card mileage onto
  the GPX display shape for progress/cue placement while preserving the route
  card values.
- Result: `1A-2. West Climb` is back to the route-card value, 4.11 mi / 113
  min, while the generated Field GPX remains the user-facing route outline.
- Validation:
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated the
    field packet with `gpx_validation_passed: true` and 0 failed GPX routes.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py` passed 43
    tests.
  - Extracting `docs/field-packet/live-map.html` script and running
    `node --check /tmp/boise-live-map.js` passed.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 13/13
    requirements.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 31/31
    routes.
  - `python years/2026/scripts/field_progress_report.py` preserved 251/251
    remaining official segments.
  - `python years/2026/scripts/field_recertification_report.py` passed with
    remaining full completion feasible.

#### May 8 field test: Harrison Hollow full rerun analyzed from Strava

- Objective: pull the latest Strava activity after a Harrison Hollow rerun and
  compare it against the corrected `1B. Harrison Hollow` field card.
- Result: latest Strava pull `2026-05-08-harrison-field-test` found the May 8
  `Lunch Run` at 6.46 mi, 1:41:15 moving, 1:58:52 elapsed recording, and
  1,186 ft gain. User-reported door-to-door time was 2:11:06.34.
- Segment evidence: local geometry matching found 12/12 planned `1B` official
  segments and 4.72/4.72 planned official miles, plus the small extra `Buena
  Vista Trail 5` segment at 0.14 mi. This is planning evidence only, not
  official BTC credit, because the challenge window has not started.
- Timing evidence: the corrected 141-minute p75 card looks conservative but
  usable; actual door-to-door was about 9.9 minutes under p75 and 26.9 minutes
  under p90.
- Artifact: public-safe field-test notes were added under
  `years/2026/field-tests/pre-challenge/2026-05-08-test-03/`.
- Current blocker: decide whether the 0.14 mi `Buena Vista Trail 5` match should
  become explicit expected extra progress on the Harrison card, or remain
  incidental post-run evidence.

#### May 8 field learning: live map Fit GPS and cue stepping

- Objective: apply field feedback from the Harrison run to the generated live
  map controls.
- Implementation: `Fit GPS` now fits the viewport between the current GPS fix
  and the next upcoming cue instead of fitting the GPS point plus the full
  route. Cue Prev/Next controls now move between distinct cue-leg starts, so
  duplicate same-location cue stops do not require a double tap to reach the
  next runnable segment.
- Validation:
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py` passed 43
    tests.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated the
    field packet and GPX zip.
  - Extracting `docs/field-packet/live-map.html` script and running
    `node --check /tmp/boise-live-map.js` passed.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 13/13
    requirements.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 31/31
    routes.
  - `python years/2026/scripts/field_progress_report.py` preserved 251/251
    remaining official segments.
  - `python years/2026/scripts/field_recertification_report.py` passed with
    remaining full completion feasible.
  - Local Playwright smoke against
    `http://127.0.0.1:8786/live-map.html?outing=1-3&v=fit-gps-cue-step` loaded
    Harrison, acquired a mocked GPS fix, confirmed `Fit GPS` produced a smaller
    viewport than full-route fit, and confirmed `Next cue` advanced from cue 02
    to cue 03 without console errors.

#### May 8 field learning: Harrison same-trail overlap cue

- Objective: address the confusing `1B. Harrison Hollow` overlap where cue 7
  doubles back on `#51 Who Now` before cue 8 exits onto `#58 Harrison Ridge`.
- Implementation: the generated field data now marks cue 7 as `OVERLAP DOUBLE
  BACK`, adds a field warning to cue 7 and the cue 8 exit, and surfaces cue
  warnings in the live map active-leg banner and cue card. `AGENTS.md` now
  records the general rule: do not offset exported GPX geometry to hide
  same-trail repeats; label the overlap and use active-leg arrows/current-next
  cue context to disambiguate it.
- Validation:
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py` passed 44
    tests.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated the
    field packet and GPX zip.
  - Extracting `docs/field-packet/live-map.html` script and running
    `node --check /tmp/boise-live-map.js` passed.
  - Generated `docs/field-packet/field-tool-data.json` now shows cue 7 as
    `overlap_repeat` / `DOUBLE BACK` and cue 8 with an exit-overlap warning.
  - `python years/2026/scripts/field_tool_completion_audit.py`,
    `field_route_walkthrough_audit.py`, `field_progress_report.py`, and
    `field_recertification_report.py` all passed, preserving 251/251 remaining
    segment coverage.
  - Local Playwright smoke against
    `http://127.0.0.1:8787/live-map.html?outing=1-3&v=overlap-warning-smoke-2`
    selected `1B. Harrison Hollow`, activated cue 7, confirmed the `DOUBLE BACK
    07 -> 08` banner and warning text, and found no console errors.

#### May 8 hardening: generic same-corridor overlap detector

- Objective: make double-back/overlap warnings durable for future route changes,
  not only the known Harrison cue 7 case.
- Implementation: `export_mobile_field_packet.py` now compares cue-leg geometry
  against earlier cue-leg geometry and automatically marks later non-credit
  connector/access double-backs as overlap warnings when enough of the leg
  reuses a previous GPS corridor in the opposite direction. This preserves the
  exported GPX geometry and adds structured `overlap_match` metadata plus
  phone-visible `OVERLAP` / `DOUBLE BACK` cue language.
- Result: the regenerated packet contains 32 auto-detected overlap cues across
  26 route cards, plus the two Harrison-specific warnings.
- Validation:
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py` passed 45
    tests.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated the
    field packet and GPX zip.
  - Extracting `docs/field-packet/live-map.html` script and running
    `node --check /tmp/boise-live-map.js` passed.
  - `python years/2026/scripts/field_tool_completion_audit.py`,
    `field_route_walkthrough_audit.py`, `field_progress_report.py`, and
    `field_recertification_report.py` all passed.
  - Local Playwright smoke against
    `http://127.0.0.1:8787/live-map.html?outing=1-3&v=durable-overlap-smoke`
    confirmed the Harrison-specific `DOUBLE BACK 07 -> 08` banner, then
    `outing=5-2` confirmed an automatically detected `DOUBLE BACK 05 -> 06`
    banner on `5B. Cartwright`, with no console errors.

#### May 8 field learning: Harrison #52 grade asymmetry

- Objective: explain why the Harrison field route used `#52 Kemper's Ridge`
  through the cue 04 -> 05 leg and make steep reverse-direction risk visible in
  the phone packet.
- Finding: the user's Strava trace between cue 04 and cue 05 was about 0.038 mi
  shorter than the generated GPX for the same endpoint pair, roughly 201 ft.
  The larger field issue was not distance; it was grade asymmetry. The planned
  direction over the required `#52` official segments is about 170 ft climb and
  482 ft descent over 0.80 mi, while reversing that leg would require about 482
  ft climb over the same distance.
- Implementation: `export_mobile_field_packet.py` now backfills missing
  cue-level official segment effort from the DEM-derived segment elevation
  table, adds descent to cue notes when it materially matters, and emits a
  phone-visible `field_warning` when the reverse direction would be steep.
- Validation:
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py` passed 46
    tests.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated the
    field packet and GPX zip.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 13/13
    requirements.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 31/31
    routes.
  - `python years/2026/scripts/field_progress_report.py` preserved 251/251
    remaining official segments.
  - `python years/2026/scripts/field_recertification_report.py` passed.
  - Local Playwright smoke against
    `http://127.0.0.1:8788/live-map.html?outing=1-3` confirmed cue 04 shows
    `Reverse direction would be steep: about 482 ft climb over 0.8 mi.` in both
    the active-leg banner and cue card.

#### May 8 failure mode: repeat connector treated as mandatory

- Objective: capture the Harrison `#53 Buena Vista` miss as a route-choice
  failure, not just a field-label problem.
- Finding: the route can be official-credit-correct while still field-wrong.
  Once a connector or official repeat has already served its credit/access
  purpose, the next non-credit movement should be re-optimized as a legal
  connector choice. In this case the map showed `#53 Buena Vista` in both
  directions; after the first pass, the route source should have compared the
  shorter legal onward path against repeating the same connector, with elevation
  cost made explicit.
- Implementation: `AGENTS.md` now records this as a named failure mode and
  routing heuristic. The phone exporter now anchors cues to true GPX route miles
  rather than card-mile scaling, and it warns when an active official cue also
  contains connector/repeat trail mileage such as `#53 Buena Vista Trail 5`.
- Boundary: this pass makes the field artifact expose the mismatch and prevents
  cue slicing from hiding it. The upstream route-building pass still needs a
  route-choice improvement so it can replace unnecessary post-credit repeats
  with the shortest legal/elevation-aware connector.
- Validation:
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py` passed 47
    tests.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated the
    field packet and GPX zip.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 13/13
    requirements.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 31/31
    routes.
  - `python years/2026/scripts/field_progress_report.py` preserved 251/251
    remaining official segments.
  - `python years/2026/scripts/field_recertification_report.py` passed.
  - Local Playwright smoke against
    `http://127.0.0.1:8788/live-map.html?outing=1-3` confirmed cue 04 uses the
    true `+1.29 mi` GPX span and surfaces the `#53 Buena Vista Trail 5`
    connector/repeat warning.

#### May 8 implementation: segment-first progress and versioned active state

- Objective: stop applying challenge progress outing-first and preserve locked
  original baselines while recalculating active routes after field tests.
- Implementation:
  - Added `years/2026/scripts/field_activity_review.py` to review a Strava/BTC
    JSON or GPX activity against all official 2026 foot segments, separating
    completed, missed, partial, extra completed, and blocked segment state.
  - Added `years/2026/scripts/field_progress_versions.py` with
    `lock-original`, `apply-day`, and `reset-epoch` commands. It keeps the
    private progress ledger under `years/2026/inputs/personal/private/`, writes
    epoch/day snapshots under `years/2026/outputs/private/progress/versions/`,
    materializes the active private planner state from the locked original plus
    ledger, derives completed outings from completed segment ids, and can
    regenerate/copy the active phone packet into the day snapshot.
  - Changed `field_progress_report.py` so completed outings are derived from
    validated segment state. Phone `completed_outing_ids` remain provisional,
    and blocked-only/no-new-credit outings are inactive rather than completed.
  - Changed `export_mobile_field_packet.py` so phone progress export writes
    `completed_segment_ids: []` and `provisional_completed_segment_ids` for
    local card taps; only validated `completed_segment_ids` and
    `extra_completed_segment_ids` are applied to the active remaining packet.
  - Extended `reset_challenge_start.py` with `--lock-original-epoch` so the real
    `challenge-2026` baseline can be locked immediately after reset.
- Validation:
  - Initial red test run showed the expected failures for outing-first
    completion, phone tap promotion, missing activity review/version scripts,
    and reset-record epoch locking.
  - `pytest -q years/2026/tests/test_field_progress_report.py
    years/2026/tests/test_field_recertification_report.py
    years/2026/tests/test_export_mobile_field_packet.py
    years/2026/tests/test_reset_challenge_start.py
    years/2026/tests/test_field_activity_review.py
    years/2026/tests/test_field_progress_versions.py` passed 69 tests.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated the
    active phone packet; generated `docs/field-packet/index.html` now keeps phone
    taps provisional.
  - `python years/2026/scripts/field_progress_report.py` reported 251 remaining
    official segments, 0 completed segments, and remaining coverage preserved.
  - `python years/2026/scripts/field_recertification_report.py` passed with
    remaining full completion feasible.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 13/13
    requirements.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 31/31
    routes.

#### May 11 audit: frame-shift is not yet a hard gate

- Objective: answer whether `frame-shift` is actually preventing "good enough"
  planner decisions after the Dry Creek / Sweet Connie OSM parking miss.
- Finding: the skill is working as a vocabulary and review posture, but the BTC
  pipeline still allows broad proxy statuses such as `field_ready`,
  `graph_validated`, `source_gap_warning=false`, `certified_route_card`, and
  `simulated_ready` to stand in for stronger source-layer evidence. The Dry
  Creek case was not a physical-route miss; it was a provenance/enrichment miss
  caused by siloed evidence lanes.
- Artifact: `years/2026/checkpoints/frame-shift-good-enough-audit-2026-05-11.md`
  records the failure pattern, adjacent frames, adversarial stories, and the
  required promotion-gate changes.
- Boundary: this is not a fix to the planner yet. The next durable repair is a
  normalized parking/access evidence inventory plus an OSM `amenity=parking`
  enrichment pass that updates existing anchors before route promotion.
- Validation: no tests were run for this audit note.

#### May 11 implementation: latent official-credit audit gate

- Objective: turn the `16A-2` Shingle latent-credit discovery into a reusable
  field-packet guard instead of another one-route note.
- Implementation:
  - Added `years/2026/scripts/field_latent_credit_audit.py` to review each
    generated route GPX against nearby official segments, detect extra full
    completions, and cross-reference those segments against every active route
    card's claimed segment ids.
  - Added focused tests for a GPX that completes another active route's segment
    and for repeat-only credit already completed at export.
  - Added `field_latent_credit_audit.py` to the field-packet certification
    chain and recorded the reusable heuristic/failure/eval cases.
- Result:
  - The current generated field packet correctly fails the new gate:
    `needs_repair`, 30 routes audited, 15 routes needing repair, 43 unexpected
    latent official segments, all 43 claimed by another active route, and no
    unclaimed uncompleted extras.
  - `15A-1` is the expected high-value case: its GPX completes Shingle segment
    `1656`, still claimed by `16A-2`.
  - `16A-2` also traverses Dry Creek segments `1542`, `1543`, and `1544`,
    still claimed by `15A-1`, confirming the current menu has cross-route
    segment-ownership debt rather than just a single bad card.
- Artifacts:
  - `years/2026/checkpoints/field-latent-credit-audit-2026-05-11.json`
  - `years/2026/checkpoints/field-latent-credit-audit-2026-05-11.md`
- Validation:
  - `pytest -q years/2026/tests/test_field_latent_credit_audit.py` passed 2
    tests.
  - `pytest -q years/2026/tests/test_field_activity_review.py
    years/2026/tests/test_field_latent_credit_audit.py` passed 4 tests.
  - `python years/2026/scripts/field_latent_credit_audit.py` wrote the dated
    checkpoint artifacts and exited nonzero as expected because the active
    packet now has detected latent-credit repairs pending.

#### May 11 implementation: cross-route ownership reconciliation and certification recovery

- Objective: repair the latent-credit class generically after the `16A-2` /
  `15A-1` cross-route ownership finding, then regenerate and rerun the field
  certification chain.
- Implementation:
  - Extended `export_mobile_field_packet.py` so each generated route reviews its
    GPX against nearby official segments and records latent full-segment
    completions as either owned by another active route card or source-repair
    debt.
  - Updated `field_latent_credit_audit.py` so declared cross-route ownership is
    reconciled, while undeclared or unclaimed latent completions still fail.
  - Fixed the multi-start replacement generator so `field_ready` parking anchors
    do not lose their verified parked-start status when rebuilt through a forced
    trailhead state.
  - Added wayfinding-mileage reconciliation in the phone packet so displayed cue
    miles remain aligned to the route-card on-foot mileage while preserving the
    original cue-mile source values.
- Result:
  - `python years/2026/scripts/multi_start_field_menu_replacements.py` regenerated
    the private multi-start replacement source with 4 accepted replacements.
  - `python years/2026/scripts/human_loop_plan.py` regenerated the canonical
    private outing-menu data/map/menu.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated 90 GPX
    files and the phone field packet.
  - `python years/2026/scripts/field_latent_credit_audit.py` passed: 30 routes,
    15 reconciled routes, 43 latent official segments reconciled to other active
    owner cards, and 0 routes needing repair.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 13/13
    requirements; `python years/2026/scripts/field_route_walkthrough_audit.py`
    passed 30/30 routes.
- Validation:
  - `pytest -q years/2026/tests/test_multi_start_alternative_audit.py
    years/2026/tests/test_export_mobile_field_packet.py` passed 78 tests.
  - `pytest -q years/2026/tests/test_field_activity_review.py
    years/2026/tests/test_field_latent_credit_audit.py
    years/2026/tests/test_multi_start_alternative_audit.py
    years/2026/tests/test_export_mobile_field_packet.py` passed 83 tests.

#### May 11 baseline reset: cleared completed segments and repaired stale generated layers

- Objective: reset the pre-challenge completed-segment baseline so the active
  phone packet starts from 0 completed and 251 remaining official segments.
- Implementation:
  - Ran `python years/2026/scripts/field_progress_versions.py reset-epoch
    --epoch pre-challenge-testing --clear-blocks`, which cleared
    `2026-planner-state.private.json`, cleared the progress ledger, and wrote a
    fresh pre-challenge-testing original lock/reset record.
  - Fixed `human_loop_plan.py` so regenerated map data syncs top-level progress
    from the active private state instead of copying stale package-map progress.
  - Fixed `human_loop_plan.py` so route-package summaries and the official
    segment feature layer are recomputed from the active 31-component field
    menu, using the authoritative 2026 official segment GeoJSON for the rendered
    official layer.
  - Re-exported the phone field packet after the source fixes.
- Result:
  - Source state, progress ledger, generated map data, and exported
    `field-tool-data.json` now agree on 0 completed segments, 0 blocked
    segments, and 251 remaining official segments.
  - Active map/menu baseline: 19 packages, 31 route components, 251 unique
    official segments, 164.42 official miles, 263.98 on-foot miles, 1.61x
    on-foot/official ratio, and 0 manual route-design holds.
  - Proof boundary: this reset and reconciliation pass made the active plan
    more executable and auditable. It did not prove a net human-effort
    reduction; proving that would require a route-card replacement or field-day
    repricing after validated progress changes the remaining segment set.
  - The rendered official feature collection now has 251 unique official segment
    features instead of the stale 238-feature hybrid source layer.
  - The legacy full `reset_challenge_start.py` pipeline still fails at
    `block_hybrid_route_pass.py` with an infeasible HiGHS set-cover result; the
    active canonical field-menu path was regenerated through `human_loop_plan.py`
    after the reset.
- Validation:
  - `python years/2026/scripts/human_loop_plan.py --field-menu-overrides-json
    years/2026/inputs/personal/private/2026-field-menu-replacements-v2-multi-start.private.json`
    regenerated the canonical map/menu artifacts.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated 93 GPX
    files and the phone field packet.
  - `python years/2026/scripts/field_progress_report.py` passed with 0
    completed, 251 remaining, and `certified_baseline_status: passed`.
  - `python years/2026/scripts/field_recertification_report.py` passed with
    `remaining_full_completion_feasible: true`.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 13/13
    requirements with 251 accounted segments.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 31/31
    routes.
  - `python years/2026/scripts/field_latent_credit_audit.py` passed with 16
    reconciled routes, 44 reconciled claimed-elsewhere latent segments, and 0
    routes needing repair.
  - `pytest -q years/2026/tests/test_human_loop_plan.py
    years/2026/tests/test_export_mobile_field_packet.py
    years/2026/tests/test_field_latent_credit_audit.py
    years/2026/tests/test_field_progress_report.py
    years/2026/tests/test_field_recertification_report.py
    years/2026/tests/test_field_tool_completion_audit.py
    years/2026/tests/test_field_route_walkthrough_audit.py` passed 114 tests.

#### May 11 15A/16A planning-level net-effort proof

- Objective: answer whether the 15A-1 latent Shingle coverage plus a Sheep
  Camp-only 16A-2 replacement provably reduces net human effort across the full
  active field menu, not just inside one route card.
- Implementation:
  - Added `years/2026/scripts/net_effort_reduction_proof.py` to compute a
    full-menu before/after from the active field packet, the refreshed 15A-1
    activity review, and the existing Sheep Camp single-segment access probe.
  - Added `years/2026/tests/test_net_effort_reduction_proof.py` to fail the
    proof when latent Shingle credit is missing or Sheep-only timing/effort
    evidence is incomplete.
- Result:
  - `years/2026/checkpoints/15a-16a-net-effort-reduction-proof-2026-05-11.md`
    proves a planning-level full-menu reduction: 263.98 -> 252.32 on-foot
    miles, 6336 -> 6132 p75 minutes, and 7111 -> 6882 p90 minutes while
    preserving 251 unique official segments.
  - Scope boundary: this does not prove official BTC app credit before a real
    challenge-window activity is validated, and it does not promote the active
    field packet. Promotion still needs a source route-card replacement,
    regeneration, human-validity review, and day-of access/condition checks.
- Validation:
  - `python -m py_compile years/2026/scripts/net_effort_reduction_proof.py`
    passed.
  - `pytest -q years/2026/tests/test_net_effort_reduction_proof.py` passed 3
    tests.
  - `python years/2026/scripts/field_activity_review.py --activity
    docs/field-packet/gpx/audit/15a-1-dry-creek-sweet-connie-roadside-parking-dry-creek-trail.gpx
    --planned-outing-id 15-1 --planned-segment-ids 1542,1543,1544,1545,1546
    --output-json years/2026/checkpoints/15a-1-latent-shingle-credit-review-2026-05-11.json`
    completed with 6 completed, 1 extra, 0 missed, and 2 partial segments.
  - `python years/2026/scripts/field_activity_review.py --activity
    docs/field-packet/gpx/audit/16a-2-dry-creek-sweet-connie-roadside-parking-sheep-camp-trail-shingle-creek-trail.gpx
    --planned-outing-id 16-2 --planned-segment-ids 1656,1653 --output-json
    years/2026/checkpoints/16a-2-activity-review-current-route-2026-05-11.json`
    completed with 5 completed, 3 extra, 0 missed, and 3 partial segments.
  - `python years/2026/scripts/net_effort_reduction_proof.py` wrote the proof
    artifacts with status `proved_planning_net_effort_reduction`.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 13/13
    requirements with 251 accounted segments.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 31/31
    routes with 0 failures.
  - `python years/2026/scripts/field_latent_credit_audit.py` passed with 31
    routes, 16 reconciled routes, 44 reconciled claimed-elsewhere latent
    segments, and 0 routes needing repair.
  - `pytest -q years/2026/tests/test_net_effort_reduction_proof.py
    years/2026/tests/test_field_latent_credit_audit.py` passed 6 tests.

#### May 11 15A/16A active route promotion

- Objective: if the Shingle-to-15A-1 / Sheep-only-16A-2 repair is better,
  promote it into the active official route cards and log the what/why.
- Decision:
  - Promoted. `15A-1` now claims Shingle Creek official segment `1656`; `16A-2`
    now carries Sheep Camp segment `1653` only.
  - Reason: `15A-1` covers `1656` end-to-end in the required ascent direction,
    so keeping Shingle on `16A-2` was duplicate human effort. The promoted menu
    preserves all `251` official segments and reduces full-menu route-card
    effort from `263.98` to `252.33` on-foot miles, `6336` to `6132` p75
    minutes, and `7111` to `6882` p90 minutes.
- Implementation:
  - Added
    `years/2026/inputs/personal/2026-cross-package-segment-promotions-v1.json`
    as the evidence-gated source for the segment ownership change.
  - Updated `years/2026/inputs/personal/2026-manual-route-designs-v1.json` so
    package 16 records Shingle as covered elsewhere and keeps `16A-2` to Sheep
    Camp.
  - Updated `multi_start_field_menu_replacements.py` to apply generic
    cross-package segment ownership promotions instead of hard-coding this
    route.
  - Updated `manual_route_design_pass.py` and `human_loop_plan.py` so manual
    route promotion can account for covered-elsewhere segment ids.
  - Regenerated the route replacements, manual route report, human-loop
    map/menu, field-day layer, and mobile phone field packet.
  - Fixed `export_field_day_layer.py` so the default phone field-day layer uses
    current certified route-card values for promoted loops; this removed the
    stale Shingle+Sheep label, old segment set, old mileage, old p75/p90,
    stale stress, and old GPX link from `16A-2`.
  - Updated `export_mobile_field_packet.py` so field-day loop records retain
    segment ids in public packet data.
- Result:
  - `15A-1`: `1542,1543,1544,1545,1546,1656`; 11.73 official miles, 11.89
    on-foot miles, 229 p75, 257 p90.
  - `16A-2`: `1653`; 0.77 official miles, 3.31 on-foot miles, 106 p75, 119
    p90.
  - Full route-card menu: 31 routes, 251 official segments, 252.33 on-foot
    miles, 6132 p75 minutes, 6882 p90 minutes.
  - Field-day layer `16A-2`: `executable_route_card`, 3.31 on-foot miles, 106
    p75, 119 p90, stress 0.331.
  - Broader field-day layer publication status is still
    `needs_route_card_promotion` because 35 unrelated non-route-card loops from
    the dated schedule remain unpromoted.
- Condition/access check:
  - Checked Ridge to Rivers home, condition reports, interactive map entrypoint,
    and the 2024 R2R map PDF. No static current page found a Dry Creek,
    Shingle Creek, Sheep Camp, or Sweet Connie closure during this promotion
    check. The visible current note was Owl's Roost/The Grove repair work.
  - This is not a day-of field clearance. R2R interactive map/RainoutLine,
    posted signage, heat, water, and parking checks remain required before
    running. The R2R map lists Sweet Connie as a trail to avoid during wet,
    winter, or marginal conditions.
- Promotion checkpoint:
  - `years/2026/checkpoints/15a-16a-route-promotion-2026-05-11.md`
  - `years/2026/checkpoints/15a-16a-route-promotion-2026-05-11.json`
- Validation:
  - JSON validation passed for the new promotion source, manual route designs,
    regenerated field-day layer, field tool data, and field-packet manifest.
  - `python years/2026/scripts/field_progress_report.py` passed with 251
    remaining and coverage preserved.
  - `python years/2026/scripts/field_recertification_report.py` passed with
    `remaining_full_completion_feasible: true`.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 13/13
    requirements with 251 accounted segments.
  - `python years/2026/scripts/field_latent_credit_audit.py` passed with 31
    routes, 0 routes needing repair, and 42 reconciled claimed-elsewhere latent
    segments.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 31/31
    routes.
  - `python years/2026/scripts/field_activity_review.py --activity
    docs/field-packet/gpx/audit/15a-1-dry-creek-sweet-connie-roadside-parking-dry-creek-trail-shingle-creek-trail.gpx
    --planned-outing-id 15-1 --planned-segment-ids
    1542,1543,1544,1545,1546,1656 --output-json
    years/2026/checkpoints/15a-1-promoted-shingle-activity-review-2026-05-11.json`
    passed with 6 completed, 0 extra, 0 missed, and 2 partial.
  - `python years/2026/scripts/field_activity_review.py --activity
    docs/field-packet/gpx/audit/16a-2-dry-creek-sweet-connie-roadside-parking-sheep-camp-trail.gpx
    --planned-outing-id 16-2 --planned-segment-ids 1653 --output-json
    years/2026/checkpoints/16a-2-promoted-sheep-only-activity-review-2026-05-11.json`
    passed for the planned Sheep segment with 3 completed total, 2 extra Dry
    Creek repeat segments, 0 missed, and 2 partial.
  - `pytest -q years/2026/tests/test_multi_start_field_menu_replacements.py
    years/2026/tests/test_manual_route_design_pass.py
    years/2026/tests/test_human_loop_plan.py
    years/2026/tests/test_export_field_day_layer.py
    years/2026/tests/test_export_mobile_field_packet.py
    years/2026/tests/test_field_tool_completion_audit.py` passed 98 tests in
    111.34s.
- Aborted attempt:
  - Tried `python years/2026/scripts/p90_near_miss_pressure_audit.py
    --inter-trailhead-drive-minutes 45 --neighbor-limit 40 --basename
    p90-near-miss-pressure-audit-drive45-n40-2026-05-06`; stopped after about
    seven minutes without a result. It is not used as promotion evidence.

## 2026-05-11 - Promote selected field-day loops into route cards

- Objective:
  - Explain and remove the live `needs_route_card_promotion` labels in the
    field-day packet.
  - Promote selected field-day loops into canonical route-card source records
    only where the loop already had access, timing, geometry, and coverage
    evidence.
- What changed:
  - Added `years/2026/scripts/promote_field_day_loops.py` to convert selected
    field-day loops into the canonical map-data shape consumed by
    `export_mobile_field_packet.py`.
  - Preserved already certified route cards instead of regenerating them from
    draft candidates.
  - Added explicit promotion-report support to `export_field_day_layer.py` so a
    stale selected loop can be mapped to its certified superset route card,
    including the 15A-1 Shingle ownership replacement.
  - Updated `promote_field_day_loops.py` to regenerate the private map HTML and
    written menu from the promoted JSON source so public map exports and the
    phone packet do not diverge.
- Result:
  - Promoted route-card source: 31 field-day packages, 50 route-card loops,
    251/251 official segments covered.
  - Field-day layer: 50 certified route-card loops, 0 route-card audit-fix
    loops, 0 route-card promotion gaps.
  - Phone packet: 50 route cards, 150 GPX files, 251 field-menu segments, GPX
    validation passed.
  - Public sanitized map data now matches the promoted source: 31 packages and
    50 components.
  - Multi-loop days still carry `needs_day_gpx_validation` as a separate
    day-level handoff status; the route-card promotion gaps are cleared.
- Evidence artifacts:
  - `years/2026/checkpoints/field-day-loop-promotion-2026-05-11.md`
  - `years/2026/checkpoints/field-day-loop-promotion-2026-05-11.json`
  - `years/2026/checkpoints/human-executable-field-day-layer-2026-05-10.md`
  - `years/2026/checkpoints/field-latent-credit-audit-2026-05-11.md`
  - `years/2026/checkpoints/field-tool-completion-audit-2026-05-06.md`
  - `years/2026/checkpoints/field-route-walkthrough-audit-2026-05-06.md`
- Validation:
  - `python years/2026/scripts/human_loop_plan.py` regenerated the canonical
    pre-promotion source with current 15A/16A ownership and 251/251 coverage.
  - `python years/2026/scripts/promote_field_day_loops.py` passed with 50
    route-card source loops, 251 covered segments, track validation passed, and
    0 source gap warnings.
  - `python years/2026/scripts/export_mobile_field_packet.py` wrote 150 GPX
    files and regenerated `docs/field-packet/`.
  - `python years/2026/scripts/export_field_day_layer.py` passed with 50
    certified route-card loops and 0 promotion gaps.
  - `python years/2026/scripts/field_latent_credit_audit.py` passed with 50
    routes and 0 routes needing repair.
  - `python years/2026/scripts/field_progress_report.py` passed with 251
    remaining segments and coverage preserved.
  - `python years/2026/scripts/field_recertification_report.py` passed with
    `remaining_full_completion_feasible: true`.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 13/13
    requirements with 251 accounted segments.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 50/50
    routes.
  - `python years/2026/scripts/export_example_map.py` regenerated the public
    sanitized map/menu from the promoted private views.
  - JSON validation passed for `outing-menu-map-data.json`,
    `years/2026/outputs/examples/2026-outing-menu-map-data.example.json`, and
    `docs/field-packet/field-tool-data.json`.
  - `pytest -q years/2026/tests/test_export_field_day_layer.py
    years/2026/tests/test_promote_field_day_loops.py
    years/2026/tests/test_export_mobile_field_packet.py
    years/2026/tests/test_field_progress_report.py
    years/2026/tests/test_field_recertification_report.py
    years/2026/tests/test_field_tool_completion_audit.py` passed 91 tests in
    106.66s.

## 2026-05-11 - Field-day timing authority repair

- Objective:
  - Fix the field-day layer regression where promoted route-card overlays made
    a passed 31-day p90 certificate look over-bound by summing door-to-door
    route-card timings across multi-start days.
- What changed:
  - Kept route-card overlays authoritative for labels, route-card refs,
    certified loop status, segment ownership, and route-card navigation.
  - Restored the calendar assignment/certificate as the authoritative source
    for field-day p75/p90 timing.
  - Added explicit diagnostic fields for route-card door-to-door timing sums and
    the legacy recomputed timing, so the old double-count pattern is visible
    without controlling day feasibility.
  - Added a hard generation guard: a passed certified baseline plus passed
    assignment cannot emit field days whose p90 exceeds their bound.
- Result:
  - Field-day layer now reports 31 field days, 50 certified loops, 251/251
    official segments, total p75 7,684 minutes, max p90 359 minutes, and 0
    schedule p90 violation days.
  - Public field packet publication status is `field_day_certified`, with
    `day_gpx_validation_passed: true`.
- Evidence artifacts:
  - `years/2026/checkpoints/human-executable-field-day-layer-2026-05-10.md`
  - `docs/field-packet/field-tool-data.json`
- Validation:
  - `python years/2026/scripts/export_field_day_layer.py` regenerated the
    field-day layer with 0 schedule p90 violations.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated
    `docs/field-packet/`.
  - `python years/2026/scripts/field_latent_credit_audit.py` passed.
  - `python years/2026/scripts/field_progress_report.py` passed with 251
    remaining segments and coverage preserved.
  - `python years/2026/scripts/field_recertification_report.py` passed with
    `remaining_full_completion_feasible: true`.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 13/13
    requirements with 251 accounted segments.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 50/50
    routes.
  - `pytest -q years/2026/tests/test_export_field_day_layer.py
    years/2026/tests/test_promote_field_day_loops.py
    years/2026/tests/test_export_mobile_field_packet.py
    years/2026/tests/test_field_progress_report.py
    years/2026/tests/test_field_recertification_report.py
    years/2026/tests/test_field_tool_completion_audit.py` passed 93 tests in
    105.02s.

## 2026-05-11 - Official repeat artifact audit

- Objective:
  - Prove the final 50 route cards are not hiding official repeat mileage in
    `start_access`, `between_links`, or `return_to_car`, and make exports fail
    when repeat official mileage lacks segment IDs or repeat/no-credit cue text.
- What changed:
  - Added an artifact-level `field_official_repeat_audit.py` with Bucket A/B/C
    classification over source route cues and public phone-packet cues.
  - Preserved `official_repeat_segment_ids` through route cue packaging,
    multi-start field replacements, source-map promotion, and phone wayfinding.
  - Added a geometry-backed enrichment pass for older canonical route-cue
    sources that had repeat official mileage but no segment IDs.
  - Fixed field-day route-card matching so ambiguous trailhead/trail matches
    cannot map Spring Creek 1 onto the Spring Creek 2 card.
- Result:
  - Final repeat audit passed: 50 routes, 102 source repeat legs, 102 public
    repeat cues, Bucket A hidden self-repeat count 0, Bucket B counted repeat
    / optimization-target count 102, Bucket C reconciled extra-credit route
    count 25, repeat legs missing segment IDs 0, repeat cues missing repeat/no
    credit text 0, unreconciled extra-credit segment count 0.
  - Coverage remained 251/251 in both `outing-menu-map-data.json` and
    `docs/field-packet/field-tool-data.json`.
- Evidence artifacts:
  - `years/2026/checkpoints/field-official-repeat-audit-2026-05-11.md`
  - `years/2026/checkpoints/field-official-repeat-audit-2026-05-11.json`
  - `years/2026/checkpoints/field-official-repeat-audit-2026-05-11-manifest.json`
- Validation:
  - Initial `python years/2026/scripts/field_official_repeat_audit.py` failed
    before repair with 49 repeat legs missing segment IDs, proving the audit
    caught the artifact-level gap.
  - Final `python years/2026/scripts/field_official_repeat_audit.py` passed
    with 0 missing repeat IDs and 0 repeat/no-credit text gaps.
  - `python years/2026/scripts/promote_field_day_loops.py` passed with 50
    route-card source loops, 251 covered segments, track validation passed, and
    0 source gap warnings.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated 150
    GPX files and the phone packet.
  - `python years/2026/scripts/export_example_map.py` regenerated the public
    sanitized map/menu artifacts.
  - `python years/2026/scripts/export_field_day_layer.py` passed with 50
    certified route-card loops, 251 covered official segments, 0 missing
    segments, and 0 route-card promotion gaps.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 13/13
    requirements with 251 accounted segments.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 50/50
    routes.
  - `python years/2026/scripts/field_latent_credit_audit.py` passed with 0
    routes needing repair and 0 unclaimed uncompleted segments.
  - `python years/2026/scripts/field_progress_report.py` preserved 251
    remaining segments.
  - `python years/2026/scripts/field_recertification_report.py` passed with
    `remaining_full_completion_feasible: true`.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py
    years/2026/tests/test_block_day_packager.py
    years/2026/tests/test_field_official_repeat_audit.py
    years/2026/tests/test_promote_field_day_loops.py
    years/2026/tests/test_export_field_day_layer.py` passed 79 tests in
    105.64s.

## 2026-05-12 - Route repeat optimization audit

- Objective:
  - Add a generated audit that catches route cards where official segment
    repeat mileage is hidden inside access, connector, or return legs, and use
    its warnings to rank the remaining high-burden route cards for manual
    redesign.
- What changed:
  - Added `route_repeat_optimization_audit.py` to compare claimed route
    segments, GPX-derived full official coverage, declared repeat segments, and
    declared owned-elsewhere segment ownership decisions.
  - Tightened the phone-packet exporter so non-credit wayfinding cues declare
    claimed official segments that are fully rerun inside their route-mile
    interval, using the same DEM-backed segment review as the audit.
  - Kept zero-rounded repeat segment IDs visibly priced at `0.01` mi so repeat
    accounting cannot disappear from the route card.
- Result:
  - Initial route-repeat audit failed before repair with 35 failed routes, 39
    hidden self-repeat segment IDs, and 105 unpriced repeat segment IDs.
  - Final route-repeat audit passed across 50 routes with 0 hidden self-repeat
    segments, 0 latent-credit segments, 0 unpriced repeat segments, and 59
    optimization warnings for follow-up route-effort review.
  - Highest non-credit burden routes remain FD30A, FD24A, FD04A, FD20A, 18,
    16A-1, and 12.
- Evidence artifacts:
  - `years/2026/checkpoints/route-repeat-optimization-audit-2026-05-12.md`
  - `years/2026/checkpoints/route-repeat-optimization-audit-2026-05-12.json`
  - `years/2026/checkpoints/route-repeat-optimization-audit-2026-05-12-manifest.json`
- Validation:
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py::test_make_wayfinding_cue_prices_zero_rounded_repeat_ids years/2026/tests/test_export_mobile_field_packet.py::test_non_credit_claimed_repeat_declarations_add_hidden_self_repeat years/2026/tests/test_route_repeat_optimization_audit.py` passed 6 tests in 0.68s.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated 150
    GPX files and `docs/field-packet/`.
  - `python years/2026/scripts/route_repeat_optimization_audit.py` passed.
  - `python years/2026/scripts/field_official_repeat_audit.py` passed.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 13/13
    requirements with 251 accounted segments.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 50/50
    routes.
  - `python years/2026/scripts/field_latent_credit_audit.py` passed with 0
    routes needing repair and 0 unclaimed uncompleted segments.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py
    years/2026/tests/test_route_repeat_optimization_audit.py
    years/2026/tests/test_field_official_repeat_audit.py` passed 62 tests in
    108.25s.

## 2026-05-12 - Latent-credit delta repricing

- Objective:
  - Determine whether reconciled latent official credit actually reduces future
    field work, rather than merely documenting overlap.
- What changed:
  - Added `latent_credit_delta_repricing_audit.py` to group reconciled latent
    credit by source route and owner route, compute pairwise route-removal
    opportunities, and simulate the current field-day order.
  - The audit prices only full route-card removals as proven savings. Partial
    shrink cases are surfaced as replacement-route work with zero proven
    p75/p90 savings until a generated route card exists.
- Result:
  - The audit found 45 latent route relationships covering 47 unique latent
    official segments.
  - Pairwise, 8 relationships can remove the owner route if the source route is
    run first; 37 are partial shrink candidates.
  - In the current field-day order, 2 future route cards are directly removable:
    FD14C after FD14B, and FD22A after route 12/4A latent credit.
  - Proven current-calendar savings: 4.39 on-foot miles, 147 p75 minutes, and
    166 p90 minutes.
  - The high-value partial reprice queue starts with FD30A, FD18A, FD20A, route
    12, route 3, and route 18.
- Evidence artifacts:
  - `years/2026/checkpoints/latent-credit-delta-repricing-audit-2026-05-12.md`
  - `years/2026/checkpoints/latent-credit-delta-repricing-audit-2026-05-12.json`
  - `years/2026/checkpoints/latent-credit-delta-repricing-audit-2026-05-12-manifest.json`
- Validation:
  - `python years/2026/scripts/latent_credit_delta_repricing_audit.py` passed
    with status `proved_current_calendar_savings`.
  - `python -m py_compile years/2026/scripts/latent_credit_delta_repricing_audit.py`
    passed.
  - `pytest -q years/2026/tests/test_latent_credit_delta_repricing_audit.py`
    passed 2 tests in 0.05s.
  - `pytest -q years/2026/tests/test_latent_credit_delta_repricing_audit.py
    years/2026/tests/test_route_repeat_optimization_audit.py
    years/2026/tests/test_field_official_repeat_audit.py` passed 8 tests in
    0.07s.

## 2026-05-12 - Cluster-level repricing audit

- Objective:
  - Move beyond route-card-level latent/repeat warnings and identify connected
    route clusters where the current certified no-shuttle route-card universe
    can cover the same official segments with fewer loops and lower field
    effort.
- What changed:
  - Added `cluster_level_repricing_audit.py` to build a route graph from
    reconciled latent credit, declared official repeats, and shared/near
    trailhead proximity.
  - For each connected component, the audit solves an exact set-cover problem
    over the existing certified no-shuttle route cards. Candidate coverage uses
    each route card's claimed segments plus GPX-derived full extra coverage.
  - The audit explicitly does not generate new route cards; high-cost
    components without enough savings are candidate-universe gaps for future
    loop generation.
- Result:
  - The graph has 77 edges, 13 components, and 3 multi-route components.
  - All 3 multi-route components optimized exactly.
  - Existing certified loops can cover those component segments with 34 route
    cards instead of 40, saving 13.73 on-foot miles, 536 p75 minutes, and 604
    p90 minutes in an order-free cluster repricing frame.
  - Savings components:
    - C02 Freestone / Military / Hulls-style component: removes FD19C and
      FD22A, saving 7.52 on-foot miles and 188 p75 minutes.
    - C01 Cartwright / Polecat / Harlow-Avimor-style component: removes FD14A,
      FD14C, FD27A, and FD27C, saving 6.21 on-foot miles and 348 p75 minutes.
  - Promotion caveat: this is not a current-calendar repair. It would require a
    new calendar assignment and full field-packet recertification before route
    cards could be skipped.
- Evidence artifacts:
  - `years/2026/checkpoints/cluster-level-repricing-audit-2026-05-12.md`
  - `years/2026/checkpoints/cluster-level-repricing-audit-2026-05-12.json`
  - `years/2026/checkpoints/cluster-level-repricing-audit-2026-05-12-manifest.json`
- Validation:
  - `python years/2026/scripts/cluster_level_repricing_audit.py` passed with
    status `optimized_existing_loop_clusters`.
  - `python -m py_compile years/2026/scripts/cluster_level_repricing_audit.py`
    passed.
  - `pytest -q years/2026/tests/test_cluster_level_repricing_audit.py` passed
    2 tests in 0.05s.
  - `pytest -q years/2026/tests/test_cluster_level_repricing_audit.py
    years/2026/tests/test_latent_credit_delta_repricing_audit.py
    years/2026/tests/test_route_repeat_optimization_audit.py
    years/2026/tests/test_field_official_repeat_audit.py` passed 10 tests in
    0.09s.

## 2026-05-12 - Ownership reassignment optimization audit

- Objective:
  - Treat official segment credit ownership as an optimization variable instead
    of a fixed provenance label, while keeping physical route-card traversal
    separate from credit assignment.
- What changed:
  - Added `ownership_reassignment_optimization_audit.py` to build a physical
    traversal graph from route-repeat `actual_full_segment_ids` plus current
    certified route-card claims.
  - For each ownership component, the audit solves the existing-route-card set
    cover, then assigns each official segment to the earliest selected physical
    coverer.
  - The report separates order-free route removals, current-calendar
    skip-ready removals, and partial shrink rows that need regenerated route
    cards before any on-foot/p75 savings can be claimed.
- Result:
  - The ownership graph has 41 edges, 25 components, and 5 relevant ownership
    components.
  - All 5 relevant components optimized exactly.
  - The audit reassigns 33 official segments covering 15.27 official miles.
  - Order-free existing-loop repricing removes 6 route cards and saves 13.73
    on-foot miles, 536 p75 minutes, and 604 p90 minutes.
  - In the current calendar, only 2 of those removed cards are immediately
    skip-ready because the replacement owner route is already scheduled no
    later; proven current-calendar savings remain 4.39 on-foot miles, 147 p75
    minutes, and 166 p90 minutes.
  - Ten selected routes lose some current credit assignment but still need a
    regenerated replacement card before any partial-shrink mileage savings can
    be priced.
- Evidence artifacts:
  - `years/2026/checkpoints/ownership-reassignment-optimization-audit-2026-05-12.md`
  - `years/2026/checkpoints/ownership-reassignment-optimization-audit-2026-05-12.json`
  - `years/2026/checkpoints/ownership-reassignment-optimization-audit-2026-05-12-manifest.json`
- Validation:
  - `python years/2026/scripts/ownership_reassignment_optimization_audit.py`
    passed with status `ownership_reassignment_reduces_existing_loop_work`.
  - `python -m py_compile years/2026/scripts/ownership_reassignment_optimization_audit.py`
    passed.
  - `pytest -q years/2026/tests/test_ownership_reassignment_optimization_audit.py`
    passed 2 tests in 0.06s.
  - `pytest -q years/2026/tests/test_ownership_reassignment_optimization_audit.py
    years/2026/tests/test_cluster_level_repricing_audit.py
    years/2026/tests/test_latent_credit_delta_repricing_audit.py
    years/2026/tests/test_route_repeat_optimization_audit.py
    years/2026/tests/test_field_official_repeat_audit.py` passed 12 tests in
    0.10s.

## 2026-05-12 - Repeat productivity audit

- Objective:
  - Stop ranking route pain by raw non-credit miles when some repeat is useful
    future-credit work and some repeat is simply the current cost of returning
    to the parked car.
- What changed:
  - Added `repeat_productivity_audit.py` to classify official repeat/latent
    mileage into `productive_repeat`, `necessary_repeat`, and
    `dead_repeat_candidate`.
  - The audit uses wayfinding-cue repeat declarations for physical repeat
    mileage and ownership-reassignment output for productive future-credit
    evidence.
  - Dead-repeat classification requires alternate order/start/ownership
    pressure. High repeat mileage alone is not enough.
- Result:
  - The audit classified 45.33 repeat/latent miles across 50 routes.
  - Productive repeat/latent: 3.73 mi, including 2.21 mi declared repeat and
    1.53 mi latent productive coverage.
  - Necessary repeat: 27.99 mi.
  - Dead-repeat candidates: 13.60 mi across 22 routes.
  - The dead-repeat queue starts with route 3, FD18A, FD20A, route 12, route
    18, and FD27A.
  - High non-credit routes that should not lead this queue include FD24A and
    FD04A because their repeat includes productive future-credit ownership, and
    16A-1 because the current audits classify its repeat as necessary
    access/return with no proven alternate.
- Evidence artifacts:
  - `years/2026/checkpoints/repeat-productivity-audit-2026-05-12.md`
  - `years/2026/checkpoints/repeat-productivity-audit-2026-05-12.json`
  - `years/2026/checkpoints/repeat-productivity-audit-2026-05-12-manifest.json`
- Validation:
  - `python years/2026/scripts/repeat_productivity_audit.py` passed with
    status `dead_repeat_candidates_found`.
  - `python -m py_compile years/2026/scripts/repeat_productivity_audit.py`
    passed.
  - `pytest -q years/2026/tests/test_repeat_productivity_audit.py` passed 3
    tests in 0.05s.
  - `pytest -q years/2026/tests/test_repeat_productivity_audit.py
    years/2026/tests/test_ownership_reassignment_optimization_audit.py
    years/2026/tests/test_cluster_level_repricing_audit.py
    years/2026/tests/test_latent_credit_delta_repricing_audit.py
    years/2026/tests/test_route_repeat_optimization_audit.py
    years/2026/tests/test_field_official_repeat_audit.py` passed 15 tests in
    0.11s.

## 2026-05-12 - Simulated progress sweep audit

- Objective:
  - Use pre-challenge simulated completion to rank routes and field days by
    how much later work they collapse, rather than treating all early routes as
    equal once their own credit is counted.
- What changed:
  - Added `simulated_progress_sweep_audit.py` to simulate completing each
    current route card and each generated field day.
  - Simulated completion applies route-card claimed segments plus GPX-derived
    full official segment coverage from the route-repeat audit.
  - The audit reports total remaining-menu reduction, future route removals,
    future route shrinks, and separates priced full removals from unpriced
    partial shrink pressure.
- Result:
  - The sweep evaluated 50 route cards and 31 field days.
  - 9 sweeps remove at least one future route card.
  - 43 sweeps create at least one future partial-shrink candidate.
  - Top route-card priority by future collapse is FD04A: completing it early
    removes FD19C, shrinks 3 future routes, and saves 4.76 future on-foot
    miles / 109 p75 minutes beyond the route itself.
  - Top field-day priority is the 2026-06-24 FD04A day for the same reason.
  - FD30A and route 12 follow as the next strongest full-removal candidates;
    several large days create useful shrink pressure but no priced future
    removal until replacement cards are generated.
- Evidence artifacts:
  - `years/2026/checkpoints/simulated-progress-sweep-audit-2026-05-12.md`
  - `years/2026/checkpoints/simulated-progress-sweep-audit-2026-05-12.json`
  - `years/2026/checkpoints/simulated-progress-sweep-audit-2026-05-12-manifest.json`
- Validation:
  - `python years/2026/scripts/simulated_progress_sweep_audit.py` passed with
    status `simulated_progress_priority_found`.
  - `python -m py_compile years/2026/scripts/simulated_progress_sweep_audit.py`
    passed.
  - `pytest -q years/2026/tests/test_simulated_progress_sweep_audit.py` passed
    3 tests in 0.06s.
  - `pytest -q years/2026/tests/test_simulated_progress_sweep_audit.py
    years/2026/tests/test_repeat_productivity_audit.py
    years/2026/tests/test_ownership_reassignment_optimization_audit.py
    years/2026/tests/test_cluster_level_repricing_audit.py
    years/2026/tests/test_latent_credit_delta_repricing_audit.py
    years/2026/tests/test_route_repeat_optimization_audit.py
    years/2026/tests/test_field_official_repeat_audit.py` passed 18 tests in
    0.13s.

## 2026-05-12 - Common route template candidate audit

- Objective:
  - Convert source-backed common human route patterns into generator inputs so
    route experiments can start from normal loops instead of graph search only.
- What changed:
  - Added public-safe template seeds for Dry Creek/Shingle, Freestone/Shane's,
    Hulls/Crestline, Bogus, and Harlow/Avimor under
    `years/2026/inputs/open-data/common-route-templates-2026-05-12.json`.
  - Added `common_route_template_candidate_audit.py` to validate template
    segment IDs against the official 2026 foot segments, map templates to
    current route cards, and join repeat-productivity / simulated-progress
    pressure.
  - The generated candidate payloads are explicitly route-experiment seeds,
    not field-packet promotions.
- Result:
  - The audit generated 5 template candidates with 0 invalid official segment
    IDs.
  - 4 templates have captured public Strava route sources.
  - The Harlow/Avimor/Spring Valley template remains a cluster seed that needs
    public route-source capture before promotion.
  - Top generator experiment by current pressure is
    `freestone-shanes-three-bears-loop`, intersecting 4 current route cards,
    8.08 dead-repeat candidate miles, 4.76 priced future-collapse miles, and
    8.43 unpriced shrink official miles.
- Evidence artifacts:
  - `years/2026/checkpoints/common-route-template-candidates-2026-05-12.md`
  - `years/2026/checkpoints/common-route-template-candidates-2026-05-12.json`
  - `years/2026/checkpoints/common-route-template-candidates-2026-05-12-manifest.json`
- Validation:
  - `jq empty years/2026/inputs/open-data/common-route-templates-2026-05-12.json`
    passed.
  - `python years/2026/scripts/common_route_template_candidate_audit.py`
    passed with status `template_candidates_generated`.
  - `python -m py_compile years/2026/scripts/common_route_template_candidate_audit.py`
    passed.
  - `pytest -q years/2026/tests/test_common_route_template_candidate_audit.py`
    passed 3 tests in 0.05s.

## 2026-05-12 - Cluster route optimization audit

- Objective:
  - Add the second-layer cluster audit requested after common-route templates:
    archetype mismatch scoring, bundle-aware replacement candidates,
    already-paid access corridor accounting, and dominance checks.
- What changed:
  - Added `cluster_route_optimization_audit.py`, consuming the field packet,
    common-route template candidates, route-repeat audit, repeat-productivity
    audit, and simulated-progress sweep.
  - The audit ranks cluster archetype mismatch without failing routes, emits
    `cluster_bundle` replacement candidates, groups repeated access/return
    corridor families, and separates hard/current dominance from post-progress
    and lower-bound bundle dominance.
  - Bundle outputs distinguish `replaces_routes` from `touches_routes` and list
    `uncovered_current_segment_ids`, so a template overlap cannot masquerade as
    a complete replacement.
- Result:
  - Top archetype mismatch target is
    `freestone-shanes-three-bears-loop` with score 14.59.
  - The next mismatch targets are Bogus Simplot/Pioneer, Harlow/Avimor, Dry
    Creek/Shingle, and Hulls/Crestline.
  - 5 cluster-bundle candidates were generated. The Freestone bundle still
    needs additional loops because it touches route 3 and FD04A without
    covering all of their official segments.
  - 7 already-paid access/return corridor families were found, led by the
    FD09A/10B Dry Creek return corridor and the Freestone access corridor paid
    across FD19C, FD04A, route 3, and FD20A.
  - 48 dominance candidates were found: 45 are post-progress remove/shrink
    actions and 3 are lower-bound bundle dominance candidates. None of these
    are route-card deletions before validated completion or generated bundle
    repricing.
- Evidence artifacts:
  - `years/2026/checkpoints/cluster-route-optimization-audit-2026-05-12.md`
  - `years/2026/checkpoints/cluster-route-optimization-audit-2026-05-12.json`
  - `years/2026/checkpoints/cluster-route-optimization-audit-2026-05-12-manifest.json`
- Validation:
  - `python years/2026/scripts/cluster_route_optimization_audit.py` passed with
    status `cluster_optimization_targets_found`.
  - `python -m py_compile years/2026/scripts/cluster_route_optimization_audit.py`
    passed.
  - `pytest -q years/2026/tests/test_cluster_route_optimization_audit.py`
    passed 3 tests in 0.05s.

## 2026-05-12 - Gate refresh and certification chain

- Objective:
  - Re-run the generated field-packet and route-optimization gates from the
    same regenerated artifact set, then make the pass/fail boundary explicit.
- Result:
  - Field packet export passed and regenerated 150 GPX files.
  - Field latent credit audit, progress report, recertification report, field
    tool completion audit, field route walkthrough audit, and route-repeat hard
    failure audit all passed.
  - The regenerated packet reports `gpx_validation_passed: true`,
    `failed_gpx_count: 0`, `day_gpx_validation_passed: true`,
    `needs_route_card_audit_fix_loop_count: 0`, and
    `needs_route_card_promotion_loop_count: 0`.
  - The newer route-optimization audits all ran successfully. Their
    `targets_found` statuses remain investigation queues, not field-packet
    failures.
  - Candidate promotion gaps such as Harlow/Avimor public-source capture,
    cluster-bundle route geometry, p75/p90, access proof, cue sheets, and
    recertification remain intentionally gated before any replacement route is
    promoted.
- Evidence artifacts:
  - `years/2026/checkpoints/gate-status-2026-05-12.md`
  - `years/2026/checkpoints/gate-status-2026-05-12.json`
- Validation:
  - `python years/2026/scripts/export_mobile_field_packet.py` passed.
  - `python years/2026/scripts/field_latent_credit_audit.py` passed.
  - `python years/2026/scripts/field_progress_report.py` passed.
  - `python years/2026/scripts/field_recertification_report.py` passed.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed.
  - `python years/2026/scripts/route_repeat_optimization_audit.py` passed with
    status `passed`.
  - `pytest -q years/2026/tests/test_route_repeat_optimization_audit.py
    years/2026/tests/test_field_official_repeat_audit.py
    years/2026/tests/test_latent_credit_delta_repricing_audit.py
    years/2026/tests/test_cluster_level_repricing_audit.py
    years/2026/tests/test_ownership_reassignment_optimization_audit.py
    years/2026/tests/test_repeat_productivity_audit.py
    years/2026/tests/test_simulated_progress_sweep_audit.py
    years/2026/tests/test_common_route_template_candidate_audit.py
    years/2026/tests/test_cluster_route_optimization_audit.py
    years/2026/tests/test_field_route_walkthrough_audit.py` passed 35 tests in
    0.22s.

## 2026-05-12 - Repeat/latent gate hardening

- Objective:
  - Convert the repeat/latent optimization audits from diagnostic-only output
    into explicit readiness gates where appropriate, and keep skip-ready route
    removals out of the active menu until they are executable route-card
    changes.
- Result:
  - Field tool completion now includes official-repeat and route-repeat hard
    gates, while latent repricing, ownership reassignment, and simulated
    progress remain advisory until they generate planned menu changes.
  - Repeat productivity now reports `dead_repeat_actual_route_miles` separately
    from official-segment pressure. The prior confusing case, `115-1: 3`, is
    now 2.26 actual route miles vs 6.45 official-pressure miles.
  - A new current-calendar skip-ready promotion audit blocks active deletion of
    `FD14C` and `FD22A` because their predecessor routes physically cover the
    segments but do not yet claim/cue them as credit, and the field-day layer
    still references the later cards.
  - No active menu deletion was promoted in this pass.
- Evidence artifacts:
  - `years/2026/checkpoints/current-calendar-skip-ready-promotion-audit-2026-05-12.md`
  - `years/2026/checkpoints/repeat-productivity-audit-2026-05-12.md`
  - `years/2026/checkpoints/field-tool-completion-audit-2026-05-06.md`
  - `years/2026/checkpoints/gate-status-2026-05-12.md`
- Validation:
  - `python years/2026/scripts/export_mobile_field_packet.py` passed.
  - `python years/2026/scripts/field_latent_credit_audit.py` passed.
  - `python years/2026/scripts/field_progress_report.py` passed.
  - `python years/2026/scripts/field_recertification_report.py` passed.
  - `python years/2026/scripts/field_official_repeat_audit.py` passed.
  - `python years/2026/scripts/route_repeat_optimization_audit.py` passed.
  - `python years/2026/scripts/latent_credit_delta_repricing_audit.py` passed.
  - `python years/2026/scripts/cluster_level_repricing_audit.py` passed.
  - `python years/2026/scripts/ownership_reassignment_optimization_audit.py` passed.
  - `python years/2026/scripts/repeat_productivity_audit.py` passed.
  - `python years/2026/scripts/simulated_progress_sweep_audit.py` passed.
  - `python years/2026/scripts/common_route_template_candidate_audit.py` passed.
  - `python years/2026/scripts/cluster_route_optimization_audit.py` passed.
  - `python years/2026/scripts/current_calendar_skip_ready_promotion_audit.py`
    wrote artifacts and exited with expected promotion-gate status
    `blocked_needs_route_card_claim_promotion`.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 15/15.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 50/50.
  - `pytest -q years/2026/tests/test_route_repeat_optimization_audit.py
    years/2026/tests/test_field_official_repeat_audit.py
    years/2026/tests/test_latent_credit_delta_repricing_audit.py
    years/2026/tests/test_cluster_level_repricing_audit.py
    years/2026/tests/test_ownership_reassignment_optimization_audit.py
    years/2026/tests/test_repeat_productivity_audit.py
    years/2026/tests/test_simulated_progress_sweep_audit.py
    years/2026/tests/test_common_route_template_candidate_audit.py
    years/2026/tests/test_cluster_route_optimization_audit.py
    years/2026/tests/test_field_route_walkthrough_audit.py
    years/2026/tests/test_field_tool_completion_audit.py
    years/2026/tests/test_current_calendar_skip_ready_promotion_audit.py`
    passed 55 tests in 0.26s.
  - `pytest -q` passed 466 tests in 120.79s.

## 2026-05-12 - Route-card credit ownership promotion

- Objective:
  - Convert the two current-calendar skip-ready removals from diagnostic audit
    findings into executable, credit-owning route-card changes.
- Result:
  - Promoted Quick Draw segment `1610` from removed `FD14C` into `FD14B`.
  - Promoted Highlands segments `1576` and `1577` from removed `FD22A` into
    route `12`.
  - Active field packet now has `48` route cards instead of `50`, preserves
    `251 / 251` official segments, and removes `4.39` on-foot miles plus `147`
    p75 minutes from the current calendar.
  - Field-day layer is `field_day_certified` with `48` certified route-card
    loops, `0` promotion gaps, `0` audit-fix gaps, and `2` skipped source
    loops.
- Evidence artifacts:
  - `years/2026/checkpoints/route-card-credit-promotion-2026-05-12.md`
  - `years/2026/checkpoints/route-card-credit-promotion-2026-05-12.json`
  - `years/2026/checkpoints/field-day-loop-promotion-2026-05-11.md`
  - `years/2026/checkpoints/human-executable-field-day-layer-2026-05-10.md`
  - `years/2026/checkpoints/field-tool-completion-audit-2026-05-06.md`
- Validation:
  - `pytest -q years/2026/tests/test_multi_start_field_menu_replacements.py
    years/2026/tests/test_promote_field_day_loops.py
    years/2026/tests/test_export_field_day_layer.py
    years/2026/tests/test_current_calendar_skip_ready_promotion_audit.py
    years/2026/tests/test_field_tool_completion_audit.py
    years/2026/tests/test_repeat_productivity_audit.py` passed 49 tests.
  - JSON validation passed for the segment promotions source, field packet data,
    field packet manifest, field-day promotion report, and field-day layer.
  - `python years/2026/scripts/field_progress_report.py` passed.
  - `python years/2026/scripts/field_recertification_report.py` passed.
  - `python years/2026/scripts/field_latent_credit_audit.py` passed with `48`
    routes and `0` routes needing repair.
  - `python years/2026/scripts/field_official_repeat_audit.py` passed.
  - `python years/2026/scripts/route_repeat_optimization_audit.py` passed.
  - `python years/2026/scripts/current_calendar_skip_ready_promotion_audit.py`
    passed with `no_skip_ready_removals` after promotion.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 15/15.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 48/48.
  - `python years/2026/scripts/latent_credit_delta_repricing_audit.py` passed.
  - `python years/2026/scripts/ownership_reassignment_optimization_audit.py` passed.
  - `python years/2026/scripts/repeat_productivity_audit.py` passed.
  - `python years/2026/scripts/simulated_progress_sweep_audit.py` passed.
  - `python years/2026/scripts/cluster_level_repricing_audit.py` passed.
  - `python years/2026/scripts/cluster_route_optimization_audit.py` passed.
  - `python years/2026/scripts/common_route_template_candidate_audit.py` passed.
  - `pytest -q` passed 476 tests in 118.91s.

## 2026-05-12 - Same-car corridor fusion probes

- Objective:
  - Turn the repeated paid access/return corridor findings into concrete
    same-parked-car fusion probes before promoting any route-card changes.
- Result:
  - Added `years/2026/scripts/same_car_corridor_fusion_experiment.py`.
  - Generated
    `years/2026/checkpoints/same-car-corridor-fusion-experiment-2026-05-12.md`
    and JSON/manifest companions.
  - Evaluated four repeated-corridor probes: Dry Creek `FD09A + 10B`,
    Freestone `FD19C + FD04A + 3 + FD20A`, Cartwright
    `FD14A + FD14B + FD18A`, and Avimor `FD27A + FD27B + FD27C`.
  - The experiment identifies two existing-route-card promotion candidates:
    `FD14B` can absorb `FD14A` after Doe Ridge claim/cue promotion, saving
    `1.08` on-foot miles / `58` p75; `FD27B` can absorb `FD27A` after
    Spring Creek 1 claim/cue promotion, saving `1.49` on-foot miles / `104`
    p75.
  - Dry Creek and Freestone remain paper-only lower-bound fusion probes until
    continuous GPX, DEM timing, cue rewrite, coverage, ascent-direction, and
    recertification gates exist.
  - No active field packet route cards were promoted or removed.
- Evidence artifacts:
  - `years/2026/checkpoints/same-car-corridor-fusion-experiment-2026-05-12.md`
  - `years/2026/checkpoints/same-car-corridor-fusion-experiment-2026-05-12.json`
  - `years/2026/checkpoints/same-car-corridor-fusion-experiment-2026-05-12-manifest.json`
- Validation:
  - `pytest -q years/2026/tests/test_same_car_corridor_fusion_experiment.py`
    passed 3 tests.
  - `python years/2026/scripts/same_car_corridor_fusion_experiment.py` wrote
    the JSON, Markdown, and manifest artifacts.
  - `pytest -q years/2026/tests/test_same_car_corridor_fusion_experiment.py
    years/2026/tests/test_calendar_reorder_for_latent_credit_experiment.py
    years/2026/tests/test_latent_credit_delta_repricing_audit.py
    years/2026/tests/test_cluster_route_optimization_audit.py` passed 11
    tests.
  - `python -m py_compile
    years/2026/scripts/same_car_corridor_fusion_experiment.py` passed.
  - `python -m json.tool` passed for the new JSON and manifest.
  - `pytest -q` passed 483 tests in 114.41s.

## 2026-05-12 - Freestone cluster route-generation experiment

- Objective:
  - Turn the top archetype mismatch, Freestone/Shane's/Three Bears, into a
    real GPX route-generation experiment before considering any route-card
    replacement.
- Result:
  - Added `years/2026/scripts/freestone_cluster_route_generation_experiment.py`.
  - Generated
    `years/2026/checkpoints/freestone-route-generation-experiment-2026-05-12.md`
    plus JSON, manifest, and two GPX candidates.
  - The template-sequence candidate preserves the common-route order and is a
    graph-validated continuous GPX, but prices at `32.52` on-foot miles /
    `673` scaled p75 with repeat-credit review still needed.
  - The nearest-segment candidate is the shorter GPX at `31.38` on-foot miles /
    `649` scaled p75, but it uses a `0.01` mile direct-gap fallback and is less
    human-normal.
  - Both candidates cover the 22 template segment IDs but leave `19` current
    route-card segment IDs uncovered, so neither is a direct replacement for
    `FD19C`, `FD20A`, `FD04A`, and route `3`.
  - No active field packet route cards were promoted or removed.
- Evidence artifacts:
  - `years/2026/checkpoints/freestone-route-generation-experiment-2026-05-12.md`
  - `years/2026/checkpoints/freestone-route-generation-experiment-2026-05-12.json`
  - `years/2026/checkpoints/freestone-route-generation-experiment-2026-05-12-manifest.json`
  - `years/2026/checkpoints/freestone-route-generation-experiment-2026-05-12/template-sequence-greedy.gpx`
  - `years/2026/checkpoints/freestone-route-generation-experiment-2026-05-12/nearest-segment-greedy.gpx`
- Validation:
  - `pytest -q years/2026/tests/test_freestone_cluster_route_generation_experiment.py`
    passed 3 tests.
  - `python years/2026/scripts/freestone_cluster_route_generation_experiment.py`
    wrote the JSON, Markdown, manifest, and GPX artifacts.
  - `python -m py_compile
    years/2026/scripts/freestone_cluster_route_generation_experiment.py` passed.
  - `python -m json.tool` passed for the new JSON and manifest.
  - `pytest -q` passed 486 tests in 118.79s.

## 2026-05-12 - Freestone/Military candidate bundle experiment

- Objective:
  - Generate smaller Freestone/Military candidate bundles instead of trying to
    solve the whole `39.54` mile component as one elegant loop.
- Result:
  - Added
    `years/2026/scripts/freestone_military_candidate_bundle_experiment.py`.
  - Generated
    `years/2026/checkpoints/freestone-military-candidate-bundles-2026-05-12.md`
    plus JSON, manifest, and GPX candidates.
  - Tested five bundles across the requested F1/F2/F3 shapes:
    upper Freestone loop with FD20A shrink, upper loop with Mountain Cove
    warm-up claimed, one upper loop for FD19C/FD04A/FD20A, Military core
    preserved with only FD19C/FD04A merged, and Curlew/Fat Tire/Freestone
    safety routing.
  - `0` bundles are promotion candidates. The best delta was still worse than
    current: `F1-upper-single-loop-all-three-current-cards` at `+0.09`
    on-foot miles / `+2` p75.
  - F3 preserved Curlew ascent direction in generation, but combining
    Curlew/Fat Tire with Freestone work priced at `+6.16` on-foot miles /
    `+135` p75 against its current scope.
  - No active field packet route cards were promoted or removed.
- Evidence artifacts:
  - `years/2026/checkpoints/freestone-military-candidate-bundles-2026-05-12.md`
  - `years/2026/checkpoints/freestone-military-candidate-bundles-2026-05-12.json`
  - `years/2026/checkpoints/freestone-military-candidate-bundles-2026-05-12-manifest.json`
  - `years/2026/checkpoints/freestone-military-candidate-bundles-2026-05-12/`
- Validation:
  - `pytest -q years/2026/tests/test_freestone_military_candidate_bundle_experiment.py
    years/2026/tests/test_freestone_cluster_route_generation_experiment.py`
    passed 7 tests.
  - `python years/2026/scripts/freestone_military_candidate_bundle_experiment.py`
    wrote the JSON, Markdown, manifest, and GPX artifacts.
  - `pytest -q years/2026/tests/test_freestone_military_candidate_bundle_experiment.py
    years/2026/tests/test_freestone_cluster_route_generation_experiment.py
    years/2026/tests/test_same_car_corridor_fusion_experiment.py
    years/2026/tests/test_cluster_route_optimization_audit.py
    years/2026/tests/test_common_route_template_candidate_audit.py` passed 16
    tests.
  - `python -m py_compile
    years/2026/scripts/freestone_cluster_route_generation_experiment.py
    years/2026/scripts/freestone_military_candidate_bundle_experiment.py` passed.
  - `python -m json.tool` passed for the Freestone route-generation JSON and
    manifest and the Freestone/Military bundle JSON and manifest.
  - `pytest -q` passed 490 tests in 131.11s.

## 2026-05-12 - Template route candidate builder

- Objective:
  - Turn the Harlow/Avimor, Bogus, Hulls/Crestline, and Dry/Shingle common
    route templates into advisory GPX probes with source labels, savings math,
    route impacts, latent/repeat blockers, and promotion gates.
- Result:
  - Added `years/2026/scripts/template_route_candidate_builder.py`.
  - Updated `years/2026/inputs/open-data/common-route-templates-2026-05-12.json`
    with public Harlow/Avimor source labels and regenerated the common-route
    template audit; the old `cluster_seed_needs_public_source` status is gone
    (`5/5` templates now have public route sources, `0` source-gated seeds).
  - Generated
    `years/2026/checkpoints/template-route-candidates-2026-05-12.md` plus
    JSON, manifest, and GPX candidates.
  - Captured public Harlow/Avimor source labels from Avimor, Trailforks, and
    MTB Project, but kept Harlow west / Hidden Springs ungenerated because the
    public access evidence is stale or conflicted.
  - H1 Avimor-native Harlow/Spring is the largest paper candidate at `-24.63`
    on-foot miles / `-718` scaled p75, but it is blocked by Avimor parking
    confidence, hidden self-repeat, direct gaps, cue work, and recertification.
  - B3 same-day Simplot/Pioneer has the largest Bogus day-pair paper delta at
    `-12.18` on-foot miles / `-399` scaled p75, but it is blocked by transfer,
    closure/date, repeat, direct-gap, cue, and recertification gates.
  - C1 Hulls/Kestrel/Crestline is a smaller paper improvement at `-2.68`
    on-foot miles / `-64` scaled p75, but remains Lower-Hulls date-gated with
    repeat/latent ownership review required.
  - D1 Dry/Shingle/Sweet Connie is worse than the current cards at `+2.28`
    on-foot miles / `+49` scaled p75, so it remains deferred unless field
    feedback changes the pressure.
  - No active field packet route cards were promoted or removed.
- Evidence artifacts:
  - `years/2026/checkpoints/template-route-candidates-2026-05-12.md`
  - `years/2026/checkpoints/template-route-candidates-2026-05-12.json`
  - `years/2026/checkpoints/template-route-candidates-2026-05-12-manifest.json`
  - `years/2026/checkpoints/template-route-candidates-2026-05-12/`
- Validation:
  - `python -m py_compile years/2026/scripts/template_route_candidate_builder.py`
    passed.
  - `pytest -q years/2026/tests/test_template_route_candidate_builder.py`
    passed 3 tests.
  - `python -m json.tool
    years/2026/inputs/open-data/common-route-templates-2026-05-12.json`
    passed.
  - `python years/2026/scripts/common_route_template_candidate_audit.py`
    regenerated the common-route template JSON, Markdown, and manifest.
  - `python years/2026/scripts/template_route_candidate_builder.py` wrote the
    JSON, Markdown, manifest, and GPX artifacts.
  - `python -m json.tool` passed for the new JSON and manifest.
  - `pytest -q
    years/2026/tests/test_template_route_candidate_builder.py
    years/2026/tests/test_common_route_template_candidate_audit.py
    years/2026/tests/test_freestone_military_candidate_bundle_experiment.py
    years/2026/tests/test_freestone_cluster_route_generation_experiment.py
    years/2026/tests/test_same_car_corridor_fusion_experiment.py
    years/2026/tests/test_calendar_reorder_for_latent_credit_experiment.py
    years/2026/tests/test_cluster_route_optimization_audit.py
    years/2026/tests/test_latent_credit_delta_repricing_audit.py` passed 24
    tests.
  - `pytest -q` passed 493 tests in 114.02s.

## 2026-05-12 - Harlow / Avimor H1 gate repair sprint

- Objective:
  - Treat `H1-avimor-native-harlow-spring-loop` as the top Harlow/Avimor
    optimization target and repair hard gates without promoting active route
    cards.
- Result:
  - Added `years/2026/scripts/harlow_h1_gate_repair_audit.py` with a dated
    checkpoint artifact, repaired GPX, direct-gap repair review, H1-specific
    route-repeat audit, explicit repeat conversion, field cue sheet, DEM
    p75/p90 repricing, parking-source sync status, and hypothetical 251/251
    coverage simulation.
  - Repaired H1 direct gaps by allowing only the target official segment for
    access snapping while still avoiding other unvisited official segments:
    direct-gap fallback dropped from `0.43` mi to `0.0` mi.
  - Converted the hidden Twisted Spring self-repeat found by the route-repeat
    audit into explicit priced repeat: H1 now declares `0.61` mi official
    repeat across `1626`, `1661`, `1687`, `1688`, `1689`, and `1704`; hidden
    self-repeat, latent credit, and unpriced repeat ids are all empty.
  - Recomputed H1 at `9.64` on-foot miles / `289` p75 / `324` p90, versus the
    current Harlow/Avimor cluster at `34.0` mi / `991` p75 / `1117` p90. This
    is still a candidate, not a promotion.
  - Fixed the forced-anchor parking source join in
    `years/2026/scripts/promote_field_day_loops.py` so trailhead-suffixed
    forced-anchor candidate ids carry the accepted Avimor parking confidence
    into preserved certified route-card cues.
  - Regenerated the promoted map source, public examples, and mobile field
    packet. FD27A/FD27B/FD27C now show Avimor Spring Valley Creek parking
    confidence `osm_amenity_parking_fee_no_capacity_36_source_checked` from
    `osm_overpass_amenity_parking_2026_05_06_plus_alltrails_spring_valley_creek`.
  - H1 remains gated by public-safe cueable access review, route-card
    promotion, and field-packet recertification. H2 remains only the fallback
    if H1 cannot clear those gates.
- Evidence artifacts:
  - `years/2026/checkpoints/harlow-h1-gate-repair-audit-2026-05-12.md`
  - `years/2026/checkpoints/harlow-h1-gate-repair-audit-2026-05-12.json`
  - `years/2026/checkpoints/harlow-h1-gate-repair-audit-2026-05-12-manifest.json`
  - `years/2026/checkpoints/harlow-h1-gate-repair-audit-2026-05-12/`
- Validation:
  - `python -m py_compile years/2026/scripts/harlow_h1_gate_repair_audit.py`
    passed.
  - `pytest -q years/2026/tests/test_harlow_h1_gate_repair_audit.py` passed
    5 tests.
  - `pytest -q years/2026/tests/test_promote_field_day_loops.py
    years/2026/tests/test_harlow_h1_gate_repair_audit.py` passed 15 tests.
  - `python years/2026/scripts/promote_field_day_loops.py` regenerated the
    promoted source and reported `covered_segment_count: 251`,
    `track_validation_passed: true`, and `source_gap_warning_count: 0`.
  - `python years/2026/scripts/export_example_map.py` regenerated public
    sanitized map/menu artifacts.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated
    `docs/field-packet/` and wrote 144 GPX files.
  - `python years/2026/scripts/harlow_h1_gate_repair_audit.py` wrote the H1
    JSON, Markdown, manifest, and repaired GPX artifacts.
  - `python -m json.tool` passed for the H1 JSON, H1 manifest, and
    `docs/field-packet/field-tool-data.json`.
  - `python years/2026/scripts/field_latent_credit_audit.py` passed.
  - `python years/2026/scripts/field_progress_report.py` passed with
    `remaining_coverage_preserved: true`.
  - `python years/2026/scripts/field_recertification_report.py` passed.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 15/15.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 48/48.
  - `python years/2026/scripts/route_repeat_optimization_audit.py` passed with
    zero hidden self-repeat, latent-credit, and unpriced-repeat failures.
  - `python years/2026/scripts/field_official_repeat_audit.py` passed.
  - `python years/2026/scripts/latent_credit_delta_repricing_audit.py` ran and
    remained advisory with `0` current-calendar savings.
  - `python years/2026/scripts/ownership_reassignment_optimization_audit.py`
    ran and remained advisory with `0` current-calendar skip-ready savings.
  - `python years/2026/scripts/simulated_progress_sweep_audit.py` ran and kept
    FD04A as the top simulated future-collapse route.
  - `pytest -q` passed 500 tests in 112.97s.

## 2026-05-12 - Harlow / Avimor H1 access/cue review

- Objective:
  - Treat `needs_public_safe_cueable_access_review` as a true promotion blocker
    for H1 and verify whether the repaired OSM/direct-gap connectors can be
    expressed as public, legal, field-readable cues.
- Result:
  - Added `years/2026/scripts/harlow_h1_access_cue_review.py`, tests, and a
    dated checkpoint packet.
  - Queried the public Avimor ArcGIS Hike/Bike trail layer and checked 181
    trail features, including 162 open/visible Hike/Bike features.
  - Resolved all opaque H1 OSM connector ids to named cueable field features:
    McLeod Way Greenbelt / Twisted Spring Trail near the start, Whistling Pig
    for the Spring Creek connector, and Burnt Car Draw / Cartwright Road / The
    Wall for the Harlow approach.
  - Promotion readiness now shows the access/cue gate clear for H1, with only
    day-of seasonal/condition checks remaining for seasonal Avimor trail
    features. This does not promote H1.
  - Added the explicit H1 replacement segment-set diff requested for promotion
    review: claimed ids equal the old FD27A/FD30A/FD27B/FD27C/FD24A union,
    with no missing ids and no extra ids; old/new cost remains `34.0`/`9.64`
    on-foot miles and `991`/`289` p75.
- Evidence artifacts:
  - `years/2026/checkpoints/harlow-h1-access-cue-review-2026-05-12.md`
  - `years/2026/checkpoints/harlow-h1-access-cue-review-2026-05-12.json`
  - `years/2026/checkpoints/harlow-h1-access-cue-review-2026-05-12-manifest.json`
- Validation:
  - `python -m py_compile years/2026/scripts/harlow_h1_access_cue_review.py`
    passed.
  - `pytest -q years/2026/tests/test_harlow_h1_access_cue_review.py` passed 3
    tests.
  - `python years/2026/scripts/harlow_h1_access_cue_review.py` wrote the JSON,
    Markdown, and manifest artifacts and reported
    `access_gate_clear_keep_unpromoted`.
  - `python -m json.tool` passed for the new JSON and manifest.
  - `pytest -q` passed 503 tests in 115.51s.

## 2026-05-12 - Harlow / Avimor H1 controlled promotion

- Objective:
  - Run the controlled active-packet promotion trial for H1 after the
    access/cue review cleared, replacing FD27A, FD27B, FD27C, FD24A, and
    FD30A only if the active source, field-day layer, public packet, repeat
    audits, cue quality, schedule placement, and recertification gates all
    pass.
- Result:
  - Added `years/2026/scripts/promote_harlow_h1_route_card.py` and
    `years/2026/scripts/harlow_h1_promotion_assertions.py`.
  - Promoted H1 into the canonical private route-card source and regenerated
    the field packet and public sanitized map/menu artifacts.
  - Active route-card count changed from 48 to 44: the five old
    Harlow/Avimor cards were removed and one certified H1 route card was
    added.
  - H1 is assigned to the weekend July 4 field day with `289` p75 and `324`
    p90 against a `360` minute p90 bound; the former June 21 and July 12
    Harlow/Avimor days are now reusable empty field days.
  - H1 claimed segment ids exactly match the removed-card union, with no
    missing or extra official segment ids. The promoted packet still accounts
    for 251/251 official foot segments.
  - Modeled Harlow/Avimor cluster cost changed from `34.00` miles / `991`
    p75 / `1117` p90 to `9.64` miles / `289` p75 / `324` p90, saving `24.36`
    miles / `702` p75 / `793` p90.
  - Runner-facing H1 cues use named field features such as McLeod Way
    Greenbelt, Twisted Spring Trail #8, Whistling Pig #3, Burnt Car Draw #10,
    Cartwright Road #20, and The Wall #29 instead of opaque OSM connector ids.
- Evidence artifacts:
  - `years/2026/checkpoints/harlow-h1-route-card-promotion-2026-05-12.md`
  - `years/2026/checkpoints/harlow-h1-route-card-promotion-2026-05-12.json`
  - `years/2026/checkpoints/harlow-h1-route-card-promotion-2026-05-12-manifest.json`
  - `years/2026/checkpoints/harlow-h1-promotion-assertions-2026-05-12.md`
  - `years/2026/checkpoints/harlow-h1-promotion-assertions-2026-05-12.json`
  - `years/2026/checkpoints/harlow-h1-promotion-assertions-2026-05-12-manifest.json`
- Validation:
  - `python -m py_compile years/2026/scripts/promote_harlow_h1_route_card.py
    years/2026/scripts/harlow_h1_promotion_assertions.py
    years/2026/scripts/export_field_day_layer.py` passed.
  - `pytest -q years/2026/tests/test_export_field_day_layer.py` passed 10
    tests.
  - `python years/2026/scripts/promote_harlow_h1_route_card.py` wrote the H1
    promotion JSON, Markdown, and manifest and reported `new_route_card_count:
    44`, `saved_on_foot_miles: 24.36`, `saved_p75_minutes: 702`, and
    `saved_p90_minutes: 793`.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated
    `docs/field-packet/` and wrote 132 GPX files.
  - `python years/2026/scripts/export_field_day_layer.py` passed with
    `loop_count: 44`, `covered_segment_count: 251`,
    `missing_segment_count: 0`, and `schedule_p90_violation_day_count: 0`.
  - `python years/2026/scripts/harlow_h1_promotion_assertions.py` passed 18/18
    explicit H1 promotion assertions.
  - `python years/2026/scripts/field_latent_credit_audit.py` passed.
  - `python years/2026/scripts/field_official_repeat_audit.py` passed.
  - `python years/2026/scripts/route_repeat_optimization_audit.py` passed with
    zero hidden self-repeat, latent-credit, and unpriced-repeat failures.
  - `python years/2026/scripts/field_progress_report.py` passed with
    `remaining_coverage_preserved: true`.
  - `python years/2026/scripts/field_recertification_report.py` passed.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 44/44.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 15/15.
  - `python -m json.tool` passed for the H1 promotion JSON, H1 promotion
    manifest, H1 assertion JSON, and H1 assertion manifest.
  - `pytest -q` passed 504 tests in 113.72s.

## 2026-05-12 - Post-H1 control-plane and optimization refresh

- Objective:
  - Refresh stale control-plane artifacts after H1 promotion and rerun the
    advisory optimization queue so removed Harlow/Avimor microcards do not
    remain active investigation targets.
- Result:
  - Regenerated `gate-status-2026-05-12` to the current 44-card active packet
    state and explicitly superseded the prior 48-card gate-status note.
  - Added `harlow-h1-active-packet-certification-2026-05-12` as the final
    active-packet certification checkpoint. The original H1 route-card
    promotion checkpoint remains the source-mutation record and now links to
    the final certification checkpoint.
  - Kept the freed `2026-06-21` and `2026-07-12` days visible as
    `reusable_empty_field_day` reserve/buffer days rather than removing them or
    treating them as tasks.
  - Reran the post-H1 optimization queue. Harlow/Avimor H1/H2/H3 template
    probes are now marked superseded because their old replaced route labels
    are absent from the active packet. Current promising template candidates
    are Bogus B1/B2/B3 and Hulls/Kestrel/Crestline C1.
  - Fixed `template_route_candidate_builder.py` so stale replaced route labels
    are reported as superseded/absent instead of crashing on `FD27A`.
- Evidence artifacts:
  - `years/2026/checkpoints/gate-status-2026-05-12.md`
  - `years/2026/checkpoints/gate-status-2026-05-12.json`
  - `years/2026/checkpoints/harlow-h1-active-packet-certification-2026-05-12.md`
  - `years/2026/checkpoints/harlow-h1-active-packet-certification-2026-05-12.json`
  - `years/2026/checkpoints/harlow-h1-active-packet-certification-2026-05-12-manifest.json`
  - `years/2026/checkpoints/template-route-candidates-2026-05-12.md`
  - `years/2026/checkpoints/template-route-candidates-2026-05-12.json`
- Validation:
  - `python years/2026/scripts/latent_credit_delta_repricing_audit.py` passed
    and refreshed the audit with 44 routes, 2 pairwise full-removal
    relationships, and 0 current-calendar savings.
  - `python years/2026/scripts/ownership_reassignment_optimization_audit.py`
    passed and refreshed the audit with 5.84 order-free on-foot miles and 167
    p75 minutes of non-current-calendar savings.
  - `python years/2026/scripts/repeat_productivity_audit.py` passed and
    refreshed the audit with 17 dead-repeat candidate routes and 6.56 actual
    route miles of dead-repeat pressure.
  - `python years/2026/scripts/simulated_progress_sweep_audit.py` passed and
    kept `FD04A` as the top future-collapse route.
  - `python years/2026/scripts/calendar_reorder_for_latent_credit_experiment.py`
    passed and found 2 supported reorder candidates worth 5.84 miles / 167 p75
    non-additively.
  - `python years/2026/scripts/same_car_corridor_fusion_experiment.py` passed
    and kept `cartwright-fd14a-fd14b-fd18a` as the one promotion candidate.
  - `python years/2026/scripts/cluster_route_optimization_audit.py` passed and
    refreshed dominance/repeated-access counts after H1.
  - `python years/2026/scripts/template_route_candidate_builder.py` passed
    after the stale-label fix and excluded Harlow/Avimor as superseded by H1.
  - `python -m py_compile years/2026/scripts/template_route_candidate_builder.py`
    passed.
  - `pytest -q years/2026/tests/test_template_route_candidate_builder.py`
    passed 5 tests.
  - `pytest -q` passed 506 tests in 123.80s.

## 2026-05-13 - Bogus B1/B2 gate-repair audit tightening

- Objective:
  - Run a scoped Bogus B1/B2 gate-repair audit only, not promotion, and tighten
    the audit so named cue substitutions do not clear the gate unless the
    candidate GPX is actually rebuilt without direct-gap fallback geometry.
- Result:
  - Updated `bogus_b1_b2_gate_repair_audit.py` to price source route-card GPX
    cue legs, keep direct-gap fallback as a hard stop, separate hard gate
    failures from later recertification requirements, and report repeat,
    connector, road-estimate, cue, ownership, signage, and closure/date gates.
  - Generated `bogus-b1-b2-gate-repair-audit-2026-05-13` with
    `active_packet_mutated: false`.
  - B1 remains blocked: 2 direct gaps can be priced from source route GPX cues,
    but the candidate GPX is not rebuilt without direct-gap fallback. Real
    GPX-priced cost is `14.17` miles / `484` p75 / `544` p90.
  - B2 remains blocked: 1 direct gap can be priced from source route GPX, but
    the candidate GPX is not rebuilt without direct-gap fallback. Real
    GPX-priced cost is `12.71` miles / `392` p75 / `441` p90.
  - Current Bogus cards stay active; B3 remains deferred.
- Evidence artifacts:
  - `years/2026/checkpoints/bogus-b1-b2-gate-repair-audit-2026-05-13.md`
  - `years/2026/checkpoints/bogus-b1-b2-gate-repair-audit-2026-05-13.json`
  - `years/2026/checkpoints/bogus-b1-b2-gate-repair-audit-2026-05-13-manifest.json`
- Validation:
  - First run of `python years/2026/scripts/bogus_b1_b2_gate_repair_audit.py`
    wrote JSON/Markdown but failed manifest generation because a directory was
    passed as a manifest input; fixed by recording the packet directory in
    manifest metadata.
  - `python years/2026/scripts/bogus_b1_b2_gate_repair_audit.py` passed and
    wrote the May 13 JSON, Markdown, and manifest.
  - `pytest -q years/2026/tests/test_bogus_b1_b2_gate_repair_audit.py`
    passed 5 tests in 0.08s.
  - `pytest -q years/2026/tests/test_template_route_candidate_builder.py`
    passed 5 tests in 0.05s.
  - `python -m py_compile years/2026/scripts/bogus_b1_b2_gate_repair_audit.py`
    passed.

## 2026-05-13 - Pairwise full-removal latent-credit audit

- Objective:
  - Recheck current field-day layer pairwise full-removal opportunities by
    moving each source card/day before the owner card/day, removing the owner
    card only when all claimed segments are covered, and recomputing p75/p90,
    field-day count, route count, coverage, date legality, closure assumptions,
    and day-level GPX continuity.
- Result:
  - Current packet still has 44 certified route-card loops across 31 field
    days, 251/251 official segments covered, 0 route-card promotion gaps, 0
    route-card audit-fix gaps, 0 schedule p90 violations, and day-level GPX
    validation passed.
  - Fresh latent-credit audit found 38 reconciled latent official segments and
    0 unreconciled latent-credit repairs.
  - Pairwise full-removal candidates remain exactly 2:
    `FD04A -> FD19C` and `FD14B -> FD14A`.
  - `FD04A -> FD19C` is supported as an experiment only: moving `FD04A` from
    2026-06-24 to 2026-06-18 and moving the remaining FD19A/FD19B owner day to
    2026-06-24 preserves 251/251 coverage, keeps 31 field days, drops the route
    count 44 -> 43, and reprices the owner day to 143 p75 / 163 p90. The
    source day is 204 p75 / 229 p90 on 2026-06-18. It remains blocked from
    active deletion until FD04A claims and cues Shane's segments 1649/1650/1651
    as credit or a post-run segment-first validation proves the source activity.
  - `FD14B -> FD14A` is supported as an experiment only: deleting FD14A on the
    same 2026-07-08 field day preserves 251/251 coverage, keeps 31 field days,
    drops the route count 44 -> 43, and reprices the day to 115 p75 / 137 p90.
    It remains blocked from active deletion until FD14B claims and cues Doe
    Ridge segment 1541 as credit or a post-run segment-first validation proves
    the source activity.
  - Combined non-overlapping portfolio remains 5.84 on-foot miles, 167 p75
    minutes, and 188 p90 minutes of hypothetical savings, with active menu
    deletion count still 0.
- Evidence artifacts:
  - `docs/field-packet/field-tool-data.json`
  - `years/2026/checkpoints/human-executable-field-day-layer-2026-05-10.md`
  - `years/2026/checkpoints/latent-credit-delta-repricing-audit-2026-05-12.md`
  - `years/2026/checkpoints/calendar-reorder-latent-credit-experiment-2026-05-12.md`
  - `years/2026/checkpoints/current-calendar-skip-ready-promotion-audit-2026-05-12.md`
- Validation:
  - `python years/2026/scripts/field_latent_credit_audit.py --output-json /tmp/boise-field-latent-credit-current.json --output-md /tmp/boise-field-latent-credit-current.md --report-only`
    passed with 44 routes, 0 routes needing repair, 38 reconciled latent
    segments, and 0 unclaimed/uncompleted latent segments.
  - `python years/2026/scripts/latent_credit_delta_repricing_audit.py --latent-credit-audit-json /tmp/boise-field-latent-credit-current.json --output-json /tmp/boise-latent-credit-delta-current-freshlatent.json --output-md /tmp/boise-latent-credit-delta-current-freshlatent.md --manifest-json /tmp/boise-latent-credit-delta-current-freshlatent-manifest.json`
    passed with 2 pairwise full-removal relationships and 0 current-calendar
    savings.
  - `python years/2026/scripts/calendar_reorder_for_latent_credit_experiment.py --latent-delta-audit-json /tmp/boise-latent-credit-delta-current-freshlatent.json --output-json /tmp/boise-calendar-reorder-current-freshlatent.json --output-md /tmp/boise-calendar-reorder-current-freshlatent.md --manifest-json /tmp/boise-calendar-reorder-current-freshlatent-manifest.json`
    passed with 2 supported reorders and 0 blocked reorders.
  - `python years/2026/scripts/current_calendar_skip_ready_promotion_audit.py --latent-repricing-audit-json /tmp/boise-latent-credit-delta-current-freshlatent.json --output-json /tmp/boise-current-calendar-skip-ready-current-freshlatent.json --output-md /tmp/boise-current-calendar-skip-ready-current-freshlatent.md --manifest-json /tmp/boise-current-calendar-skip-ready-current-freshlatent-manifest.json`
    passed with `no_skip_ready_removals`.
  - `pytest -q years/2026/tests/test_latent_credit_delta_repricing_audit.py years/2026/tests/test_calendar_reorder_for_latent_credit_experiment.py`
    passed 5 tests in 0.07s.

## 2026-05-13 - FD04A -> FD19C focused route-card promotion path

- Objective:
  - Test only the ownership/calendar route-card promotion path for letting
    FD04A claim and cue FD19C Shane's Trail segments `1649`, `1650`, and
    `1651`, then remove FD19C after the calendar reorder. Do not generate new
    Freestone mega-routes.
- Result:
  - The focused experiment passed. FD04A now claims segments `1649`, `1650`,
    and `1651` in the temp promoted map data, with phone-visible credit cues for
    Shane's Trail 2, Shane's Trail 3, and Shane's Trail 1.
  - FD19C is absent from the temp final route list and from the temp field-day
    layer. The combined promotion report skips one FD19C source loop.
  - Coverage stays `251/251`; route count and field-day loop count are `43`.
  - Calendar reassignment keeps 31 field days, total p75 `6846`, max p90 `359`,
    and 0 p90/date/route-card audit violations.
  - Repeat and latent-credit checks are clean: 0 hidden self-repeat, 0 unpriced
    repeat, 0 unreconciled latent credit, and 0 official repeat legs missing
    segment ids after the promoted Shane's repeat annotation was cleaned up.
  - No new Freestone route-card source promotion was generated.
- Evidence artifacts:
  - `years/2026/checkpoints/fd04a-fd19c-route-card-promotion-path-2026-05-13.md`
  - `years/2026/checkpoints/fd04a-fd19c-route-card-promotion-path-2026-05-13.json`
  - `years/2026/checkpoints/fd04a-fd19c-route-card-promotion-path-2026-05-13-prep.md`
  - Private generated packet and audits under
    `years/2026/outputs/private/fd04a-fd19c-route-card-promotion-path-2026-05-13/`.
- Validation:
  - `python years/2026/scripts/fd04a_fd19c_route_card_promotion_path_experiment.py prepare`
    passed and wrote the experiment-only segment-promotion and reordered-calendar
    inputs.
  - `python years/2026/scripts/fd04a_fd19c_route_card_promotion_path_experiment.py materialize`
    passed with `route_card_count: 43`.
  - `python years/2026/scripts/export_mobile_field_packet.py --map-data-json years/2026/outputs/private/fd04a-fd19c-route-card-promotion-path-2026-05-13/focused-promoted-map-data.json --field-day-layer-json years/2026/outputs/private/fd04a-fd19c-route-card-promotion-path-2026-05-13/field-day-layer.public-safe.json --output-dir years/2026/outputs/private/fd04a-fd19c-route-card-promotion-path-2026-05-13/field-packet-final`
    passed and wrote 129 GPX files.
  - `python years/2026/scripts/export_field_day_layer.py --assignment-json years/2026/outputs/private/fd04a-fd19c-route-card-promotion-path-2026-05-13/fd04a-fd19c-calendar-reordered.json --field-tool-json years/2026/outputs/private/fd04a-fd19c-route-card-promotion-path-2026-05-13/field-packet-focused-initial/field-tool-data.json --promotion-json years/2026/outputs/private/fd04a-fd19c-route-card-promotion-path-2026-05-13/combined-route-card-promotion-report.json --output-json years/2026/outputs/private/fd04a-fd19c-route-card-promotion-path-2026-05-13/field-day-layer.json --output-md years/2026/outputs/private/fd04a-fd19c-route-card-promotion-path-2026-05-13/field-day-layer.md --manifest-json years/2026/outputs/private/fd04a-fd19c-route-card-promotion-path-2026-05-13/field-day-layer-manifest.json`
    passed with `loop_count: 43`, `covered_segment_count: 251`, and
    `schedule_p90_violation_day_count: 0`.
  - `python years/2026/scripts/field_official_repeat_audit.py ...` passed with
    `repeat_legs_missing_segment_ids: 0` and
    `unreconciled_extra_credit_segment_count: 0`.
  - `python years/2026/scripts/route_repeat_optimization_audit.py ...` passed
    with 0 hidden self-repeat, 0 latent-credit repeat, and 0 unpriced repeat.
  - `python years/2026/scripts/field_latent_credit_audit.py ...` passed with 43
    routes and 0 unclaimed/uncompleted latent segments.
  - `python years/2026/scripts/field_progress_report.py ...` passed with
    `remaining_coverage_preserved: true`.
  - `python years/2026/scripts/field_recertification_report.py ...` passed with
    `remaining_full_completion_feasible: true`.
  - `python years/2026/scripts/field_route_walkthrough_audit.py ...` passed
    43/43 routes.
  - `python years/2026/scripts/field_tool_completion_audit.py ...` passed 15/15.
  - `python years/2026/scripts/fd04a_fd19c_route_card_promotion_path_experiment.py verify ...`
    passed all focused gates.
  - `pytest -q years/2026/tests/test_fd04a_fd19c_route_card_promotion_path_experiment.py`
    passed 3 tests.

## 2026-05-13 - FD04A -> FD19C active packet promotion

- Objective:
  - Promote the already-proven FD04A ownership path into the active canonical
    packet: move FD04A before FD19C, claim and cue Shane's Trail segments
    `1649`, `1650`, and `1651` on FD04A, remove FD19C, regenerate the field-day
    layer and phone packet, and rerun certification gates.
- Result:
  - Active packet now has 43 route cards and 31 field days with `251/251`
    official segments covered.
  - FD04A is scheduled on 2026-06-18 and now claims segments `1558`, `1649`,
    `1650`, `1651`, `1652`, and `1748`; FD19C is removed from the active
    route list and field-day layer.
  - Field-day layer remains certified with total p75 `6846`, max p90 `359`, 0
    p90 violations, 0 route-card promotion gaps, 0 route-card audit-fix gaps,
    and day-level GPX validation passed.
  - Completion audit passed 15/15 requirements; route-repeat optimization has 0
    hidden self-repeat, 0 latent-credit repeat, and 0 unpriced repeat segments.
  - The historical FD04A experiment script now reports
    `active_packet_already_promoted` when run against the active packet instead
    of expecting FD19C to remain present.
- Evidence artifacts:
  - `docs/field-packet/field-tool-data.json`
  - `years/2026/inputs/personal/2026-cross-package-segment-promotions-v1.json`
  - `years/2026/checkpoints/fd04a-fd19c-calendar-assignment-2026-05-13.json`
  - `years/2026/checkpoints/fd04a-fd19c-route-card-promotion-report-2026-05-13.json`
  - `years/2026/checkpoints/fd04a-fd19c-route-card-promotion-path-2026-05-13.md`
  - `years/2026/checkpoints/human-executable-field-day-layer-2026-05-10.md`
  - `years/2026/checkpoints/field-tool-completion-audit-2026-05-06.md`
- Validation:
  - `python years/2026/scripts/fd04a_fd19c_route_card_promotion_path_experiment.py prepare --output-calendar-json years/2026/checkpoints/fd04a-fd19c-calendar-assignment-2026-05-13.json --report-json years/2026/checkpoints/fd04a-fd19c-route-card-promotion-path-2026-05-13-prep.json --report-md years/2026/checkpoints/fd04a-fd19c-route-card-promotion-path-2026-05-13-prep.md`
    passed with 3 promotion rows.
  - `python years/2026/scripts/fd04a_fd19c_route_card_promotion_path_experiment.py materialize --calendar-json years/2026/checkpoints/fd04a-fd19c-calendar-assignment-2026-05-13.json --output-map-data-json years/2026/outputs/private/2026-outing-menu-map-data.json --output-promotion-report-json years/2026/checkpoints/fd04a-fd19c-route-card-promotion-report-2026-05-13.json`
    passed with `route_card_count: 43`.
  - `python years/2026/scripts/export_mobile_field_packet.py --field-day-layer-json /tmp/no-field-day-layer-fd04a.json`
    passed and wrote the initial regenerated packet.
  - `python years/2026/scripts/export_field_day_layer.py` passed with
    `loop_count: 43`, `covered_segment_count: 251`, and
    `schedule_p90_violation_day_count: 0`.
  - `python years/2026/scripts/export_mobile_field_packet.py` passed and wrote
    the final packet with the regenerated field-day layer.
  - `python years/2026/scripts/export_example_map.py` passed and regenerated the
    public-safe example map/menu.
  - `python years/2026/scripts/field_official_repeat_audit.py` passed.
  - `python years/2026/scripts/route_repeat_optimization_audit.py` passed.
  - `python years/2026/scripts/field_latent_credit_audit.py` passed.
  - `python years/2026/scripts/field_progress_report.py` passed.
  - `python years/2026/scripts/field_recertification_report.py --calendar-json years/2026/checkpoints/fd04a-fd19c-calendar-assignment-2026-05-13.json`
    passed.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 43/43
    routes.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 15/15.
  - `python years/2026/scripts/fd04a_fd19c_route_card_promotion_path_experiment.py verify --field-tool-data-json docs/field-packet/field-tool-data.json --field-day-layer-json years/2026/checkpoints/human-executable-field-day-layer-2026-05-10.json --promote-report-json years/2026/checkpoints/fd04a-fd19c-route-card-promotion-report-2026-05-13.json --route-repeat-json years/2026/checkpoints/route-repeat-optimization-audit-2026-05-12.json --latent-json years/2026/checkpoints/field-latent-credit-audit-2026-05-11.json --progress-json years/2026/outputs/private/progress/field-progress-latest.json --recertification-json years/2026/outputs/private/progress/field-recertification-latest.json --completion-json years/2026/checkpoints/field-tool-completion-audit-2026-05-06.json --walkthrough-json years/2026/checkpoints/field-route-walkthrough-audit-2026-05-06.json --verify-report-json years/2026/checkpoints/fd04a-fd19c-route-card-promotion-path-2026-05-13.json --verify-report-md years/2026/checkpoints/fd04a-fd19c-route-card-promotion-path-2026-05-13.md`
    passed all focused gates.
  - `python years/2026/scripts/fd04a_fd19c_credit_promotion_experiment.py`
    passed and regenerated the historical experiment artifact with
    `active_packet_already_promoted`.
  - `python -m json.tool` passed for the active promotions JSON, FD04A focused
    verification JSON, historical FD04A experiment JSON, field-day layer JSON,
    field-tool data JSON, and completion-audit JSON.
  - `pytest -q years/2026/tests/test_fd04a_fd19c_route_card_promotion_path_experiment.py years/2026/tests/test_fd04a_fd19c_credit_promotion_experiment.py years/2026/tests/test_export_field_day_layer.py years/2026/tests/test_export_mobile_field_packet.py years/2026/tests/test_promote_field_day_loops.py years/2026/tests/test_field_tool_completion_audit.py years/2026/tests/test_field_official_repeat_audit.py years/2026/tests/test_field_latent_credit_audit.py years/2026/tests/test_route_repeat_optimization_audit.py years/2026/tests/test_field_route_walkthrough_audit.py`
    passed 118 tests in 112.72s.

## 2026-05-13 - Post-H1 field-day cleanup

- Objective:
  - Complete the open post-H1 cleanup items: repair stale single-loop schedule
    timing, move the final Bogus day off the last challenge date if the reserve
    slot can absorb it, label reserve days explicitly, split calendar days from
    active execution days, and stop presenting weekday/weekend as capacity
    proof.
- Result:
  - `16A-2` is repaired generically by the field-day exporter. Single-loop
    field days now use the certified route-card door-to-door p75/p90 unless an
    explicit timing override explains the calendar value.
  - `16A-2` on 2026-07-11 changed from stale `310/348` to route-card `106/119`.
  - The same generic check also repaired three one-minute p90 rounding
    mismatches on single-loop days; no unrepaired single-loop timing mismatch
    remains.
  - Bogus route `18` moved from 2026-07-18 to the 2026-07-12 reserve slot.
    2026-07-18 is now the final reserve/buffer day.
  - The field-day summary now reports 31 calendar days, 29 active execution
    days, and 2 reserve days (`2026-06-21`, `2026-07-18`).
  - Empty field days render as `Reserve / buffer day - no route planned.` in
    both the field-day markdown and phone packet.
  - The field-day layer now exposes `available_minutes_p90` per date and marks
    `day_type_capacity_proxy_used: false`; weekday/weekend labels remain
    context only.
  - Pushback: no real per-date personal availability windows were invented.
    The current artifact uses the existing dated p90 bounds from the calendar
    assignment. Replacing the upstream optimizer's availability model with real
    hard-stop windows still needs an authoritative personal availability input.
  - Active packet remains 43 route cards, 31 calendar days, 29 active execution
    days, 13 multi-start days, 251/251 official coverage, 0 p90 violations, and
    field-day certified. Total p75 is now `6642`; max p90 remains `359`.
- Evidence artifacts:
  - `years/2026/checkpoints/post-h1-cleanup-calendar-assignment-2026-05-13.json`
  - `years/2026/checkpoints/post-h1-cleanup-calendar-assignment-2026-05-13-report.md`
  - `years/2026/checkpoints/human-executable-field-day-layer-2026-05-10.md`
  - `docs/field-packet/field-tool-data.json`
  - `docs/field-packet/index.html`
  - `years/2026/checkpoints/fd04a-fd19c-route-card-promotion-path-2026-05-13.md`
- Validation:
  - `python years/2026/scripts/post_h1_field_day_cleanup.py` passed and wrote
    the cleaned assignment plus report/manifest.
  - `python years/2026/scripts/export_field_day_layer.py` passed with 31
    calendar days, 29 active execution days, 2 reserve days, total p75 `6642`,
    max p90 `359`, 4 single-loop timing repairs, and 0 unrepaired mismatches.
  - `python years/2026/scripts/export_mobile_field_packet.py` passed and wrote
    129 GPX files plus the regenerated phone packet.
  - `python years/2026/scripts/export_example_map.py` passed and regenerated
    the public-safe map/menu artifacts.
  - `python years/2026/scripts/field_official_repeat_audit.py` passed.
  - `python years/2026/scripts/route_repeat_optimization_audit.py` passed.
  - `python years/2026/scripts/field_latent_credit_audit.py` passed.
  - `python years/2026/scripts/field_progress_report.py` passed.
  - `python years/2026/scripts/field_recertification_report.py` passed.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 43/43
    routes.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 15/15.
  - `python years/2026/scripts/fd04a_fd19c_route_card_promotion_path_experiment.py verify --field-tool-data-json docs/field-packet/field-tool-data.json --field-day-layer-json years/2026/checkpoints/human-executable-field-day-layer-2026-05-10.json --promote-report-json years/2026/checkpoints/fd04a-fd19c-route-card-promotion-report-2026-05-13.json --route-repeat-json years/2026/checkpoints/route-repeat-optimization-audit-2026-05-12.json --latent-json years/2026/checkpoints/field-latent-credit-audit-2026-05-11.json --progress-json years/2026/outputs/private/progress/field-progress-latest.json --recertification-json years/2026/outputs/private/progress/field-recertification-latest.json --completion-json years/2026/checkpoints/field-tool-completion-audit-2026-05-06.json --walkthrough-json years/2026/checkpoints/field-route-walkthrough-audit-2026-05-06.json --verify-report-json years/2026/checkpoints/fd04a-fd19c-route-card-promotion-path-2026-05-13.json --verify-report-md years/2026/checkpoints/fd04a-fd19c-route-card-promotion-path-2026-05-13.md`
    passed all focused gates.
  - `python years/2026/scripts/fd04a_fd19c_credit_promotion_experiment.py`
    passed with `active_packet_already_promoted`.
  - `python -m json.tool` passed for the post-H1 calendar assignment/report,
    field-day layer JSON, field-tool data JSON, completion-audit JSON, and
    FD04A focused verifier JSON.
  - `python -m py_compile years/2026/scripts/export_field_day_layer.py years/2026/scripts/export_mobile_field_packet.py years/2026/scripts/post_h1_field_day_cleanup.py years/2026/scripts/field_recertification_report.py`
    passed.
  - `pytest -q years/2026/tests/test_post_h1_field_day_cleanup.py years/2026/tests/test_export_field_day_layer.py years/2026/tests/test_export_mobile_field_packet.py years/2026/tests/test_field_recertification_report.py years/2026/tests/test_fd04a_fd19c_route_card_promotion_path_experiment.py years/2026/tests/test_fd04a_fd19c_credit_promotion_experiment.py years/2026/tests/test_promote_field_day_loops.py years/2026/tests/test_field_tool_completion_audit.py years/2026/tests/test_field_official_repeat_audit.py years/2026/tests/test_field_latent_credit_audit.py years/2026/tests/test_route_repeat_optimization_audit.py years/2026/tests/test_field_route_walkthrough_audit.py`
    passed 130 tests in 118.69s.

## 2026-05-15 - Accepted-anchor preservation regression fix

- Objective: repair the FD14D Full Sail / lower-36th accepted-anchor regression
  and add a durable preservation gate so certified/runnable route cards cannot
  be preserved when an accepted anchor makes the same official segment set
  materially dominated.
- Result:
  - Added a public-safe accepted route replacement manifest at
    `years/2026/inputs/accepted-route-replacements-v1.json`.
  - `FD14D` now regenerates from `Full Sail Trailhead, N 36th St Parking` for
    segment `1482` instead of preserving the stale Full Sail parked start.
    The regenerated field packet shows `0.74` official miles, `1.50` on-foot
    miles, p75 `60`, p90 `68`, `route_card_status:
    provisional_re_anchored`, `packet_visibility:
    visible_with_provisional_badge`, `certified_route_card: false`,
    `requires_field_walkthrough: true`, and `cue_generation_mode:
    regenerated_for_reanchored_candidate`.
  - `FD09A` is not auto-promoted. It is marked
    `route_card_status: investigation_required`, `packet_visibility:
    visible_with_investigation_badge`, `certified_route_card: false`, and
    `requires_field_walkthrough: true` until the Barn Owl / West Hidden Springs
    geography is proven.
  - Package `114` remains multi-start/re-park aware: Cartwright for `FD14A/B`
    and lower 36th for `FD14D`; it was not collapsed into one car-to-car loop.
  - Added a generic accepted-anchor preservation audit that fails when a
    material accepted-anchor dominance candidate is missing a manifest record
    or when a manifest record is not reflected in the field packet.
- Evidence artifacts:
  - `years/2026/inputs/accepted-route-replacements-v1.json`
  - `years/2026/checkpoints/accepted-anchor-preservation-audit-2026-05-15.md`
  - `years/2026/checkpoints/field-day-loop-promotion-2026-05-11.md`
  - `years/2026/outputs/private/2026-outing-menu-map-data.json`
  - `docs/field-packet/field-tool-data.json`
  - `docs/field-packet/index.html`
- Validation:
  - `pytest -q years/2026/tests/test_accepted_anchor_preservation_audit.py years/2026/tests/test_export_field_day_layer.py years/2026/tests/test_promote_field_day_loops.py`
    passed 28 tests in 0.82s before packet integration.
  - `pytest -q years/2026/tests/test_accepted_anchor_preservation_audit.py years/2026/tests/test_export_field_day_layer.py years/2026/tests/test_export_mobile_field_packet.py years/2026/tests/test_promote_field_day_loops.py`
    passed 85 tests in 104.01s.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py years/2026/tests/test_promote_field_day_loops.py`
    passed 69 tests in 110.64s.
  - `pytest -q years/2026/tests/test_accepted_anchor_preservation_audit.py`
    passed 3 tests in 0.05s after adding audit detail fields.
  - `python -m py_compile years/2026/scripts/accepted_route_replacements.py years/2026/scripts/accepted_anchor_preservation_audit.py years/2026/scripts/export_field_day_layer.py years/2026/scripts/promote_field_day_loops.py years/2026/scripts/export_mobile_field_packet.py years/2026/scripts/block_day_packager.py`
    passed.
  - `python years/2026/scripts/promote_field_day_loops.py` passed and wrote the
    promoted private map/menu/report with `accepted_replacement_blocker_count:
    2`.
  - `python years/2026/scripts/export_mobile_field_packet.py` passed and wrote
    141 GPX files plus the regenerated phone packet.
  - `python years/2026/scripts/export_field_day_layer.py` passed with 31
    calendar days, 29 active execution days, 251/251 coverage, and 0 p90
    violations.
  - `python years/2026/scripts/accepted_anchor_preservation_audit.py` passed
    with 1 manifest-covered accepted-anchor dominance candidate, 2 manifest
    records, 0 manifest failures, and 0 missing manifest records.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 47/47
    routes.
  - `python years/2026/scripts/field_recertification_report.py` passed and
    preserved 251/251 remaining official segment coverage.
  - `python years/2026/scripts/route_repeat_optimization_audit.py` passed with
    0 hidden self-repeat, 0 latent-credit, and 0 unpriced-repeat hard failures;
    it still reports 51 optimization warnings for future review.
  - `python years/2026/scripts/field_official_repeat_audit.py --map-data-json years/2026/outputs/private/2026-outing-menu-map-data.json`
    passed with 0 bad hidden self-repeat and 0 unreconciled extra-credit
    segments.
  - `python years/2026/scripts/field_latent_credit_audit.py` passed with 0
    routes needing repair.
  - `python years/2026/scripts/route_efficiency_audit.py --map-data-json years/2026/outputs/private/2026-outing-menu-map-data.json`
    completed and wrote the efficiency audit artifacts; it still reports known
    time-estimate quality advisories.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 15/15
    requirements.

## 2026-05-15 - Route-review gate implementation

- Objective: make the FD14D lesson a repo-native 2026 route-promotion gate
  rather than a separate AI-review side project.
- Result:
  - Added `docs/route-review-policy.md`, route-review doctrine in `AGENTS.md`,
    and FD14D regression memory in the BTC heuristic/failure/case/eval docs.
  - Added deterministic route-review pack generation, exact-credit dominance
    review, local Codex review wrapper, structured review schema, and
    route/source-hashed waiver gate under `years/2026/`.
  - Wired `start_justification` from accepted replacement records through
    promotion, outing grouping, field-tool export, and route-review packs.
  - Built the FD14D dev pack. The current field-tool route now passes the
    deterministic gate as `PASS_WITH_JUSTIFIED_BURDEN` because the lower 36th
    accepted replacement is already applied, while the stale Full Sail fixture
    fails as `FAIL_DOMINATED`.
- Evidence artifacts:
  - `years/2026/outputs/private/route-reviews/route-review-fd14d-dev.pack.json`
  - `years/2026/outputs/private/route-reviews/route-review-fd14d-dev.review.json`
  - `years/2026/checkpoints/route-review-fd14d-dev.public.json`
  - `years/2026/checkpoints/route-review-fd14d-dev.public.md`
- Validation:
  - `python years/2026/scripts/build_route_review_pack.py --route-label FD14D --basename route-review-fd14d-dev`
    passed and wrote private plus public-safe route-review artifacts.
  - `python years/2026/scripts/gate_route_reviews.py years/2026/outputs/private/route-reviews/*.review.json`
    passed.
  - `python -m pytest years/2026/tests/test_route_review_pack.py years/2026/tests/test_gate_route_reviews.py`
    passed 9 tests in 0.66s.
  - `python -m pytest years/2026/tests/test_promote_field_day_loops.py years/2026/tests/test_block_day_packager.py years/2026/tests/test_export_mobile_field_packet.py`
    passed 79 tests in 119.86s.
  - `python -m py_compile years/2026/scripts/build_route_review_pack.py years/2026/scripts/gate_route_reviews.py years/2026/scripts/run_ai_route_review.py years/2026/scripts/promote_field_day_loops.py years/2026/scripts/block_day_packager.py years/2026/scripts/export_mobile_field_packet.py`
    passed.
  - `python years/2026/scripts/build_route_review_pack.py --all-field-packet-routes --basename route-review-all-dev`
    passed and reviewed 47 routes. The baseline found 46 deterministic
    blockers: 44 missing `start_justification`, plus `FAIL_DOMINATED` for
    `FD09A` and `FD03A`.
  - `python years/2026/scripts/gate_route_reviews.py years/2026/outputs/private/route-reviews/route-review-all-dev.review.json`
    failed as expected for the first all-routes baseline. `FD14D` passed as
    `PASS_WITH_JUSTIFIED_BURDEN`; `FD09A` was dominated by West Hidden Springs
    Drive road-parking anchor by 1.64 miles / 36 p75 minutes; `FD03A` was
    dominated by a private Strava parking anchor by 1.12 miles / 29 p75
    minutes.

## 2026-05-15 - Route-review gate full repair run

- Objective: resolve the all-routes route-review gate blockers and complete a
  full validation run from regenerated source artifacts.
- Result:
  - Promoted `FD09A` from the user-reviewed West Hidden Springs Drive
    road-parking anchor and `FD03A` from the private Strava-derived Chukar
    Butte anchor as provisional re-anchored cards.
  - Added generator support for active accepted replacements whose accepted
    anchor is a reviewed street-probe anchor, so those starts can be rebuilt
    from connector geometry plus parking-review decisions.
  - Added default `start_justification` generation in field-tool export for
    ordinary field-packet route cards, while accepted replacements carry their
    explicit data-backed justification.
  - Regenerated the promoted map data and mobile field packet. The all-routes
    route-review gate now passes with 47 routes reviewed, 0 deterministic
    failures, 44 `PASS_NON_DOMINATED`, and 3 `PASS_WITH_JUSTIFIED_BURDEN`
    provisional re-anchored cards (`FD03A`, `FD09A`, `FD14D`).
- Validation:
  - `python years/2026/scripts/promote_field_day_loops.py` passed with 47
    component routes, 251/251 covered segments, 3 newly promoted loops, and
    track validation passed.
  - `python years/2026/scripts/export_mobile_field_packet.py` passed and wrote
    141 GPX files plus the regenerated phone packet.
  - `python years/2026/scripts/build_route_review_pack.py --all-field-packet-routes --basename route-review-all-dev`
    passed with 47 routes reviewed and 0 deterministic failures.
  - `python years/2026/scripts/gate_route_reviews.py years/2026/outputs/private/route-reviews/route-review-all-dev.review.json`
    passed.
  - `python years/2026/scripts/accepted_anchor_preservation_audit.py` passed
    with 3 manifest records and 0 failures.
  - `python years/2026/scripts/export_field_day_layer.py` passed with 31
    calendar days, 29 active execution days, 251/251 coverage, and 0 p90
    violations.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 47/47
    routes.
  - `python years/2026/scripts/field_recertification_report.py` passed and
    preserved 251/251 remaining official segment coverage.
  - `python years/2026/scripts/route_repeat_optimization_audit.py` passed with
    0 hidden self-repeat, 0 latent-credit, and 0 unpriced-repeat hard failures;
    it reports 49 optimization warnings.
  - `python years/2026/scripts/field_official_repeat_audit.py --map-data-json years/2026/outputs/private/2026-outing-menu-map-data.json`
    passed.
  - `python years/2026/scripts/field_latent_credit_audit.py` passed with 0
    routes needing repair.
  - `python years/2026/scripts/route_efficiency_audit.py --map-data-json years/2026/outputs/private/2026-outing-menu-map-data.json`
    completed and still reports known time-estimate quality advisories.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 15/15
    requirements.
  - `python -m pytest years/2026/tests` passed 539 tests in 116.13s.

## 2026-05-16 - Route-review advisory closure

- Objective: evaluate and resolve or explicitly close the remaining
  non-blocking notes from the route-review gate full run: 49 route-repeat
  optimization warnings and route-efficiency time-estimate quality advisories.
- Result:
  - Resolved the route-efficiency time-quality advisory. The audit was treating
    incomplete top-level zero `effort` placeholders as authoritative instead of
    falling back to segment-level DEM effort. `route_efficiency_audit.py` now
    accepts segment-level DEM effort when top-level effort is absent or
    incomplete.
  - Refreshed `route-efficiency-audit-2026-05-06.json`; time quality now has 0
    problems, 0 missing p75, 0 stale p75, 0 missing moving p75, and 0 missing
    DEM effort. The audit still says `not_proven` overall because of broader
    optimization-proof gaps, not timing-data quality.
  - Closed the 49 route-repeat optimization warnings as non-blocking backlog in
    the route-repeat audit output. The refreshed route-repeat audit still has 0
    hidden self-repeat, 0 latent-credit, 0 unpriced-repeat, and 0 missing-GPX
    hard failures.
  - Refreshed ownership/repeat-productivity classification. It still shows
    optimization backlog, but ownership reassignment reports 0 current-calendar
    skip-ready saved miles; the available savings require calendar reorder, so
    they are not a quick route-review-gate patch.
- Evidence artifact:
  - `years/2026/checkpoints/route-review-advisory-closure-2026-05-16.md`
- Validation:
  - `python -m pytest years/2026/tests/test_route_efficiency_audit.py years/2026/tests/test_route_repeat_optimization_audit.py years/2026/tests/test_repeat_productivity_audit.py`
    passed 24 tests in 0.11s.
  - `python years/2026/scripts/route_efficiency_audit.py --map-data-json years/2026/outputs/private/2026-outing-menu-map-data.json`
    regenerated the audit with 0 time-estimate quality problems.
  - `python years/2026/scripts/route_repeat_optimization_audit.py` passed with
    47 routes, 0 hard failures, and 49 warnings closed as non-blocking
    optimization backlog.
  - `python years/2026/scripts/ownership_reassignment_optimization_audit.py`
    refreshed ownership optimization classification.
  - `python years/2026/scripts/repeat_productivity_audit.py` refreshed repeat
    productivity classification for the 47-route packet.
  - `python -m pytest years/2026/tests` passed 541 tests in 118.42s.

## 2026-05-16 - Public map artifact drift repair

- Objective: reconcile the root public outing map/menu artifacts with the
  corrected 2026 private canonical map data after the FD03A/FD09A/FD14D
  re-anchors.
- Result:
  - Regenerated `outing-menu.md`, `outing-menu-map.html`,
    `outing-menu-map-data.json`, and the matching example artifacts from
    `years/2026/outputs/private/2026-outing-menu-map-data.json`.
  - Fixed the public example exporter so the accepted lower-36th FD14D anchor
    remains visible as public-safe map data, while the private Chukar Butte
    prior-parking anchor is label-sanitized and its route/parking geometry is
    redacted from public map data.
  - Added regression coverage that fails when root/example public map data or
    embedded HTML data drift from the private canonical metrics for FD03A,
    FD09A, and FD14D.
- Validation:
  - `python years/2026/scripts/export_example_map.py` passed and regenerated
    the root/example public map and written menu artifacts.
  - `python -m json.tool outing-menu-map-data.json >/dev/null` passed.
  - `python -m json.tool years/2026/outputs/examples/2026-outing-menu-map-data.example.json >/dev/null`
    passed.
  - `python -m json.tool years/2026/outputs/private/2026-outing-menu-map-data.json >/dev/null`
    passed.
  - `python -m pytest years/2026/tests/test_export_example_map.py years/2026/tests/test_public_map_artifact_consistency.py`
    passed 9 tests in 0.45s.
  - `git diff --check -- years/2026/scripts/export_example_map.py years/2026/tests/test_export_example_map.py years/2026/tests/test_public_map_artifact_consistency.py outing-menu.md outing-menu-map-data.json outing-menu-map.html years/2026/outputs/examples/2026-outing-menu.example.md years/2026/outputs/examples/2026-outing-menu-map-data.example.json years/2026/outputs/examples/2026-outing-menu-map.example.html`
    passed.
  - `python -m pytest years/2026/tests` passed 546 tests in 123.14s.

## 2026-05-16 - Full manual route map-challenge review

- Objective: manually challenge all 47 current field-packet route cards from
  route-card, runner/local-map, partition, and adversarial frames, then compare
  the pre-review route-review gate confidence with the post-review human
  fatigue risk.
- Result:
  - Added `years/2026/checkpoints/manual-route-map-challenge-2026-05-16/` with
    the report, route-by-route notes, and manifest.
  - Found a blocking route-truth contradiction: the current generated field
    packet has 47 routes and still contains `FD24A`, `FD27A`, `FD27B`, `FD27C`,
    and `FD30A`, while the tracked H1 active-packet certification checkpoint
    says those five cards were removed and replaced by certified `H1`.
  - Quantified the modeled H1 reconciliation effect against the current packet:
    289.58 to 265.22 on-foot miles, 7662 to 6960 p75 minutes, and 1.761x to
    1.613x on-foot/official ratio, assuming H1 remains valid and no other route
    changes are made.
  - Confirmed FD14D itself remains fixed from the prior exact-credit dominance
    issue; the new finding is an artifact/source-truth and partition regression,
    not a same-credit accepted-anchor failure.
- Validation:
  - `python -m json.tool years/2026/checkpoints/manual-route-map-challenge-2026-05-16/manifest.json >/tmp/manual-route-map-challenge-manifest.pretty.json`
    passed.
  - A manifest-vs-field-packet label check confirmed 47/47 current route labels
    are represented with no missing or extra labels.
  - `git diff --check -- years/2026/checkpoints/manual-route-map-challenge-2026-05-16`
    passed.
  - `python -m pytest years/2026/tests/test_public_map_artifact_consistency.py years/2026/tests/test_route_review_pack.py years/2026/tests/test_gate_route_reviews.py`
    passed 14 tests in 1.34s.

## 2026-05-16 - H1 packet reconciliation and adversarial disproof closure

- Objective: repair the blocking H1 contradiction from the manual map-challenge
  review, then try to disprove the remaining weird high-ratio, high-overhead,
  declared-repeat, and same-trailhead routes before calling the packet clean.
- Result:
  - Repointed the field-day/export path at the H1 route-card promotion payload
    and made the H1 route-count assertions data-derived. The active packet now
    has 43 routes, includes `H1`, removes `FD24A`, `FD27A`, `FD27B`, `FD27C`,
    and `FD30A`, and preserves 251/251 official segment coverage.
  - Added `years/2026/checkpoints/adversarial-route-disproof-2026-05-16.*`,
    a manual proof ledger that attacks the suspicious route groups from
    accepted-anchor, same-trailhead, boundary-recombination, global-optimizer,
    and skeptical-hiker frames.
  - Updated the efficiency audit to count proofed historical challenge targets
    separately from active split-route candidate ids. The refreshed efficiency
    audit is now `proven`: 164.43 official miles, 265.22 on-foot miles,
    1.613x, and no failed gates under the current single-car,
    public-road-allowed, p75-aware proof rules.
  - Updated the repeat optimization audit to distinguish total warning pressure
    from open warning pressure. The refreshed repeat audit still records 39
    total optimization warning signals, but all 39 are closed by the adversarial
    route-disproof registry; open optimization warnings are now 0.
- Validation:
  - `python years/2026/scripts/promote_harlow_h1_route_card.py` passed and
    wrote the 43-route H1 promotion checkpoint.
  - `python years/2026/scripts/export_mobile_field_packet.py` passed after the
    H1 promotion and field-day export.
  - `python years/2026/scripts/export_field_day_layer.py` passed with 31 field
    days, 43 loops, and 251/251 coverage.
  - `python years/2026/scripts/harlow_h1_promotion_assertions.py` passed 19/19
    assertions.
  - `python years/2026/scripts/build_route_review_pack.py --all-field-packet-routes --basename route-review-all-dev`
    reviewed 43 routes with 0 deterministic failures.
  - `python years/2026/scripts/gate_route_reviews.py years/2026/outputs/private/route-reviews/route-review-all-dev.review.json`
    passed.
  - `python years/2026/scripts/route_efficiency_audit.py --map-data-json years/2026/outputs/private/2026-outing-menu-map-data.json`
    regenerated a `proven` audit with no failed gates.
  - `python years/2026/scripts/route_repeat_optimization_audit.py` passed with
    0 hard failures, 0 open optimization warnings, and 39/39 warnings closed by
    route disproof.
  - `python years/2026/scripts/field_official_repeat_audit.py --map-data-json years/2026/outputs/private/2026-outing-menu-map-data.json`
    passed.
  - `python years/2026/scripts/field_latent_credit_audit.py` passed with 0
    routes needing repair.
  - `python years/2026/scripts/field_progress_report.py` and
    `python years/2026/scripts/field_recertification_report.py` passed and
    preserved remaining full-completion feasibility.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed 15/15
    requirements.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 43/43
    routes.
  - `python -m pytest years/2026/tests/test_export_field_day_layer.py years/2026/tests/test_post_h1_field_day_cleanup.py years/2026/tests/test_harlow_h1_promotion_assertions.py years/2026/tests/test_route_efficiency_audit.py years/2026/tests/test_route_repeat_optimization_audit.py years/2026/tests/test_public_map_artifact_consistency.py years/2026/tests/test_route_review_pack.py years/2026/tests/test_gate_route_reviews.py years/2026/tests/test_export_mobile_field_packet.py`
    passed 110 tests in 113.02s.
  - `python -m pytest years/2026/tests` passed 551 tests in 120.22s.

## 2026-05-16 - All-route adversarial disproof pass

- Objective: strengthen the grouped route-disproof closure into an explicit
  route-by-route adversarial ledger for every current 2026 field-packet route.
- Result:
  - Added `years/2026/checkpoints/all-route-adversarial-disproof-2026-05-16.*`
    with 43 route proof records.
  - Each current route is attacked from exact-credit/start, bundle/partition,
    runnable-cost, field-artifact, and global-displacement frames.
  - Decisions remain bounded: 13 routes are held as current, 12 are held despite
    bundle pressure, 9 are held despite high-cost pressure, 6 Bogus routes stay
    condition-gated, and 3 accepted replacements (`FD03A`, `FD09A`, `FD14D`)
    remain field-walkthrough gated.
  - Wired the all-route proof registry into the route-efficiency and
    route-repeat optimization audits so future runs require the per-route
    disproof record, not only the older group-level proof.
- Validation:
  - `python -m json.tool years/2026/checkpoints/all-route-adversarial-disproof-2026-05-16.json`
    passed, along with the matching manifest JSON file.
  - `python years/2026/scripts/build_route_review_pack.py --all-field-packet-routes --basename route-review-all-dev`
    reviewed 43 routes with 0 deterministic failures.
  - `python years/2026/scripts/gate_route_reviews.py years/2026/outputs/private/route-reviews/route-review-all-dev.review.json`
    passed.
  - `python years/2026/scripts/route_efficiency_audit.py --map-data-json years/2026/outputs/private/2026-outing-menu-map-data.json`
    regenerated a `proven` audit with 43 accepted active route-proof ids.
  - `python years/2026/scripts/route_repeat_optimization_audit.py` passed with
    0 hard failures, 0 open optimization warnings, and 39/39 warnings closed.
  - `python years/2026/scripts/field_latent_credit_audit.py`,
    `python years/2026/scripts/field_official_repeat_audit.py --map-data-json years/2026/outputs/private/2026-outing-menu-map-data.json`,
    `python years/2026/scripts/field_tool_completion_audit.py`, and
    `python years/2026/scripts/field_route_walkthrough_audit.py` passed.
  - `python -m pytest years/2026/tests/test_all_route_adversarial_disproof.py years/2026/tests/test_route_efficiency_audit.py years/2026/tests/test_route_repeat_optimization_audit.py`
    passed 25 tests in 0.31s.
  - `python -m pytest years/2026/tests` passed 555 tests in 123.65s.

## 2026-05-16 - Public-source route reevaluation

- Objective: rerun the route-proof confidence frame against current public
  sources after the adversarial disproof pass, specifically looking for access,
  closure, direction, or condition evidence that scripts would not invent.
- Result:
  - Added `years/2026/checkpoints/public-source-route-reevaluation-2026-05-16.*`.
  - Initially challenged H1 because Avimor's public owner page frames trail use
    as resident access while the route start proof relied on OSM plus AllTrails.
    The user then confirmed Avimor access, so H1 is restored to
    `accepted_current` with `accepted_user_reviewed` access evidence.
  - Reaffirmed the existing Bogus condition gates through the June 19, 2026
    Deer Point/Pat's/Bogus Basin Road closure window.
  - Reaffirmed day-of signage/date checks for Lower Hulls, Polecat, Around the
    Mountain, and Bucktail-related pedestrian access.
- Validation:
  - `python -m json.tool years/2026/checkpoints/all-route-adversarial-disproof-2026-05-16.json`
    and matching manifest/public-source checkpoint JSON validations passed.
  - `python years/2026/scripts/route_efficiency_audit.py --map-data-json years/2026/outputs/private/2026-outing-menu-map-data.json`
    passed and now reports 43 accepted active route proofs plus 0 public-access
    gated active route proofs.
  - `python years/2026/scripts/route_repeat_optimization_audit.py` passed with
    0 hard failures, 0 open optimization warnings, and 39/39 warnings closed.
  - `python -m pytest years/2026/tests/test_all_route_adversarial_disproof.py years/2026/tests/test_public_source_route_reevaluation.py years/2026/tests/test_route_efficiency_audit.py years/2026/tests/test_route_repeat_optimization_audit.py`
    passed 30 tests in 0.22s after recording user-confirmed Avimor access.
  - `python -m pytest years/2026/tests` passed 560 tests in 106.95s.

## 2026-05-22 - FD18A Polecat directional-rule audit

- Objective: evaluate whether `FD18A` from Cartwright follows the published
  Polecat Loop #81 direction rule for 2026.
- Result:
  - Current Ridge to Rivers public guidance says Polecat Loop #81 is designated
    directional for all users, clockwise through 2026, with multi-directional
    exceptions for the first half-mile from Polecat/Collister and the short
    Cartwright Trailhead to lower Doe Ridge junction section.
  - `FD18A` starts at Cartwright and the generated route/GPX traverses the
    main Polecat loop in the counterclockwise direction before continuing to
    Peggy's Trail. That movement is outside the Cartwright-to-lower-Doe-Ridge
    exception corridor, so the route should not be treated as field-ready under
    the published 2026 direction rule.
  - Existing packet validation and direction evidence pass only the official
    BTC `direction`/`ascent` segment flags; they do not enforce the land-manager
    special-management clockwise rule. This is a planner/audit gap, not just a
    cue wording issue.
- Current blocker:
  - FD18A needs a rule-aware redesign or route hold before field use, and the
    generator/audit chain needs durable special-management direction validation
    for Polecat-style rules.
- Validation:
  - Reviewed `docs/field-packet/field-tool-data.json`,
    `outing-menu-map-data.json`, the FD18A official GPX, and the 2026 official
    segment GeoJSON. No full test suite was run for this audit note.

### Implementation follow-up

- Added a data-backed special-management rule layer at
  `years/2026/inputs/open-data/special-management-rules-2026.json`, starting
  with Polecat, Around the Mountain, Lower Hulls, and Bucktail.
- Added `years/2026/scripts/special_management_rule_audit.py` and wired it into
  `years/2026/scripts/field_tool_completion_audit.py` as a hard requirement:
  published route cards now fail certification when a known land-manager
  special-management rule is violated.
- Marked `FD18A` as not field-ready by audit result rather than by a cue note.
  The generated special-management audit fails `FD18A` with
  `special_management_direction_violated` on the main Polecat Loop segments.
- The same gate also surfaced additional published-route blockers to repair or
  verify before field use: `FD26A` against Around the Mountain counter-clockwise
  direction, plus `FD04A` and route `3` against the Bucktail on-foot mode
  restriction.
- Validation after implementation:
  - `python -m pytest years/2026/tests/test_special_management_rule_audit.py years/2026/tests/test_field_tool_completion_audit.py -q`
    passed 21 tests in 3.00s.
  - `python -m pytest years/2026/tests/test_field_route_walkthrough_audit.py -q`
    passed 11 tests in 0.07s.
  - `python -m json.tool years/2026/inputs/open-data/special-management-rules-2026.json`
    passed.
  - `python -m py_compile years/2026/scripts/special_management_rule_audit.py years/2026/scripts/field_tool_completion_audit.py`
    passed.
  - `python years/2026/scripts/special_management_rule_audit.py` wrote the
    checkpoint and exited non-zero as intended: status `failed`, 4 failed
    routes, with `FD18A` blocked by `special_management_direction_violated`.
  - `python years/2026/scripts/field_tool_completion_audit.py` wrote the
    checkpoint and exited non-zero as intended: status `failed`, 15/16
    requirements passing, with the special-management gate as the failing
    requirement.

## 2026-05-22 - Human route-name regeneration

- Objective: replace route-code/source-derived route titles with names based on
  the starting common trail and major Ridge to Rivers trail system.
- Result:
  - Added R2R open-data-backed route naming in the field-menu generator and
    propagated `route_name`, `route_code`, and name-source metadata through the
    public example map, phone field packet, live map selector, field-day layer,
    and GPX filename/title generation.
  - Duplicate R2R rows such as `Chukar Butte` vs `Chukar Butte (Dog On-Leash)`,
    `Veterans` vs `Veterans (Dog On-Leash)`, and `Polecat Loop` vs
    `Polecat Loop (STM)` now resolve to the clean common trail name when the
    subsystem is unambiguous.
  - Public-facing route names now include examples such as
    `Camels Back / Hulls Gulch: Kestrel`, `Dry Creek: Chukar Butte`,
    `Western Foothills: Veterans`, and `Polecat Gulch: Polecat Loop`.
  - Sanitized the Chukar Butte start display from the private Strava-derived
    source phrase to `Chukar Butte prior parking anchor` in route-facing packet
    surfaces while preserving canonical private source data.
- Validation:
  - `python -m py_compile years/2026/scripts/block_day_packager.py years/2026/scripts/export_mobile_field_packet.py years/2026/scripts/export_example_map.py` passed.
  - `pytest -q years/2026/tests/test_export_example_map.py` passed 5 tests in
    0.49s.
  - `pytest -q years/2026/tests/test_block_day_packager.py years/2026/tests/test_export_mobile_field_packet.py` passed 75 tests in 120.04s.
  - `python years/2026/scripts/export_example_map.py` regenerated the public
    example map/menu artifacts.
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated the
    certified phone packet and wrote 129 GPX files.
  - JSON validation passed for `docs/field-packet/field-tool-data.json`,
    `docs/field-packet/manifest.json`, `outing-menu-map-data.json`, and
    `years/2026/outputs/examples/2026-outing-menu-map-data.example.json`.
  - Generated packet check found 76 route-card headings and 0 code-like `h2`
    headings; all 43 field-tool routes have non-empty human route names.
  - `python years/2026/scripts/field_progress_report.py --output-json /tmp/boise-route-naming-audit/field-progress.json --output-md /tmp/boise-route-naming-audit/field-progress.md`
    passed with `remaining_coverage_preserved: true`.
  - `python years/2026/scripts/field_route_walkthrough_audit.py --output-json /tmp/boise-route-naming-audit/field-route-walkthrough-audit.json --output-md /tmp/boise-route-naming-audit/field-route-walkthrough-audit.md`
    passed all 43 routes.
  - `python years/2026/scripts/field_latent_credit_audit.py --output-json /tmp/boise-route-naming-audit/field-latent-credit-audit.json --output-md /tmp/boise-route-naming-audit/field-latent-credit-audit.md`
    passed with 0 routes needing repair.
- Current blocker:
  - Route naming is complete. The broader field-tool completion audit still
    exits non-zero because the in-progress special-management gate currently
    blocks `FD04A`, route `3`, `FD18A`, and `FD26A`; that blocker predates and
    is separate from route naming.
  - A heavy recertification check with `--run-heavy-optimizer` was stopped after
    it remained CPU-bound for more than ten minutes without output; no repo
    files were written by that aborted optional check.

## 2026-05-22 - Production route legality hold enforcement

- Objective: correct the field-packet state after `FD18A` showed that a route
  could be known-broken against land-manager direction rules while still being
  presented as runnable in the live map.
- Result:
  - Regenerated the phone field packet after the special-management gate was
    wired into route-card readiness, live-map launch state, GPX action links,
    and field-tool summary data.
  - `FD18A` is now held in generated packet data with
    `field_readiness_status: blocked_special_management`, `field_ready: false`,
    and disabled field actions instead of relying on a cue note or day-of
    reminder.
  - The generated route index now reports `39 runnable outings`, `4 held`, and
    `special-management blocks present`; the held cards are visible for repair
    accounting but are not production-runnable route choices.
  - The in-app browser live map for `outing=118-1` was verified to show
    `[HELD] Polecat Gulch: Polecat Loop (FD18A) · Cartwright`, a disabled route
    locate action, and the special-management warning.
- Validation:
  - `python years/2026/scripts/export_mobile_field_packet.py` regenerated the
    packet and wrote 129 GPX files.
  - `python years/2026/scripts/special_management_rule_audit.py` wrote the
    checkpoint and exited non-zero as intended: status `failed`, 43 routes
    checked, 39 passed, and 4 held failures.
  - `python years/2026/scripts/field_tool_completion_audit.py` wrote the
    checkpoint and exited non-zero as intended: status `failed`, 15/16
    requirements passing, with 39 field-ready route cards and 4 held route
    cards.
  - Targeted pytest coverage passed for the special-management audit,
    field-tool completion audit, and live-map/route-card held-route rendering.
- Current blocker:
  - All currently field-ready production routes pass the known
    special-management rule audit. Full field-packet certification is still not
    green because `FD04A`, route `3`, `FD18A`, and `FD26A` need redesign or
    fresher authoritative evidence before they can become field-ready again.

## 2026-05-22 - FD12A live-map cue 09 readability repair and hold

- Objective: repair the FD12A cue 09 live-map surface after field review showed
  a long raw connector cue, sparse active-leg arrows, and no clear arrival
  context for how the runner reaches cue 09; then determine whether cue 09 was
  only hard to read or actually wrong.
- Result:
  - Live-map connector cue text now prefers signed trail labels over generic
    OSM connector ids when signposts are available.
  - The active cue view now keeps the previous cue span visible by default and
    Fit leg frames that approach context with the active leg.
  - Active-leg arrows are generated per visible display segment, so split or
    schematic active legs receive arrows on each usable segment instead of only
    along a single route-mile sample pass.
  - FD12A cue 09 now displays signpost-only guidance:
    `#53 Buena Vista / #52 Kemper's Ridge / #51 Who Now Loop` toward Full Sail.
  - The active-cue mileage label now shows cue mileage separately from the GPX
    map span when they diverge; cue 09 reports `+1.16 mi cue · map +4.49 mi`.
  - Follow-up inspection confirmed the field suspicion: the private FD12A route
    line passes the parked car mid-route, and cue 09 crosses the second car pass
    near route mile 6.32 while spanning 4.50 GPX mi for a 1.16 mi cue.
  - Added a generic navigation-source hold for cue-anchor mismatches that cross
    a parked-car pass. FD12A is now exported as
    `blocked_navigation_source`, with field GPX/live-map actions held until the
    canonical route is repaired and recertified.
  - The same generic hold also caught route `12` for the same failure class.
- Validation:
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py -k "active_cue_leg_navigation_artifact or previous_cue_context or active_leg_direction_arrows or signposts_over_generic_osm_connector_ids or field_packet_surfaces_r2r_signpost_cues or splits_backtracking_cue_legs"`
    passed 6 tests with 60 deselected in 16.77s after the final source edits.
  - `python3 years/2026/scripts/export_mobile_field_packet.py` regenerated the
    packet and wrote 129 GPX files.
  - `python3 -m json.tool docs/field-packet/field-tool-data.json` and
    `python3 -m json.tool docs/field-packet/manifest.json` passed.
  - Extracted generated live-map JavaScript from
    `docs/field-packet/live-map.html`; `node --check /tmp/boise-live-map.js`
    passed.
  - `git diff --check` passed.
  - Browser verification on
    `http://127.0.0.1:8766/live-map.html?outing=112-1&cue=9&v=20260522-fd12a-final`
    showed no raw OSM connector ids in the banner/panel, 9 active direction
    arrows, cue 08 visible as arrival context, and the cue/map mileage split.
  - Follow-up tests for the navigation-source hold and held-route live-map
    behavior passed:
    `pytest -q years/2026/tests/test_export_mobile_field_packet.py -k "navigation_source_anchor_mismatch or live_map_declares_special_management_held_route_warning or map_leg_banner or active_cue_leg_navigation_artifact"`
    passed 3 tests with 64 deselected in 3.76s.
  - After regeneration, JSON validation, generated JS syntax validation, and
    `git diff --check` passed again. Browser verification on
    `http://127.0.0.1:8766/live-map.html?outing=112-1&cue=9&v=20260522-fd12a-hold2`
    showed `[HELD] FD12A`, disabled `Route held` GPS action, and no active cue
    banner.
- Current blocker:
  - FD12A remains a source-route repair item, not a runnable route. The
    canonical route needs redesign/splitting/recertification so the car-to-car
    route, cue sequence, GPX geometry, and field mileage describe the same
    human-valid outing.

## 2026-05-22 - FD12A source-route repair

- Supersession note: the later "Route-source regeneration correction" section
  below supersedes this intermediate result. The FD12 split described here did
  not survive source-safe regeneration and is not current field truth.
- Objective: fix the FD12A source defect instead of leaving the route held in
  the live map. Field review showed the selected Field Day 12 hybrid loop was a
  collapsed source route that joined West Climb/Full Sail work to the Harrison
  Hollow/Who Now work as one car-to-car outing.
- Result:
  - Follow-up correction: cue confusion, active-leg display trouble, or
    mid-route car-pass ambiguity is not a valid reason to collapse or split a
    route. The source decision has to be real on-foot effort, p75/runnable
    cost, legal access, and field-valid cues/GPX. If a one-route candidate is
    the best on-foot effort, the fix is to produce clean cues and a known
    timing estimate, not to split it away.
  - Current FD12 source evidence keeps the split on runnable cost, not on cue
    convenience: the repaired FD12A/FD12B pair is 10.84 on-foot miles and 214
    total p75 minutes, while rebuilding the old one-route trail set from the
    current graph gives 13.16 on-foot miles / 287 total minutes and still carries
    connector-validation flags.
  - `promote_field_day_loops.py` now expands selected field-day loops through
    accepted field-menu replacement components only when those components
    exactly partition the selected loop's official segment set and the
    replacement manifest is scoped to that selected candidate.
  - `promote_field_day_loops.py` now recovers from half-regenerated field
    packets by preserving certified source cards whose GPX file was deleted by
    a failed export when the canonical map route feature is still present; the
    final packet export must still recreate and validate the GPX before the
    route is field-ready.
  - `field_route_walkthrough_audit.py` now caches normalized preferred cue text
    during edge matching so full packet regeneration no longer stalls while
    repeatedly normalizing the same route text.
  - Field Day 12 now exports as two route cards from source:
    `112-1 FD12A` for Bob Smylie / Buena Vista / Full Sail from West Climb,
    and `112-2 FD12B` for Who Now Loop / Harrison Ridge / Harrison Hollow /
    Kemper's Ridge / Hippie Shake from Harrison Hollow.
  - The old collapsed candidate
    `combo-who-now-loop-trail-harrison-ridge-harrison-hollow-kempers-ridge-trail-full-sail-trail-buena-vista-trail-bob-smylie-hippie-shake-trail`
    no longer appears as the active Field Day 12 navigation unit.
  - Intermediate packet state from this pass exported 45 runnable route cards,
    40 field-ready route cards, 5 held route cards, and all 251 official
    segments. At this intermediate point, FD12A and FD12B passed route
    validation and had no navigation-source blockers.
- Validation:
  - `pytest -q years/2026/tests/test_promote_field_day_loops.py` passed 16
    tests.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py` passed 67
    tests.
  - Targeted exporter regression slice passed 10 tests with 57 deselected:
    `pytest -q years/2026/tests/test_export_mobile_field_packet.py -k "navigation_source_anchor_mismatch or special_management_failures_hold_route_card or render_card_marks_special_management_failure or live_map_declares_special_management_held_route_warning or active_cue_leg_navigation_artifact or previous_cue_context or active_leg_direction_arrows or signposts_over_generic_osm_connector_ids or field_packet_surfaces_r2r_signpost_cues or splits_backtracking_cue_legs"`.
  - Regenerated the canonical source via `human_loop_plan.py`,
    `export_mobile_field_packet.py`, `promote_field_day_loops.py`,
    `promote_harlow_h1_route_card.py`, then final
    `export_mobile_field_packet.py`; the final export wrote 135 GPX files and
    completed certification.
  - JSON validation passed for `docs/field-packet/field-tool-data.json`,
    `docs/field-packet/manifest.json`,
    `years/2026/outputs/private/2026-outing-menu-map-data.json`, and
    `years/2026/checkpoints/field-day-loop-promotion-2026-05-11.json`.
  - Generated live-map JavaScript parsed successfully with Node, and
    `git diff --check` passed.
  - Browser verification:
    `http://127.0.0.1:8766/live-map.html?outing=112-1&v=20260522-fd12a-sourcefix`
    selects `Hillside to Hollow: Bob Smylie (FD12A) - West Climb` and starts
    on `#55 West Climb`; `outing=112-2` selects `Hillside to Hollow: Who Now
    Loop (FD12B) - Harrison Hollow`. Neither route shows the old
    `blocked_navigation_source` warning.
- Superseded blocker note:
  - This section's conclusion that FD12A was repaired is no longer current.
    Later source-safe regeneration restored FD12A as a held
    `blocked_navigation_source` route until a package-112-scoped replacement
    source exists and is certified.

## 2026-05-22 - Route and planning logic error register

- Objective: consolidate the known route/planning logic errors into one
  public-safe register instead of leaving them spread across field-test notes,
  checkpoint reports, memory, and generated packet summaries.
- Result:
  - Added
    `years/2026/checkpoints/route-planning-logic-error-register-2026-05-22.md`.
  - Initial register snapshot: 45 runnable route cards, 40 field-ready cards, 5
    held cards, and 251/251 official segment coverage. The later
    route-source regeneration correction below supersedes these counts.
  - Initial held-card logic blockers were `FD04A`, `FD15A` / former route `3`,
    `FD18A`, `FD23A` / route `12`, and `FD26A`; the final held set adds
    `FD19C` and `FD12A`.
  - Added canonical BTC cases, failure modes, and behavior evals for the source-route
    `blocked_navigation_source` class and the H1-style one-route-truth /
    certified-replacement preservation class.
- Validation:
  - Documentation-only update. No planner, exporter, or pytest run was needed
    for the register itself; run the normal field-packet certification chain
    before clearing any route blocker named in the register.

## 2026-05-22 - Route-source regeneration correction

- Objective: finish the fixable logic-error work by making the generated packet
  match current evidence, not the earlier intermediate FD12 split and stale
  segment-ownership promotions.
- Result:
  - Retracted six unsupported active segment-ownership promotions in
    `years/2026/inputs/personal/2026-cross-package-segment-promotions-v1.json`.
    Only Shingle `1656` remains `promoted`.
  - Quick Draw `1610`, Highlands `1576`/`1577`, and Shane's
    `1649`/`1650`/`1651` are restored to source cards because their current
    promotion evidence did not survive regeneration/evidence-gate checks.
  - `FD04A` no longer claims Shane's segments; `FD19C` is restored as a held
    Shane's route card.
  - `FD23A` no longer owns Highlands segments, but the remaining Upper 8th /
    Corrals / Sidewinder source is still held by a parked-car-pass
    navigation-source mismatch.
  - The earlier FD12 split is not current field truth. Source-safe
    regeneration restored `FD12A` as the combined Who Now / Harrison / Full
    Sail / Bob Smylie card, and it remains held as `blocked_navigation_source`
    until a package-112-scoped replacement source is produced and certified.
  - Added a reusable generator guard in `promote_field_day_loops.py` so blocked
    canonical field-menu loops can opt into a stored personal/hybrid candidate
    source only when the source exists, and restored route labels are de-duped
    inside the field-day packet.
  - Refreshed the public/example outing menu artifacts after the source
    regeneration so the public Markdown and embedded map data match the current
    private canonical map data.
  - Refreshed `all-route-adversarial-disproof-2026-05-16` to match the current
    46-card route set after restored Quick Draw, Shane's, and Highlands cards.
  - The final packet exports 46 runnable route cards, 39 field-ready route
    cards, 7 held route cards, and all 251 official segments.
  - Current held cards are `FD19C`, `FD04A`, `FD12A`, `FD15A`, `FD18A`,
    `FD23A`, and `FD26A`.
- Validation:
  - `python years/2026/scripts/multi_start_field_menu_replacements.py` completed
    and wrote the private replacement manifest with 4 multi-start replacements
    across packages `1`, `4`, `5`, and `15`.
  - `python years/2026/scripts/promote_field_day_loops.py` completed and wrote
    the regenerated field-day route-card source.
  - `python years/2026/scripts/promote_harlow_h1_route_card.py` completed and
    preserved H1 in the regenerated route-card source.
  - `python years/2026/scripts/export_mobile_field_packet.py` completed and
    wrote 138 GPX files.
  - `jq empty years/2026/inputs/personal/2026-cross-package-segment-promotions-v1.json years/2026/inputs/personal/private/2026-field-menu-replacements-v2-multi-start.private.json years/2026/outputs/private/2026-outing-menu-map-data.json docs/field-packet/field-tool-data.json docs/field-packet/manifest.json`
    passed.
  - `pytest -q years/2026/tests/test_promote_field_day_loops.py` passed 18 tests
    in 0.35s.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py::test_navigation_source_anchor_mismatch_holds_route_card_and_field_tool_record years/2026/tests/test_export_mobile_field_packet.py::test_special_management_failures_hold_route_card_and_field_tool_record years/2026/tests/test_special_management_rule_audit.py`
    passed 6 tests in 2.38s.
  - Active promotion evidence check returned no failing promoted rows.
  - Duplicate route-label check over `docs/field-packet/field-tool-data.json`
    returned 0 duplicates.
  - `python years/2026/scripts/export_example_map.py` completed and wrote the
    public/example map data, map HTML, menu Markdown, and PNG.
  - `pytest -q years/2026/tests/test_all_route_adversarial_disproof.py years/2026/tests/test_fd04a_fd19c_credit_promotion_experiment.py years/2026/tests/test_public_map_artifact_consistency.py`
    passed 12 tests in 0.55s after the disproof/public-artifact corrections.
  - `pytest -q` passed 582 tests in 140.04s.
  - `git diff --check` passed after the documentation update.
- Current blocker:
  - The remaining 7 held route cards need route redesign, fresher management
    evidence, or a new package-scoped replacement source. They are no longer
    fixable by trusting stale ownership rows or by changing only the live-map
    display.

## 2026-05-23 - Logic-error register audit refresh

- Objective: refresh the route/planning logic-error register against the
  current generated packet and standalone certification-support audits.
- Result:
  - Regenerated the special-management checkpoint from current packet truth. It
    now reports 46 routes, 5 failed routes, and current labels `FD19C`,
    `FD04A`, `FD15A`, `FD18A`, and `FD26A`; the previous checkpoint artifact
    still showed 43 routes, 4 failed routes, and old labels from before source
    regeneration.
  - Regenerated progress, recertification, latent-credit, walkthrough, and
    field-tool completion reports.
  - Updated
    `years/2026/checkpoints/route-planning-logic-error-register-2026-05-22.md`
    with the refreshed audit status and two planning-proof guards:
    standalone checkpoint drift after source regeneration, and future-progress
    preservation after completed/missed/blocked segment updates.
  - Current generated packet remains 46 route cards, 39 field-ready cards, 7
    held cards, and 251 / 251 official segments represented.
- Validation:
  - `python years/2026/scripts/special_management_rule_audit.py` wrote updated
    artifacts and returned nonzero because it found the expected 5
    special-management failed routes.
  - `python years/2026/scripts/field_progress_report.py` passed and preserved
    251 / 251 remaining segments with 0 completed, missed, or blocked segments.
  - `python years/2026/scripts/field_recertification_report.py` passed with
    remaining full completion feasible from the active menu.
  - `python years/2026/scripts/field_latent_credit_audit.py --report-only`
    passed with 46 routes, 0 routes needing repair, and 39 latent segments
    reconciled to other active route cards.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed 46 / 46
    routes.
  - `python years/2026/scripts/field_tool_completion_audit.py` wrote refreshed
    artifacts and returned nonzero because land-manager special-management
    compliance still fails; 15 / 16 requirements passed.
- Current blocker:
  - Documentation now matches the known logic-error set, but route
    certification is still blocked by the 7 held cards: `FD19C`, `FD04A`,
    `FD12A`, `FD15A`, `FD18A`, `FD23A`, and `FD26A`.

## 2026-05-23 - Field packet fail-closed repair and recertification

- Objective: finish the route-source repairs instead of publishing held or
  warning-only phone routes. Normal field-packet export must fail if any route
  is not field-ready.
- Result:
  - Changed the mobile packet exporter to certify the manifest before writing
    the normal packet, omit non-field-ready diagnostic routes from field-tool
    data, and raise on any route card that is not certifiable.
  - Repaired the current source chain instead of relying on browser warnings:
    package 112 is now backed by the accepted two-card West Climb and Harrison
    source repair, package 123 is backed by current 8th Street/Corrals/
    Sidewinder source cards, FD18A Polecat ordering follows the
    special-management clockwise rule, stale unsafe Veterans connector links are
    replaced from the current foot-safe connector graph, and FD01A car-pass
    wayfinding is split at the actual parked-car pass.
  - Promoted the accepted FD14D, FD09A, and FD03A replacements as certified
    route cards and regenerated the public/example map artifacts from the same
    source.
  - Refreshed the all-route adversarial disproof registry from the current
    49-card packet so route-efficiency and repeat-optimization proof rows match
    current candidate ids.
  - Removed the generated source-route mismatch warning copy from the live-map
    template. The browser packet now has 49 route cards, all 49 are
    `field_ready`, 0 held routes, 0 omitted non-field-ready routes, certified
    field-day publication status, and 251 / 251 official segments covered.
- Validation:
  - `python3 years/2026/scripts/promote_field_day_loops.py` passed and reported
    31 packages, 53 component routes, 251 covered official segments, and 0
    source-gap warnings.
  - `python3 years/2026/scripts/promote_harlow_h1_route_card.py` passed and
    reduced the active route-card set from 53 to 49.
  - `python3 years/2026/scripts/export_mobile_field_packet.py` passed and
    wrote 147 GPX files plus the regenerated phone packet.
  - Generated packet scan found no `NOT FIELD READY`, held-route, provisional,
    or blocked route strings in `docs/field-packet/index.html`,
    `docs/field-packet/live-map.html`, `docs/field-packet/field-tool-data.json`,
    or `docs/field-packet/manifest.json`.
  - `python3 -m json.tool docs/field-packet/field-tool-data.json` passed.
  - `python3 -m json.tool docs/field-packet/manifest.json` passed.
  - Packet summary check passed: 49 / 49 `field_ready`, 0 navigation-source
    failures, 0 not-ready field-tool records, field-day baseline `passed`, 251
    covered official segments, and 0 missing segments.
  - `pytest -q years/2026/tests/test_export_execution_gpx.py years/2026/tests/test_promote_field_day_loops.py years/2026/tests/test_export_mobile_field_packet.py years/2026/tests/test_export_field_day_layer.py years/2026/tests/test_accepted_anchor_preservation_audit.py years/2026/tests/test_public_map_artifact_consistency.py -q`
    passed 51 tests.
  - `pytest -q years/2026/tests/test_all_route_adversarial_disproof.py years/2026/tests/test_fd04a_fd19c_credit_promotion_experiment.py years/2026/tests/test_special_management_rule_audit.py -q`
    passed 14 tests after refreshing the current-route proof artifact and
    special-management expectations.
  - `pytest -q` passed 589 tests in 140.67s.
  - `python3 years/2026/scripts/field_tool_completion_audit.py` passed 16 / 16
    requirements with 49 field-ready routes and 0 held routes.
  - `python3 years/2026/scripts/special_management_rule_audit.py` passed 49 /
    49 routes.
  - `python3 years/2026/scripts/accepted_anchor_preservation_audit.py` passed
    with 3 manifest records and 0 failures.
  - `python3 years/2026/scripts/field_official_repeat_audit.py --map-data-json years/2026/outputs/private/2026-outing-menu-map-data.json`
    passed with 0 bad hidden repeats and 0 unreconciled extra-credit segments.
  - `python3 years/2026/scripts/field_route_walkthrough_audit.py` passed 49 /
    49 routes.
  - `python3 years/2026/scripts/field_latent_credit_audit.py` passed with 0
    missing GPX files, 0 routes needing repair, and all latent credit
    reconciled.
  - `python3 years/2026/scripts/export_example_map.py` completed and refreshed
    the public/example map data, map HTML, menu Markdown, and PNG.
- Current blocker:
  - No known packet/source certification blocker remains in the generated field
    packet. Standard same-day condition and closure checks still apply before
    running any route.

## 2026-06-20 - Remove route holds after dashboard progress sync

- Objective: complete the BTC dashboard progress sync without leaving manual
  route-source holds in the live phone packet.
- Result:
  - Repaired route-card mileage anchoring so displayed cue mileage follows the
    GPX route interval while preserving source mileage for audit context.
  - Removed the route-truth manual hold area from the active manual design
    source and regenerated the canonical map, phone packet, GPX exports, and
    audit checkpoints.
  - Field packet now has 28 runnable / field-ready routes, 0 manual holds, 0
    held routes, 0 omitted routes, and 84 GPX files.
  - Same-anchor spur audit now skips non-runnable-cost absorptions when the host
    route is already over the p90 bound.
- Validation:
  - `python3 years/2026/scripts/human_loop_plan.py` passed.
  - `python3 years/2026/scripts/export_mobile_field_packet.py` passed.
  - `python3 years/2026/scripts/field_latent_credit_audit.py` passed.
  - `python3 years/2026/scripts/field_progress_report.py` passed with 19
    completed, 231 remaining, and 0 held remaining segments.
  - `python3 years/2026/scripts/field_recertification_report.py` passed.
  - `python3 years/2026/scripts/same_anchor_spur_split_audit.py` passed with 0
    findings and 0 advisories.
  - `python3 years/2026/scripts/route_edge_cover_audit.py` passed.
  - `python3 years/2026/scripts/field_tool_completion_audit.py` passed 21 / 21
    requirements.
  - `python3 years/2026/scripts/field_route_walkthrough_audit.py` passed 28 /
    28 routes.
  - `python3 years/2026/scripts/post_credit_connector_audit.py` passed.
  - `python3 -m pytest -q years/2026/tests/test_export_mobile_field_packet.py::test_official_wayfinding_boundary_uses_next_connector_first_source_pass years/2026/tests/test_export_mobile_field_packet.py::test_wayfinding_display_miles_follow_route_intervals_and_preserve_source years/2026/tests/test_same_anchor_spur_split_audit.py` passed 9 tests.
  - `git diff --check` passed.
  - Changed JSON files parsed successfully.
- Current blocker:
  - No route-card hold or certification blocker remains. The old dated field-day
    layer was not promoted; `python3 years/2026/scripts/export_field_day_layer.py`
    still fails on a 2026-07-04 p90-bound violation in the historical assignment.

## 2026-06-20 - Challenge epoch progress reset

- Objective: reset any pre-challenge or provisional completion state now that
  the 2026 challenge window has started.
- Result:
  - Ran the segment-first epoch reset for `challenge-2026`, clearing
    `completed_segment_ids`, `blocked_segment_ids`, and `blocked_trail_names`
    from the private planner state and removing any challenge-epoch ledger
    events.
  - Locked the clean `challenge-2026` original baseline under
    `years/2026/outputs/private/progress/versions/challenge-2026/original/`.
  - Confirmed the private ledger has 0 events and 0 `challenge-2026` events.
- Validation:
  - `python years/2026/scripts/field_progress_versions.py reset-epoch --epoch challenge-2026 --clear-blocks`
    passed.
  - `python years/2026/scripts/field_progress_report.py` passed and reported 0
    completed, provisional, missed, and blocked segments.
  - `python years/2026/scripts/field_recertification_report.py --skip-heavy-optimizer`
    passed and reported 0 completed/provisional/blocked segments.
- Current blocker:
  - Progress is clean, but the current field-ready packet still only exposes
    143 of 250 official segments because 11 route cards remain held for
    route-truth repair.

## 2026-06-19 - Shingle / Dry Creek route-truth repair

- Objective: explain and repair the field-packet confusion where `16C-2`
  repeats Shingle Creek while Dry Creek credit is assigned to `15B`.
- Result:
  - Confirmed the active route is a planner artifact, not a field insight:
    the natural field shape is lower Dry Creek access, Shingle Creek ascent,
    then Dry Creek descent back to the car.
  - Added a hard completion-audit gate for per-cue mismatch between written cue
    mileage and live-map cue span. The current packet fails this gate with 21
    mismatched movement legs across 11 route cards, including `16C-2`.
  - Extended the exporter `blocked_navigation_source` guard so large cue/map
    mileage drift holds a route card before publication.
  - Retracted the tracked Shingle `1656` ownership assumption that referred to
    stale `15A-1` evidence, and added an explicit Dry Creek / Shingle natural
    loop design target.
  - Added `years/2026/inputs/personal/2026-route-truth-repairs-v1.json` and
    wired `human_loop_plan.py` to apply route-truth repairs after manual route
    promotion. The active private map source now assigns Dry Creek `1542-1546`
    and Shingle `1656` to `16A-D1`, with lower Dry repeated as the intentional
    lollipop stem; `15B` is pruned to Highlands / Connector only.
  - Added an explicit route-truth mismatch hold area for the 11 other active
    cards whose cue mileage and live-map/GPX spans materially disagreed:
    `1B`, `15B`, `18B`, `16C-1`, `3`, `12`, `16A-1`, `18A`, `10A`, `6`,
    and `13`. These cards are now manual holds instead of runnable field
    artifacts.
- Validation:
  - `python3 -m json.tool years/2026/inputs/personal/2026-manual-route-designs-v1.json`
    passed.
  - `python3 -m json.tool years/2026/inputs/personal/2026-cross-package-segment-promotions-v1.json`
    passed.
  - `python3 -m json.tool years/2026/inputs/personal/2026-route-truth-repairs-v1.json`
    passed.
  - `pytest -q years/2026/tests/test_field_tool_completion_audit.py years/2026/tests/test_export_mobile_field_packet.py::test_navigation_source_anchor_mismatch_holds_route_card_and_field_tool_record years/2026/tests/test_export_mobile_field_packet.py::test_navigation_source_cue_map_mileage_mismatch_holds_route_card`
    passed 29 tests.
  - `python3 years/2026/scripts/human_loop_plan.py` regenerated the active
    private outing menu/map source with `16A-D1`.
  - `python3 years/2026/scripts/export_mobile_field_packet.py` passed and
    regenerated `docs/field-packet/` with 20 field-ready routes and 11 manual
    holds.
  - `python3 years/2026/scripts/field_tool_completion_audit.py` passed 21/21
    hard requirements: 250/250 official segments accounted, 0 cue/map mismatch
    failures, 0 canonical route metric failures, and 0 special-management
    failures.
- Current blocker:
  - The packet is publishable as a smaller field-ready set with explicit holds:
    the held cards still need future route-truth repair before they can return
    as runnable route cards, but they are no longer planner artifacts in the
    field-facing packet.

## 2026-05-27 - Post-credit connector savings audit and repair pass

- Objective: treat the missed West Climb / Kemper connector saving as a class
  failure, not a one-off mileage tweak.
- Result:
  - Added generator and audit behavior so post-credit connector, repeat, exit,
    and return cues are checked against the shortest legal connector graph path.
  - Repaired source selection so stale route-feature tracks do not beat safer
    cue-derived tracks when they use blocked/bike-only connector geometry.
  - Repaired connector cue stitching so regenerated `between_links` beat stale
    per-segment `pre_connector_link` copies.
  - The post-credit connector audit now passes with 103 connector proofs, 0
    shorter-connector findings, 0 unproved-connector findings, and 0 route-card
    / GPX mismatches.
- Validation:
  - `python years/2026/scripts/export_mobile_field_packet.py` passed and
    regenerated the field packet / GPX artifacts.
  - `pytest -q years/2026/tests/test_post_credit_connector_audit.py
    years/2026/tests/test_export_mobile_field_packet.py::test_track_segments_for_route_cues_prefers_repaired_between_link_over_stale_prelink
    years/2026/tests/test_export_mobile_field_packet.py::test_track_selection_prefers_foot_legal_cue_track_over_blocked_feature_track
    years/2026/tests/test_export_mobile_field_packet.py::test_shortest_connector_repair_replaces_unsafe_path_even_when_legal_path_is_longer
    years/2026/tests/test_export_mobile_field_packet.py::test_unsafe_connector_labels_are_removed_without_path_repair
    years/2026/tests/test_export_mobile_field_packet.py::test_track_selection_prefers_special_management_legal_cue_track
    years/2026/tests/test_export_mobile_field_packet.py::test_stitch_inter_segment_track_gaps_splits_unstitched_internal_gap
    years/2026/tests/test_export_mobile_field_packet.py::test_link_for_group_transition_matches_trail_names_before_position
    years/2026/tests/test_export_execution_gpx.py
    years/2026/tests/test_multi_start_field_menu_replacements.py
    years/2026/tests/test_personal_route_planner.py` passed 96 tests.
  - `python years/2026/scripts/post_credit_connector_audit.py
    --field-tool-data-json docs/field-packet/field-tool-data.json
    --packet-dir docs/field-packet
    --output-json years/2026/checkpoints/post-credit-connector-audit-2026-05-27.json
    --output-md years/2026/checkpoints/post-credit-connector-audit-2026-05-27.md
    --manifest-json years/2026/checkpoints/post-credit-connector-audit-2026-05-27-manifest.json`
    passed.
  - `python years/2026/scripts/field_route_walkthrough_audit.py
    --field-tool-data-json docs/field-packet/field-tool-data.json
    --packet-dir docs/field-packet
    --output-json years/2026/checkpoints/field-route-walkthrough-audit-2026-05-27.json
    --output-md years/2026/checkpoints/field-route-walkthrough-audit-2026-05-27.md`
    still failed: 25 / 31 routes passed.
- Current blocker:
  - Do not publish this packet yet. Remaining walkthrough blockers are
    direction evidence on Hawkins and Around the Mountain, unmatched route
    geometry on 15B / 16C-2 / 18B, and two ambiguous service-connector decision
    points on 16C-1. The connector-savings class is fixed, but full packet
    certification is not clean.

## 2026-05-27 - Post-credit connector invariant follow-up

- Objective:
  - Treat the West Climb/Kemper connector miss as a route-packet invariant
    failure, not a one-off 70-foot savings issue.
- Result:
  - Added a post-credit connector audit for connector, repeat, exit, car-pass,
    and return cues after official credit has started.
  - Updated route-cue and wayfinding repair logic so stale scalar mileage
    cannot hide a shorter legal connector when source geometry is longer.
  - Preserved connector graph source geometry and avoided-unearned segment ids
    so shortest-path proof can use the actual legal connector line.
  - Optimized walkthrough edge-name enrichment to filter the graph to local
    route edges before preferred-text matching.
  - The current tracked field packet fails the new audit, confirming the issue
    is systemic: 49 / 49 routes failed with 76 shorter-connector findings and
    8 unproved connector findings.
  - A refreshed diagnostic packet with the merged repair logic reduced shorter
    legal connector findings to 0, but still has 15 unproved connector-proof
    blockers. The certifiable public export is also blocked by route-source
    issues outside this connector class: 5B and 17 special-management
    direction failures, plus 15A-1 and 15A-2 missing track/parking data.
- Validation:
  - `python -m py_compile years/2026/scripts/personal_route_planner.py years/2026/scripts/export_execution_gpx.py years/2026/scripts/block_day_packager.py years/2026/scripts/export_mobile_field_packet.py years/2026/scripts/post_credit_connector_audit.py`
    passed.
  - `pytest -q years/2026/tests/test_export_execution_gpx.py years/2026/tests/test_personal_route_planner.py years/2026/tests/test_export_mobile_field_packet.py -k 'custom_traversal_geometry or declared_inter_segment_link or shortest_connector_path_preserves_source_edge_geometry or shortest_connector_path_reports_connector_edge_classes or shortest_connector_path_can_avoid_required_official_repeat_edges or track_segments_for_route_cues_follow_cue_source_order or select_track_segments or apply_shortest_connector_repairs or return_wayfinding_cue_uses_total_repaired_distance or apply_shortest_repairs_to_wayfinding_cues or turn_step_sync or live_gps_map_uses_consistent_active_leg_direction_arrows'`
    passed 16 tests.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py -k 'prices_stale_source_mileage_from_geometry or apply_shortest_repairs_to_wayfinding_cues'`
    passed 3 tests.
  - `python years/2026/scripts/multi_start_field_menu_replacements.py` passed.
  - `python years/2026/scripts/human_loop_plan.py` passed.
  - `python years/2026/scripts/export_mobile_field_packet.py` failed certification as expected on the route-source blockers above.
  - `python years/2026/scripts/export_mobile_field_packet.py --allow-uncertified --output-dir /tmp/btc-post-credit-rebase` wrote a diagnostic packet.
  - `python years/2026/scripts/post_credit_connector_audit.py --field-tool-data-json /tmp/btc-post-credit-rebase/field-tool-data.json --packet-dir /tmp/btc-post-credit-rebase --output-json years/2026/checkpoints/post-credit-connector-audit-2026-05-27.json --output-md years/2026/checkpoints/post-credit-connector-audit-2026-05-27.md --manifest-json years/2026/checkpoints/post-credit-connector-audit-2026-05-27-manifest.json --report-only`
    completed with 0 shorter-connector findings and 15 unproved connector
    findings.
- Current blocker:
  - Do not publish a regenerated field packet yet. Repair or replace the
    special-management/missing-parking route sources, then resolve the 15
    remaining unproved connector-proof findings and rerun the full certification
    chain.

## 2026-05-26 - Closed edge-cover guard and FD12A source repair

- Objective: prevent one-car route cards from being generated as trailhead-
  anchored segment-cluster excursions, and repair `FD12A` from the upstream
  route source instead of the generated packet.
- Result:
  - Added the closed edge-cover route-card doctrine to `AGENTS.md` and the BTC
    heuristic/failure/eval docs.
  - Added `route_edge_cover_audit.py` and wired its hard gate into
    `field_tool_completion_audit.py`.
  - Repaired the private `FD12A` replacement source for
    `combo-full-sail-trail-buena-vista-trail-bob-smylie` as a West Climb
    closed walk: Bob Smylie, Buena Vista spur clear, Buena Vista descent,
    Full Sail spur clear, and return to West Climb.
  - Repaired the route-source/exporter issues the new gate exposed before
    shipping the packet: `FD19C` now uses the official Shane's geometry,
    `FD23C` clears Corrals without the prior long post-credit repeat, and the
    automatic repeat repair no longer rewrites necessary out-and-backs such as
    `16B`.
  - Regenerated the promoted source and phone packet. `FD12A` now exports at
    4.86 on-foot miles, p75 126 minutes, p90 142 minutes, covers
    `1504,1505,1506,1507,1565,1566,1718,1719,1755`, and has no depot
    phase-reset failure in the new audit.
- Validation:
  - `python years/2026/scripts/export_mobile_field_packet.py` passed and wrote
    147 GPX files plus the regenerated phone packet.
  - `python years/2026/scripts/route_edge_cover_audit.py` passed with 49
    routes, 0 hard failures, and 2 disconnected-component phase-reset
    advisories. `FD12A` passed with generated 4.86 mi / lower bound 3.91 mi /
    efficiency 1.241.
  - `python years/2026/scripts/field_latent_credit_audit.py` passed with 49
    routes and no routes needing repair.
  - `python years/2026/scripts/field_progress_report.py` passed with all 251
    official segments remaining available.
  - `python years/2026/scripts/field_recertification_report.py` passed with
    remaining full completion feasible.
  - `python years/2026/scripts/field_route_walkthrough_audit.py` passed with
    49 / 49 routes passing.
  - `python years/2026/scripts/route_repeat_optimization_audit.py` passed with
    0 hidden self-repeat blockers and 0 avoidable post-credit repeat blockers.
  - `python years/2026/scripts/field_official_repeat_audit.py` passed with 0
    bad hidden self-repeat labels and 0 unreconciled extra-credit segments.
  - `python years/2026/scripts/field_tool_completion_audit.py` passed all 17
    requirements with 49 / 49 field-ready routes and 251 / 251 accounted
    official segments.
  - `pytest -q years/2026/tests/test_route_edge_cover_audit.py
    years/2026/tests/test_route_repeat_optimization_audit.py
    years/2026/tests/test_field_tool_completion_audit.py` passed 38 tests.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py` passed 72
    tests.
- Current blocker: none for this repair; standard day-of closure, condition,
  heat, and access checks still apply before field use.

## 2026-05-25 - FD12A Full Sail route-truth drift

- Objective: classify and guard the FD12A failure where the live map sent the
  runner through a repeat/out-and-back that consumed a short field window and
  left at least one segment uncompleted.
- Result:
  - Classified the event as route artifact/source drift plus plan-repair
    feedback, not progress. Activity geometry still needs validation before
    any completed, missed, or partial segment state is applied.
  - Found that `112-1` / `FD12A` had 4.47 mi route-card and scaled phone cue
    mileage, but about 5.90 mi of live-map route-anchor traversal and about
    7.26 mi measured from the exported Nav GPX.
  - Added a generic field-tool completion audit guard so live-map
    `route_miles` / `route_leg_miles` anchors must reconcile with route-card
    on-foot mileage. This catches hidden repeat distance even when the scaled
    phone cue labels match and repeated official mileage is declared as
    no-new-credit.
  - Added public-safe field-test and heuristic/eval documentation for the
    failure class.
- Validation:
  - `pytest -q years/2026/tests/test_field_tool_completion_audit.py` passed
    19 tests.
  - `python3 -m json.tool docs/field-packet/field-tool-data.json` passed.
  - `python3 -m json.tool docs/field-packet/manifest.json` passed.
  - `python3 years/2026/scripts/field_tool_completion_audit.py` wrote the
    canonical checkpoint and failed as expected with 15 / 16 requirements
    passing. The new blocker is live-map route-anchor mileage disagreement,
    including `FD12A West Climb` at about 5.90 mi of route-anchor traversal
    versus a 4.47 mi route card.
  - A direct `route_field_failures(...)` check found 29 current route-anchor
    mileage failures, so this is a packet/source class, not a one-card display
    issue.
- Current blocker:
  - Need the actual activity geometry review to identify the left/missed
    segment(s) and apply segment-first progress repair. The active packet also
    needs route-source repair and recertification before calling FD12A
    field-ready again.

## 2026-05-25 - Avoidable post-credit repeat repair and recertification

- Objective: finish the repair pass after the PR1 hard gate exposed avoidable
  post-credit repeats, regenerate the field packet, and return the packet to a
  field-ready audit state.
- Result:
  - Tightened the avoidable-repeat proof so graph-scaled or official mileage is
    provenance only; hard failures now require actual cue-to-cue replacement
    geometry to be materially shorter than the current physical cue interval.
  - Fixed the exporter repair pass to evaluate only repeat ids the cue interval
    actually completes, to preserve claimed endpoint coverage, and to avoid
    mutating `_track_segments` when a candidate replacement is rejected.
  - Regenerated the packet. The remaining true hard failure was `109-2: 10B`
    cue 5; it now records one avoidable post-credit repeat repair for segment
    `1497`, drops route-card on-foot mileage from about 7.61 to 6.26 miles,
    and reconciles GPX, live-map route anchors, and field data.
- Validation:
  - `time python3 years/2026/scripts/export_mobile_field_packet.py` completed
    successfully and wrote 147 GPX files plus the field packet HTML/manifest.
  - GPX file-set settle check confirmed 49 official, 49 audit, and 49 cue GPX
    files; `docs/field-packet/gpx/all-field-packet-gpx.zip` exists.
  - `python3 years/2026/scripts/route_repeat_optimization_audit.py` passed:
    49 routes, 0 failed routes, 0 avoidable post-credit repeat instances, and
    45 no-alternate advisories.
  - `python3 years/2026/scripts/field_tool_completion_audit.py` passed 16 / 16
    requirements with 49 field-ready route cards and 251 / 251 official
    segments accounted.
  - `python3 years/2026/scripts/repeat_productivity_audit.py` exited 0 with
    `dead_repeat_candidates_found`, preserving non-blocking repeat-productivity
    review output.
  - `pytest -q years/2026/tests/test_personal_route_planner.py -k "shortest_connector_path"` passed 5 tests, 30 deselected.
  - `pytest -q years/2026/tests/test_route_repeat_optimization_audit.py years/2026/tests/test_repeat_productivity_audit.py years/2026/tests/test_field_tool_completion_audit.py` passed 37 tests.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py -k "rejected_avoidable_repeat_repair or non_credit_claimed_repeat or wayfinding_cue_mileage_reconciles_to_route_card or prices_route_card"` passed 4 tests, 68 deselected.
- Current blocker: none for the avoidable post-credit repeat class. Private
  2026-05-25 activity progress remains unapplied until activity geometry is
  validated segment-first.

## 2026-05-25 - Avoidable post-credit repeat hard gate

- Objective: implement the PR1 detection-only guard for routes that earn
  official credit and then re-run the same already-credited section as
  no-new-credit movement when the connector graph proves a materially shorter
  legal path.
- Result:
  - Added cue-level avoidable post-credit repeat detection to the route-repeat
    optimization audit. The gate tracks completed-at-export ids plus prior cue
    credit, searches for connector alternatives that avoid the repeated ids,
    hard-fails proven avoidable repeats, and preserves no-alternate cases as
    route-source review advisories.
  - Wired field-tool completion to fail on the new hard gate and repeat
    productivity to classify proven avoidable repeats as dead-repeat
    candidates instead of necessary return mileage.
  - Added FD12A-shaped regression coverage and updated the connector
    provenance docs/eval without creating a duplicate failure family.
- Validation:
  - `pytest -q years/2026/tests/test_personal_route_planner.py -k "shortest_connector_path"`
    passed 3 tests, 30 deselected.
  - `pytest -q years/2026/tests/test_route_repeat_optimization_audit.py years/2026/tests/test_repeat_productivity_audit.py years/2026/tests/test_field_tool_completion_audit.py`
    passed 36 tests.
  - `python3 years/2026/scripts/route_repeat_optimization_audit.py` wrote the
    canonical checkpoint and failed as expected: 37 failed routes, 54
    avoidable post-credit repeat instances, 155 affected repeated segment ids,
    and 10 no-alternate advisories.
  - `python3 years/2026/scripts/repeat_productivity_audit.py` passed and wrote
    the canonical checkpoint with `dead_repeat_candidates_found`: 39 routes
    with dead-repeat candidates and 14.32 actual route miles of dead-repeat
    exposure.
  - `python3 years/2026/scripts/field_tool_completion_audit.py` wrote the
    canonical checkpoint and failed as expected with 14 / 16 requirements
    passing. Remaining blockers are route-anchor mileage disagreement and the
    new avoidable post-credit repeat hard gate.
  - JSON validation passed for route-repeat, repeat-productivity, and
    field-tool completion checkpoint JSON plus route/repeat manifest JSON.
  - `git diff --check` passed.
- Current blocker:
  - PR2 must repair route sources or accepted replacement/promotion inputs
    before the packet can return to field-ready. This pass intentionally did
    not apply private activity progress, repair FD12A, or mass-edit generated
    route sources.

## 2026-06-13 - Official segment data refresh

- Objective: refresh current official Boise Trails Challenge segment data before
  challenge start and make future route-list drift checks repeatable.
- Result:
  - Added `years/2026/scripts/pull_official_challenge_data.py` to pull public
    read-only `/api/trails` data, derive foot-only official files, save challenge
    metadata without raw leaderboard rows, and compare against the prior official
    pull.
  - Saved the latest public official pull under
    `years/2026/inputs/official/api-pull-2026-06-13/`.
  - The live trail payload reports `lastUpdatedUTC=2026-06-11T01:45:43`.
  - On-foot official data changed from 251 segments / 164.43 mi to 250 segments
    / 159.0 mi. Removed: Stack Rock Connector `1663` and `1664`. Added: Stack
    Rock Connector `1762`. Changed common segments: Polecat `1601` / `1603`
    geometry, Sweet Connie `1667` length/geometry, Around the Mountain `1750`
    activity type.
  - The active field packet is now stale against the latest official route list:
    route `16C-1` claims removed segments `1663` / `1664`, new official segment
    `1762` is unclaimed, and routes `5B`, `16A-1`, and `17` touch changed
    common segments.
- Validation:
  - `pytest -q years/2026/tests/test_pull_official_challenge_data.py` passed 2
    tests.
  - `python3 years/2026/scripts/pull_official_challenge_data.py --pull-date 2026-06-13 --trails-json /tmp/btc_trails_latest.json --leaderboard-json /tmp/btc_leaderboard_latest.json --compare-to years/2026/inputs/official/api-pull-2026-05-04 --field-tool-data docs/field-packet/field-tool-data.json` passed and wrote the new pull plus drift report.
- Current blocker:
  - This is a route-list change event. The canonical field packet should not be
    treated as current until affected route cards are repaired/regenerated and
    the field-packet certification chain is run against the June 13 official
    data.

## 2026-05-24 - Route-card-first phone packet

- Objective: make certified route cards the default phone packet view and keep
  Field Days as the secondary calendar/sequencing view.
- Result:
  - Updated the field-packet exporter, field-day layer generator, and packet
    doctrine so `route_cards` is the primary execution artifact and `routes` is
    the default phone view.
  - Regenerated the field-day checkpoint and public phone packet. The local
    browser now opens `docs/field-packet/index.html` with Route Cards first and
    active, while the Field Days tab remains available.
- Validation:
  - `python3 years/2026/scripts/export_field_day_layer.py` passed.
  - `python3 years/2026/scripts/export_mobile_field_packet.py` passed and wrote
    147 GPX files plus the regenerated phone packet.
  - `python3 -m json.tool docs/field-packet/field-tool-data.json` passed.
  - `python3 -m json.tool docs/field-packet/manifest.json` passed.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py years/2026/tests/test_export_field_day_layer.py` passed 83 tests.
  - `pytest -q` passed 589 tests in 139.26s.
  - `python3 years/2026/scripts/field_latent_credit_audit.py` passed with 49
    routes, 0 routes needing repair, and all latent credit reconciled.
  - `python3 years/2026/scripts/field_progress_report.py` passed with 251
    remaining available official segments and the original target still
    possible from the menu.
  - `python3 years/2026/scripts/field_recertification_report.py` passed with
    remaining full completion feasible.
  - `python3 years/2026/scripts/field_tool_completion_audit.py` passed 16 / 16
    requirements with 49 field-ready routes and 0 held routes.
  - `python3 years/2026/scripts/field_route_walkthrough_audit.py` passed 49 /
    49 routes.
- Current blocker:
  - No known packet/source certification blocker remains in the generated field
    packet. Standard same-day condition and closure checks still apply before
    running any route.

## 2026-05-24 - Collapsed route-card cue sheets

- Objective: reduce route-card scrolling cost by collapsing each `Field Cue
  Sheet` by default while preserving the full cue text inside the route card.
- Result:
  - Updated the field-packet exporter so every route card renders the cue sheet
    as a closed disclosure row with a cue count, rather than an always-expanded
    section.
  - Regenerated the public phone packet. Local browser verification showed 49
    route-card cue sheets, 0 open by default, and the first cue sheet expanding
    normally.
- Validation:
  - `python3 years/2026/scripts/export_mobile_field_packet.py` passed and wrote
    147 GPX files plus the regenerated phone packet.
  - `python3 -m json.tool docs/field-packet/field-tool-data.json` passed.
  - `python3 -m json.tool docs/field-packet/manifest.json` passed.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py` passed 69
    tests.
  - `pytest -q` passed 589 tests in 134.09s.
  - `python3 years/2026/scripts/field_latent_credit_audit.py` passed with 49
    routes, 0 routes needing repair, and all latent credit reconciled.
  - `python3 years/2026/scripts/field_progress_report.py` passed with 251
    remaining available official segments and the original target still
    possible from the menu.
  - `python3 years/2026/scripts/field_recertification_report.py` passed with
    remaining full completion feasible.
  - `python3 years/2026/scripts/field_tool_completion_audit.py` passed 16 / 16
    requirements with 49 field-ready routes and 0 held routes.
  - `python3 years/2026/scripts/field_route_walkthrough_audit.py` passed 49 /
    49 routes.
- Current blocker:
  - No known packet/source certification blocker remains in the generated field
    packet. Standard same-day condition and closure checks still apply before
    running any route.

## 2026-05-24 - Remove route-card audit sections from phone view

- Objective: keep route cards focused on field execution by removing visible
  audit/reconciliation sections such as `Cross-route segment ownership`.
- Result:
  - Removed the phone-visible cross-route ownership section from generated route
    cards.
  - Preserved `segment_ownership_reconciliation` in `field-tool-data.json` and
    `manifest.json` for audits; the generated packet still has 21 reconciled
    ownership records in JSON.
  - Regenerated the public phone packet. Local browser verification showed 49
    route cards, no `Cross-route segment ownership` or `planned owner` text,
    and no console warnings or errors.
- Validation:
  - `python3 years/2026/scripts/export_mobile_field_packet.py` passed and wrote
    147 GPX files plus the regenerated phone packet.
  - `python3 -m json.tool docs/field-packet/field-tool-data.json` passed.
  - `python3 -m json.tool docs/field-packet/manifest.json` passed.
  - `pytest -q years/2026/tests/test_export_mobile_field_packet.py` passed 69
    tests.
  - `pytest -q` passed 589 tests in 137.41s.
  - `python3 years/2026/scripts/field_latent_credit_audit.py` passed with 49
    routes, 0 routes needing repair, and all latent credit reconciled.
  - `python3 years/2026/scripts/field_progress_report.py` passed with 251
    remaining available official segments and the original target still
    possible from the menu.
  - `python3 years/2026/scripts/field_recertification_report.py` passed with
    remaining full completion feasible.
  - `python3 years/2026/scripts/field_tool_completion_audit.py` passed 16 / 16
    requirements with 49 field-ready routes and 0 held routes.
  - `python3 years/2026/scripts/field_route_walkthrough_audit.py` passed 49 /
    49 routes.
- Current blocker:
  - No known packet/source certification blocker remains in the generated field
    packet. Standard same-day condition and closure checks still apply before
    running any route.

# 2026 Daily Work Log

This is the short daily log for what we are trying, what changed, and what still
needs proof. It complements the longer planning decision log and the public
field-test logs.

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

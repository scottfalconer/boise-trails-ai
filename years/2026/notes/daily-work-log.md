# 2026 Daily Work Log

This is the short daily log for what we are trying, what changed, and what still
needs proof. It complements the longer planning decision log and the public
field-test logs.

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
- Reviewed `/Users/scott/Desktop/btc-2026-integrated-outing-efficiency-response.docx`
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
- Decision: when the live map detects inter-track gaps or a Nav GPX/card length
  mismatch, it shows a visible route-review warning. This is intentionally not
  a cosmetic failure; the runner needs to see that the source route should be
  reviewed rather than trust a cleaned-up drawing.
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
  - Added failing regressions for: live map must warn but not mask Nav GPX/card
    mismatch; `validate_outing_export()` must not treat a named connector cue as
    a hidden track-gap explanation; and `field_tool_completion_audit.py` must
    fail `source_gap_warning` even when generic named connector metadata exists.
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
  - `python -m pytest years/2026/tests/test_multi_start_alternative_audit.py years/2026/tests/test_export_mobile_field_packet.py years/2026/tests/test_field_tool_completion_audit.py`
    passed 72 tests.

#### May 8 correction: GPX/card mismatch is now a certification failure

- Objective: ensure the live map, field guide link, route card mileage, and GPX
  all describe the same car-to-car artifact instead of letting the map mask or
  compensate for source mismatches.
- Finding: the live map and field guide already used the same user-facing GPX
  href, but the refreshed route artifacts still disagreed with the route cards.
  Example: `1A-2. West Climb` shows 4.11 mi on the card while the field GPX
  measures about 11.33 mi. This is a source/export certifiability problem, not a
  map-display problem.
- Implementation: added `route_gpx_mileage_mismatch` to field-packet GPX
  validation with a 0.35 mi tolerance, changed the field-guide link copy to
  `Open Field GPX`, and changed the live-map warning copy from `Official GPX`
  to `Route GPX`.
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

#### May 8 correction follow-up: no placeholder field packet

- Objective: restore the phone packet as a usable field artifact before field
  use, while preserving the rule that the route card and GPX must describe the
  same route.
- Decision: do not publish a placeholder and do not show per-route
  `GPX validation failed` warnings in the field guide. If a stale upstream
  mileage estimate disagrees with the actual generated field GPX, the field
  packet now derives the displayed on-foot mileage and time bucket from the GPX
  track, preserving the original value as `source_on_foot_miles` for audit
  evidence.
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

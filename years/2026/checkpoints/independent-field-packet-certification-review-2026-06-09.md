# Independent Field Packet Certification Review - 2026-06-09

Independent re-certification of the 31-route 2026 field packet (`docs/field-packet/`,
canonical source `2026-outing-menu-map-data.json`, HEAD `5e1582f` "Certify 2026 field
packet maps"). Method: seven parallel read-only audit dimensions (source consistency,
certification history/drift, privacy, route-review policy gate, hiker-eye human
validity, live-map contract, full test suite), an independent re-run of the binding
8-command certification chain, and adversarial verification of every blocker/major
finding by 2-3 independent skeptics per finding. 22 findings reached verification:
20 confirmed, 2 refuted.

## Verdict

The packet is **runnable but not certifiable as accepted**. The repo's own
distinction (route-review-policy.md: "Certification is not route acceptance",
Heuristic 17: "Certification is not non-dominance") is exactly where it fails:
the 8-command chain passes cleanly, but the dominance/acceptance layer above it
has silently degraded into a rubber stamp, and two routes that the chain cannot
see are field-wrong by the repo's own accepted-anchor history.

## Independent chain re-run (2026-06-09)

All 8 commands passed on the current worktree: exporter wrote 93 GPX + packet,
31/31 routes field-ready, 251/251 official segments covered, 20/20 completion
requirements including the land-manager special-management hard gate, 31/31
headless walkthroughs, 42/42 latent cross-route segments reconciled, 0 edge-cover
hard failures, 96 post-credit connector proofs with 0 findings (7 preserved
advisories). Regeneration drift vs HEAD is timestamp/version-hash only; the field
payload is byte-identical. A skeptic also re-ran the two chain audits from strict
HEAD code (without the uncommitted relaxations) and the packet still passes, so
the committed certification does not depend on the uncommitted leniency changes.

One-route-truth also passes fully: 0 metric diffs across the private canonical
JSON, public/example sanitized JSONs, field-tool-data.json, manifest.json, all 31
index.html cards, and all three menu markdowns; SHA chain intact; GPX layer exact
(31/31/31, no orphans/missing, zip and service-worker mirror).

## Confirmed blockers

1. **Privacy leak in committed sanitizer code.** Private-address-derived literals
   are embedded in committed public scripts (the redaction layer encodes the value
   it redacts) and personal Strava activity ids/names/dates appear in public
   sanitized artifacts and committed experiment files. Specifics intentionally not
   written here; delivered privately. Fix: load private patterns from an ignored
   file, scrub or accept history exposure, re-run the public-safety check.
2. **Adversarial disproof registry is a rubber stamp.**
   `refresh_all_route_adversarial_disproof.py` emits identical proof text for all
   31 routes, hardcodes all attack checks to True and
   `deterministic_same_credit_failure_count=0`/`route_efficiency_achieved=True`,
   and decision `HOLD_CURRENT_RECERTIFIED` for every card. The actual
   deterministic review pack (route-review-all-dev) covers 43 routes of the OLD
   49-card menu - zero of the current 31 cards at current hashes. The dominance
   gate has never run against the shipped menu. The registry refresh must fail
   closed when no current review pack exists.
3. **Route 10A accepted-replacement regression.** 10A claims the exact H1 segment
   set but ships from the "Harlow's / Hidden Springs west access probe" anchor
   (parking_confidence `manual_required`, parking.field_ready=false) at 21.84
   on-foot mi / p75 360, vs the accepted H1 Avimor Spring Valley Creek card at
   9.64 mi / p75 289 (PASS_NON_DOMINATED, user-confirmed access). Checkpoint
   10a-ms-08-access-verification-2026-05-10 explicitly ruled this start
   parking-gated and "do not promote". Its 8.32-mi access leg is a single
   follow-the-GPX cue with no road names. Do not run 10A as published.
4. **Segment 1680 (The Face Trail 1, 1.15 mi) double-claimed** as exact credit by
   routes 17 and 18A in the canonical source (252 claim entries / 251 unique);
   plan-wide official miles 165.59 vs the 164.43 official total. Neither card
   declares it owned-elsewhere. Assign one owner, demote the other traversal to
   declared repeat, and add a dual-claim check to the latent-credit/edge-cover
   audit.
5. **12 committed tests fail at the certified HEAD** (reproduced in a clean HEAD
   worktree): 4 label-pinned tests still expect the retired FD*/49-card menu
   (including the only regression proofs of the FD18A special-management gate and
   the FD12A depot-phase-reset fix), and 8 exporter synthetic-fixture tests fail
   on real behavior drift (fixture routes dropped, zero access gaps, cue-mileage
   mismatch). Full suite: 12 failed / 692 passed in ~33 min (~15x runtime
   regression vs the 137s recorded baseline; test_export_mobile_field_packet.py
   alone is ~701s). No full-suite run recorded since 2026-05-24.

## Confirmed majors (summary)

- **1A-1 recreates the canonical FD14D regression**: single-segment 1482 card from
  "prior parking anchor 13" at 3.17 mi / p75 119 vs the accepted N 36th St anchor
  at 1.5 mi / p75 60 - the exact regression the policy names as canonical. Also:
  area name renders as raw "13", colliding with route 13; return cued only as an
  OSM connector id.
- All 31 `start_justification` fields are exporter-synthesized circular
  boilerplate; the canonical source contains none.
- No verified water, heat, shade, or bailout annotations anywhere in the packet
  for a June 18 - July 18 challenge; route 13 is 32.47 mi / 4,964 ft with its
  biggest climb starting at mile ~24, car_pass_count 0, no water.
- 54 of 260 phone cues carry physically impossible repeat-official mileage notes
  (repeat miles exceed the leg's own length), e.g. a 0.21-mi connector noted
  "Includes 4.56 mi repeat official".
- Registry summary asserts `route_efficiency_achieved=true` while the live
  route-efficiency audit verdict is `not_proven` with two failed gates.
- Committed certification depends on uncommitted code: the exporter imports four
  dirty modules; the 05-26..28 working session (scripts, tests, checkpoints,
  work-log entries) is unpushed 9 days before the window.
- Live map: no direction chevrons render on source-path active legs, including
  all 25 overlap/double-back legs (the case arrows exist for); Start GPS omits
  distance-to-route/progress readout.

## Refuted findings (for the record)

- "No dated checkpoint for the 2026-06-05 chain": refuted - 5e1582f itself
  commits refreshed checkpoints for the checkpoint-writing audits; the real
  (minor) defect is that audits overwrite stale-dated filenames instead of
  writing current-date files.
- "Chain passes may depend on uncommitted audit relaxations": refuted by
  re-execution from strict HEAD code; the packet passes either way, and the
  relaxations match documented walker-debugging doctrine.

## Recommended order of work before 2026-06-18

1. Privacy remediation (see above; includes history decision).
2. Re-run the real deterministic dominance gate against the current 31 cards;
   make the registry refresh fail closed; write real start_justifications.
3. Repair 10A (re-promote H1 Avimor or waiver) and 1A-1 (N 36th anchor).
4. Fix segment 1680 ownership; add dual-claim audit.
5. Commit or reconcile the uncommitted working session; fix the 12 red tests;
   record a full-suite run.
6. Add verified water/heat/bailout annotations for every route with p75 > ~180
   min; reconsider route 13's single-day shape.
7. Live-map: overlap-leg chevrons, distance-to-route readout, repeat-mileage cue
   note fix.

# Full 2026 Manual Route Map-Challenge Review

Generated: 2026-05-16

Status: `blocking_contradiction_found`

This checkpoint is a manual, route-by-route map challenge of the current 2026 field packet. It does not regenerate or mutate the active field packet. It asks the question the deterministic exact-credit gate does not fully answer:

> Could a human looking at the route map, route grouping, known access, and prior audit evidence reasonably find a better anchor, partition, or field-day shape than the current card?

## Bottom Line

Yes, the manual review made a material difference.

The exact-credit route-review gate found no active same-credit anchor dominance in the current 47-route packet: `44 PASS_NON_DOMINATED` and `3 PASS_WITH_JUSTIFIED_BURDEN`. That is useful, but it did not catch a larger route-truth contradiction: the current generated field packet still contains five Harlow / Avimor split cards that a tracked certification checkpoint says were removed and replaced by certified route card `H1`.

The active packet currently contains:

- `47` route cards.
- `164.44` official miles.
- `289.58` on-foot miles.
- `7662` p75 minutes.
- `8628` p90 minutes.
- On-foot / official ratio: `1.761x`.

The tracked H1 certification says:

- `H1` exists and was certified after recertification.
- `FD24A`, `FD27A`, `FD27B`, `FD27C`, and `FD30A` were removed.
- Those five old cards cost `34.00 mi / 991 p75 / 1117 p90`.
- `H1` costs `9.64 mi / 289 p75 / 324 p90`.
- Savings: `24.36 mi / 702 p75 / 793 p90`.

If H1 is reconciled into the current 47-route packet without changing any other route, the modeled packet becomes:

- `43` route cards.
- `164.43` official miles.
- `265.22` on-foot miles.
- `6960` p75 minutes.
- `7835` p90 minutes.
- On-foot / official ratio: `1.613x`.

That single reconciliation moves the plan from failing the accepted proof limit (`1.761x`) to nearly the preferred target (`1.613x`, about `2.13` on-foot miles above a strict `1.600x` target). It also removes the worst human-fatigue cluster in the current packet.

## Before / After Assessment

| Question | Before manual review | After manual review |
|---|---|---|
| Does the deterministic exact-credit gate pass? | Yes. It only checks known same-credit accepted anchors. | Still yes, but that is not enough. |
| Is the active field packet publication-clean? | It looked close because hard repeat failures were closed. | No. It contradicts the H1 certification checkpoint. |
| Is FD14D-style same-credit anchor dominance fixed? | Yes, FD14D is now lower N 36th. | Confirmed. FD14D remains keep-current. |
| Is the route set proven human-optimal? | No. Efficiency audit verdict was `not_proven`. | Still no, but the largest concrete gap is now identified and quantified. |
| Main blocking issue | Not visible from exact-credit gate alone. | Reconcile H1 versus current packet before calling the map site/field packet clean. |
| Main non-blocking queue | 49 optimization warnings, 21 unchallenged >2x components. | Converted into route-specific map-challenge actions. |

## Decision Counts

| Decision | Count | Meaning |
|---|---:|---|
| `BLOCKING_CONTRADICTION` | 5 | Current route should not be treated as publication-clean until the H1 packet contradiction is resolved. |
| `MAP_CHALLENGE_REQUIRED` | 11 | Route is runnable or certified enough to discuss, but a human-map/partition challenge is still required before claiming optimality. |
| `KEEP_GATED_FIELD_CHECK` | 14 | Route shape is plausible, but field-day legality, closure, direction, weather, or access checks remain material. |
| `KEEP_CURRENT` | 17 | No material human-footmile weirdness found in this pass beyond normal day-of checks. |

## Route Index

| Route | Decision | Why |
|---|---|---|
| `FD01A` | `KEEP_GATED_FIELD_CHECK` | Warm Springs/Table Rock route is plausible but has declared-repeat burden and heat exposure. |
| `9` | `KEEP_CURRENT` | Veterans route has reasonable ratio and no stronger adjacent anchor surfaced. |
| `FD03A` | `KEEP_CURRENT` | Chukar Butte re-anchor is efficient; keep private-source details out of public artifacts. |
| `FD04A` | `MAP_CHALLENGE_REQUIRED` | Freestone/Shane's has 6.01 overhead miles and shares pressure with route `3` and `FD20A`. |
| `4A` | `KEEP_CURRENT` | Bob's/Urban is short, plausible, and low overhead. |
| `FD05A` | `KEEP_CURRENT` | Hull's Gulch Interpretive route is efficient and normal. |
| `FD06A` | `MAP_CHALLENGE_REQUIRED` | Lower Interpretive to Fat Tire/Curlew is over 2x and needs proof that access is not avoidable. |
| `FD07A` | `KEEP_GATED_FIELD_CHECK` | Short Bogus route; field-date access and mountain operating state matter more than map optimality. |
| `FD07B` | `KEEP_GATED_FIELD_CHECK` | Deer Point has high ratio but likely mountain-access constrained; do not claim clean without Bogus/R2R check. |
| `FD08A` | `MAP_CHALLENGE_REQUIRED` | Cartwright Ridge same-trailhead pressure with `FD08B`; challenge as one Cartwright outing. |
| `FD08B` | `MAP_CHALLENGE_REQUIRED` | Cartwright Connector same-trailhead pressure with `FD08A`; challenge as one Cartwright outing. |
| `10B` | `KEEP_GATED_FIELD_CHECK` | Dry Creek/Bitterbrush route is plausible but still a >2x access-heavy outing. |
| `FD09A` | `KEEP_CURRENT` | West Hidden Springs re-anchor is the intended fix class and no new same-credit dominance surfaced. |
| `19` | `KEEP_GATED_FIELD_CHECK` | Cervidae is naturally out-and-back, but p75 and heat should remain gated. |
| `4B` | `KEEP_CURRENT` | Upper Interpretive/Scott's is small and plausible. |
| `14` | `KEEP_CURRENT` | Orchard/Five Mile/Watchman has good ratio despite long day; no stronger partition surfaced. |
| `FD12A` | `KEEP_GATED_FIELD_CHECK` | Harrison/Full Sail package is plausible but has enough repeat burden to keep proof notes. |
| `16A-1` | `KEEP_GATED_FIELD_CHECK` | Sweet Connie is a known wet-weather avoid trail and a long access outing; shape may be right but not clean year-round. |
| `FD14A` | `MAP_CHALLENGE_REQUIRED` | Doe Ridge is tiny official credit with order-free removal pressure; price Cartwright/Polecat bundling. |
| `FD14B` | `KEEP_GATED_FIELD_CHECK` | Quick Draw/CHBH is short but same Cartwright package should be reviewed with `FD14A` and `FD18A`. |
| `FD14D` | `KEEP_CURRENT` | FD14D regression is fixed: lower N 36th start, same 36th Chute credit, justified burden. |
| `3` | `MAP_CHALLENGE_REQUIRED` | Top dead-repeat candidate in Military/Freestone family; challenge as a combined Military Reserve route. |
| `15B` | `KEEP_CURRENT` | Red Tail/Landslide route is efficient and paired with Dry Creek access. |
| `7` | `KEEP_CURRENT` | Seamans/Wild Phlox route has normal overhead and no obvious better anchor. |
| `11` | `KEEP_CURRENT` | Hawkins is near 1.0x and should stay. |
| `16B` | `KEEP_CURRENT` | Stack Rock Connector is clean enough, with normal seasonal access checks. |
| `FD18A` | `MAP_CHALLENGE_REQUIRED` | Polecat/Peggy's is efficient on ratio but sits in Cartwright ownership/direction-rule pressure. |
| `FD19A` | `KEEP_GATED_FIELD_CHECK` | Short Kestrel card is okay only if Hulls same-day bundling and day rules remain clear. |
| `FD19B` | `KEEP_GATED_FIELD_CHECK` | Lower Hulls/Red Cliffs is plausible but Lower Hulls odd/even rule must be preserved. |
| `FD20A` | `MAP_CHALLENGE_REQUIRED` | Three Bears/Freestone Ridge has 6.38 overhead miles and partial-shrink evidence. |
| `FD21A` | `KEEP_GATED_FIELD_CHECK` | Homestead/PVO/Harris has high ratio but may be access-constrained; retain day-of proof. |
| `FD21B` | `KEEP_CURRENT` | Old Pen/Table Rock packet is plausible and moderate. |
| `FD22B` | `MAP_CHALLENGE_REQUIRED` | Crestline has high ratio and partial-shrink pressure with Hulls/8th Street ownership. |
| `FD22C` | `KEEP_CURRENT` | Grove/Owl's Roost/15th/Gold Finch is plausible and public-safe. |
| `12` | `KEEP_CURRENT` | 8th Street/Sidewinder/Corrals/Highlands is long but efficient for official credit. |
| `FD24A` | `BLOCKING_CONTRADICTION` | Current packet contains this Harlow split card, but H1 certification says it was removed. |
| `FD25A` | `KEEP_GATED_FIELD_CHECK` | Elk Meadows has high ratio but mountain access likely explains it; check Bogus/FS status. |
| `FD25B` | `KEEP_GATED_FIELD_CHECK` | The Face has high ratio but mountain access likely explains it; check Bogus/FS status. |
| `FD26A` | `KEEP_GATED_FIELD_CHECK` | Around the Mountain is direction-managed and date/access-sensitive. |
| `FD27A` | `BLOCKING_CONTRADICTION` | Current packet contains this tiny Spring Creek split card, but H1 certification says it was removed. |
| `FD27B` | `BLOCKING_CONTRADICTION` | Current packet contains this Spring Creek split card, but H1 certification says it was replaced by H1. |
| `FD27C` | `BLOCKING_CONTRADICTION` | Current packet contains this Whistling Pig split card, but H1 certification says it was removed. |
| `15A-1` | `KEEP_CURRENT` | Dry Creek/Shingle has near 1.0x ratio and should stay unless condition checks fail. |
| `FD28A` | `KEEP_CURRENT` | Miller Gulch Connector is short and plausible. |
| `16A-2` | `MAP_CHALLENGE_REQUIRED` | Sheep Camp is 4.30x for 0.77 official miles; this remains a manual-design target. |
| `FD30A` | `BLOCKING_CONTRADICTION` | Current packet contains this Harlow/Twisted Spring card, but H1 certification says it was removed. |
| `18` | `MAP_CHALLENGE_REQUIRED` | Bogus/Pioneer route has 6.17 overhead miles and partial-shrink pressure from adjacent Bogus cards. |

## Manual Review Findings

### 1. H1 is the blocking route-truth issue

This is the largest concrete result. The current packet has a route set that a certified checkpoint says should have been superseded. This is not merely "not globally optimal." It is conflicting route truth across generated artifacts.

The manual conclusion is stricter than the route-review gate:

```text
Do not call the current map site data fully clean until H1 versus the current 47-card packet is reconciled.
```

This needs a source-of-truth decision, not a new optimizer:

1. If H1 is still valid, regenerate the active packet from the canonical source that contains H1, remove `FD24A`, `FD27A`, `FD27B`, `FD27C`, and `FD30A`, and rerun the field-packet gate.
2. If H1 was intentionally backed out, write the reason and retire or supersede the H1 certification checkpoint so future agents do not treat it as active truth.
3. If H1 is valid but date assignment changed, preserve H1 and reassign the calendar rather than restoring the five expensive split cards.

### 2. FD14D is fixed, but FD14D is not the whole class

FD14D now uses the lower N 36th Street start and should remain `KEEP_CURRENT`. The route-review gate caught the same-credit accepted-anchor class it was built for.

The manual review found the next layer: route quality can regress through packet/source drift and partition shape even when exact-credit anchor dominance passes.

### 3. The Freestone/Military family still needs a human map challenge

Routes `FD04A`, `FD20A`, and `3` are all locally runnable, but they are not proven as a human-optimal partition. The route-repeat and repeat-productivity audits point to dead-repeat and ownership pressure, especially route `3` and `FD20A`. A human challenge should look at the whole Freestone / Shane's / Three Bears / Military Reserve field shape, not one card at a time.

### 4. Bogus routes are not publication-clean without current access checks

The Bogus group should not be rewritten just because of high ratios. Mountain trail access, road closures, seasonal status, and chair/lodge starts can make high ratios legitimate. But several cards (`18`, `FD25A`, `FD25B`, `FD07B`) remain day-of-access and closure sensitive. The current public Bogus sources show a 2026 Deer Point stewardship closure window and current mountain operating status; those must be checked against the actual scheduled dates before field use.

### 5. Hulls and Polecat need rule-aware review, not just distance review

Lower Hulls has odd/even separation of use; Polecat and Around the Mountain are direction-managed. These are route-quality constraints, not annotation details. A route can be short and still fail the human route if it sends the runner into the wrong date/direction rule.

## Adjacent Frames Checked

### Frame A: Exact-credit route-review gate

Question answered: Is a current card dominated by a known accepted same-credit anchor?

Result: useful, but insufficient. It caught FD14D-style anchor dominance, not H1 source drift or larger partition failures.

### Frame B: Runner/hiker outing frame

Question answered: Would a person starting from the parked car experience unnecessary dead miles, confusing repeat, wrong-day restrictions, or weird route intent?

Result: exposed high-priority challenge targets where the current route may be runnable but still annoying or wasteful.

### Frame C: Field-day / partition frame

Question answered: Is the route card the right decision unit, or should this trail family be grouped, replaced, or scheduled differently?

Result: exposed H1 as the real decision unit for Harlow/Avimor and exposed Freestone/Military, Cartwright/Polecat, Hulls, Bogus, and Dry/Sweet as family-level review problems.

### Frame D: Artifact-source frame

Question answered: Do the map site, field-packet data, checkpoints, and certification records describe one route truth?

Result: no. The current packet conflicts with the H1 active-packet certification checkpoint.

## Adversarial Failure Stories

- The exact-credit gate passes all routes, but the active packet silently restores five old Harlow cards after H1 was certified, adding about 24 miles of avoidable human fatigue.
- A route card is certified and has GPX, but the map site and checkpoint disagree about whether it should exist, leaving the runner to follow a stale public artifact.
- A high-ratio mountain card is rejected too aggressively, but the apparent shortcut crosses a seasonal road closure or closed trail. This is why Bogus remains field-gated, not automatically redesigned.
- A short Hulls or Polecat route looks efficient on distance but violates day/direction management and becomes invalid or unsafe in the field.
- A Freestone card is locally coherent, but three adjacent cards reuse the same access corridors in a way a human would consolidate or repartition.
- A prior replacement checkpoint remains in the repo after a legitimate rollback, so future agents keep trying to reinstate a route that was intentionally rejected. The fix is to reconcile or supersede the checkpoint.
- A public map is refreshed from one source while the phone packet or outing menu uses another, recreating the one-route-truth failure class.
- The review answers "does this known candidate pass?" while the actual downstream question is "would I choose this outing from scratch tomorrow?"

## Required Next Actions

1. Reconcile H1 against the current active packet before calling the map site data clean.
2. Re-run field-packet export and consistency checks after the H1 source decision.
3. Run a focused manual map challenge for:
   - Freestone/Military: `FD04A`, `FD20A`, `3`.
   - Cartwright/Polecat: `FD08A`, `FD08B`, `FD14A`, `FD14B`, `FD18A`.
   - Hulls/Crestline: `FD19A`, `FD19B`, `FD22B`.
   - Dry/Sweet/Sheep: `16A-1`, `16A-2`, `15A-1`.
   - Bogus/Pioneer: `FD07A`, `FD07B`, `FD25A`, `FD25B`, `FD26A`, `18`.
4. Keep FD14D as the canonical fixed regression example, but add H1 as the canonical artifact-drift/partition-regression example.

## Evidence Used

Local files:

- `docs/field-packet/field-tool-data.json`
- `years/2026/outputs/private/route-reviews/route-review-all-dev.review.json`
- `years/2026/checkpoints/harlow-h1-active-packet-certification-2026-05-12.md`
- `years/2026/checkpoints/harlow-h1-active-packet-certification-2026-05-12.json`
- `years/2026/checkpoints/harlow-h1-promotion-assertions-2026-05-12.json`
- `years/2026/checkpoints/route-efficiency-audit-2026-05-06.md`
- `years/2026/checkpoints/route-repeat-optimization-audit-2026-05-12.json`
- `years/2026/checkpoints/ownership-reassignment-optimization-audit-2026-05-12.json`
- `years/2026/checkpoints/repeat-productivity-audit-2026-05-12.json`

Public sources checked on 2026-05-16:

- Ridge to Rivers home and conditions links: `https://www.ridgetorivers.org/`
- Ridge to Rivers Military Reserve page: `https://www.ridgetorivers.org/trails/trail-areas/military-reserve/`
- Ridge to Rivers management strategies: `https://www.ridgetorivers.org/trail-news/ridge-to-rivers-adopts-management-strategies-from-pilot-trail-program/`
- Ridge to Rivers 2024 map PDF: `https://www.ridgetorivers.org/media/1181/r2r_2024_map.pdf`
- Avimor trails page: `https://www.avimor.com/trails-and-outdoors`
- Bogus Basin Deer Point Stewardship Project 2026: `https://bogusbasin.org/about-bogus/culture/deer-point-stewardship-project-2026/`
- Bogus Basin conditions: `https://bogusbasin.org/your-mountain/conditions-webcams/`

## Scope Boundaries

- This report does not mutate the active packet, route cards, GPX, phone packet, or map site.
- This report does not prove a global optimum.
- Public-source checks are current as of 2026-05-16 and must be rechecked near field execution.
- No raw private GPS traces, exact private coordinates, activity IDs, dashboard data, tokens, or home-origin data are included.

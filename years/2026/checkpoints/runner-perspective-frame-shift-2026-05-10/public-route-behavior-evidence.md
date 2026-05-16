# Public Route Behavior Evidence For Optimization

## Purpose

This file adds the clarified evidence lane: `what do you see?` is not the deliverable. It is a way to step outside the route-card frame and ask how real runners and riders usually solve the same terrain.

Public Strava routes are useful behavioral evidence for starts, connectors, loop direction, heat/shade, crowding, and common bailout/return patterns. They are not official 2026 BTC truth, do not prove current trail legality, and do not prove the user's challenge credit.

## Evidence Boundary

- Evidence type: public route descriptions and segment lists from Strava Routes pages.
- Date risk: most source routes are from 2020-2022 and may be stale for signage, closures, and current trail management.
- Use: optimization hypothesis generation and proof-target selection.
- Do not use: official segment truth, current access legality, day-of conditions, or completion credit.

## Source-Backed Optimization Leads

### Dry Creek / Shingle / Sheep Camp / Sweet Connie Cluster

Applies to: `15A-1`, `16A-1`, `16A-2`, and nearby Dry Creek lower-cluster repairs.

Public pattern:

- Strava's public `Shingle Creek and Dry Creek Loop` starts from a large Bogus Basin Road pullout, climbs Shingle Creek, then descends Dry Creek back to the road.
- The same source calls out Shingle as exposed and climbing-heavy, Dry Creek as the downhill return, and the lower canyon as potentially crowded with hikers.
- Its segment list includes `Shingle Creek Trail#79 (ascent)`, `Dry Creek - Sheep's Camp Bridge to Shingle Creek`, `Sweet Connie to Dry Creek connector`, and `Dry Creek to Sweet Connie connector`.

Optimization hypothesis:

The current `16A-2` route is probably carrying too much same-corridor overhead because it treats Shingle/Sheep Camp/Dry Creek-side movement as one long roadside out-and-back. A real-world loop pattern exists for at least Shingle plus Dry Creek from the Bogus Basin Road pullout. Test whether the BTC route set should split or recombine `15A-1`, `16A-1`, and `16A-2` around that public loop pattern instead of preserving the current long return legs.

Proof needed:

- Confirm the pullout is legal/repeatable for current use.
- Verify the exact loop covers required official BTC segment edges and ascent-only rules.
- Price variants: Shingle-up/Dry-down loop, Shingle+Sheep Camp split, Sweet Connie separate climb, and same-day re-park.
- Check current closures, mud, heat, water, and day-of crowding.

Sources:

- https://www.strava.com/routes/3247884021205428732?hl=en-GB
- https://www.strava.com/routes/3247884092950786556

### Freestone / Three Bears / Shane's / Central Ridge Cluster

Applies to: `3`, `13`, and Military Reserve/Freestone split candidates.

Public pattern:

- Strava's public `Shane's Loop from Freestone Creek` begins from Freestone Creek trailhead.
- It describes Mountain Cove as a gentle warm-up, warns that the opposite direction climbs sharply right away, and treats Shane's plus Three Bears as the upper loop.
- It returns via Central Ridge and emphasizes watching for the correct lower junction back to the trailhead.

Optimization hypothesis:

The current `13` route is a giant catchall with many repeat/connector warnings. A real-world smaller Freestone loop pattern exists for Shane's/Three Bears/Central Ridge. Use that as a decomposition hint: split the large `13` cluster into human-recognizable Freestone loops before trying to locally optimize the 25-mile all-in-one card.

Proof needed:

- Compare a Shane's/Three Bears loop card against the current `13` segment coverage.
- Move Freestone Ridge / Curlew / Two Point leftovers into a second legal loop or re-park candidate.
- Preserve ascent rules and future-day coverage.

Source:

- https://www.strava.com/routes/3247884125053918210?hl=en-GB

### Bogus / Around The Mountain / Deer Point / Pioneer-Simplot Return

Applies to: `17`, `18`, and Bogus field-day pairing.

Public pattern:

- Strava's public `Around the Mountain` route frames ATM as a counter-clockwise runner route starting with Deer Point from Simplot Lodge.
- It notes that ATM does not form a complete loop by itself, ends near Pioneer Lodge, and that runners can descend via Yellow Brick Road to Bogus Creek to return to the start.
- It also frames Bogus as cooler than lower foothills in summer, but with exposed/open meadow sections.

Optimization hypothesis:

The current Bogus route cards use Simplot and Pioneer starts plus several OSM track/service connectors. Public route behavior suggests the important optimization question is not only `which official trails are grouped?`, but `what named return from Pioneer-side ATM/lodge terrain do runners actually use to get back to Simplot?` Replace generic OSM return proof where possible with a named runner-used descent/return pattern, or split `17` and `18` as a day-level Simplot/Pioneer pair with explicit transfer.

Proof needed:

- Verify current 2026 ATM direction/signage and the June 18-19 closure window.
- Confirm foot legality for any Yellow Brick/Bogus Creek return path.
- Price Simplot-only loop, Pioneer-only loop, and same-day transfer variants.

Source:

- https://www.strava.com/routes/3247883968082699266

### Hulls / Kestrel / Crestline Frontside

Applies to: `2`, `12`, and lower frontside split candidates.

Public pattern:

- Strava's public `Hulls Gulch and Kestrel Loop` uses a compact loop: up Lower Hulls, briefly on Motorcycle Trail, then Crestline and down Kestrel.
- It explicitly calls out the odd/even Lower Hulls foot-travel constraint and says the Lower Hulls climb / Kestrel return direction is commonly preferred.

Optimization hypothesis:

The current `2` route is a large combined frontside route with many connector/repeat warnings. Public route behavior supports extracting a compact Hulls/Kestrel/Crestline loop as a stable human unit, then treating remaining lower-frontside BTC edges as separate add-ons rather than forcing a single oversized route card.

Proof needed:

- Preserve Lower Hulls day legality.
- Compare compact loop plus add-on cards against the current 17.26-mile on-foot `2`.
- Validate whether `12` shares connector/repeat cost that should be separated from Hulls/Crestline work.

Source:

- https://www.strava.com/routes/trail-running/usa/idaho/3247884062304991234

### Hillside To Hollow / West Climb / Buena Vista / Full Sail

Applies to: `1A-1`, `1A-2`, and any Harrison/Hillside repair.

Public pattern:

- Strava's public `Hillside to Hollow Ridge Loop` starts from Hillside Park and says the area is a complex web with wide-open views but confusing signs.
- It also says the Harrison Hollow side can be an alternate start and warns the route has no shade and is best in milder conditions.

Optimization hypothesis:

For `1A-1` and `1A-2`, runner-perspective confusion is not just a UX risk; it may point to route decomposition around known public starts: Hillside Park, Harrison Hollow, West Climb, and Full Sail/Who Now/Kemper's. The public pattern supports keeping the accepted split/re-park logic alive instead of collapsing the area back into a single graph-valid card.

Proof needed:

- Compare Hillside Park and West Climb starts on p75, shade/heat, and segment coverage.
- Preserve the already field-learned Harrison/West Climb split behavior.
- Avoid relying on trail names alone in the dense cue network.

Source:

- https://www.strava.com/routes/3247884062377233410

### Polecat / Cartwright

Applies to: `5A`, `5B`, `6`.

Public pattern:

- Strava's public `Polecat Gulch Loop` is a 6.49-mile loop and says it starts from Polecat Gulch trailhead, with Cartwright Road parking as an alternate start.
- It also calls out counter-clockwise travel and slower drying/mud sensitivity.

Optimization hypothesis:

The Cartwright/Polecat cards should preserve direction/current-management constraints and compare Polecat Gulch versus Cartwright starts explicitly. Public route behavior says both starts are real-world patterns, so a graph-only nearest-start choice is too narrow.

Proof needed:

- Verify current Polecat direction/signage before any challenge-window run.
- Reprice Cartwright versus Polecat Gulch starts, including `5B` and `6` future-day preservation.
- Treat mud/slow-drying risk as a scheduling constraint.

Source:

- https://www.strava.com/routes/3247926981866071220

### Table Rock / Castle Rock / Rock Island / Quarry

Applies to: `4A`, `4B`, `4C-1`, `4C-2`.

Public pattern:

- Strava's public `Quarry Trail and Eagle Rock Loop` starts from the Old Penitentiary / Eagle Rock area and uses Quarry plus Native-tribe-named trails as a compact loop.
- It notes steep/rocky sections, erodible clay soils, and closure/mud sensitivity.
- Strava's public `Rock Island Loops` is bike-oriented but describes a common Tram / Rock Island East / Rock Island West / Rock Garden loop pattern and the technical nature of the Mesa Reserve trails.

Optimization hypothesis:

The `4C-2` Strava parking anchor and the `4C-1` Warm Springs Golf Course anchor should be compared against Old Penitentiary/Eagle Rock and Tram/Rock Island real-world loop patterns. The public routes suggest multiple compact loop units rather than one parking-anchor-dependent Table Rock/Castle/Rock Island bundle.

Proof needed:

- Verify current pedestrian legality and any bike-priority/yield constraints on Rock Island.
- Compare Old Pen, Warm Springs Golf Course, and Castle-side starts.
- Preserve Shoshone-Paiute / Quarry / Rock Island official edge coverage without overfitting to a stale public route.

Sources:

- https://www.strava.com/routes/3247884018258545002
- https://www.strava.com/routes/3247926945418657276

## How To Use This Evidence Next

1. Start with the top optimization-index routes: `13`, `3`, `2`, `10A`, `16A-2`, `17`, `18`, `6`.
2. For each, compare the generated optimization leads against the relevant public route pattern above.
3. Convert only the promising pattern into a bounded route-local experiment: candidate anchor(s), connector options, segment coverage check, ascent-direction check, p75/p90 cost, and future-day preservation.
4. Keep any public Strava route as behavioral evidence only; the current-year official BTC API and Ridge to Rivers/current-condition checks still decide route legality and challenge credit.

# Outing Efficiency Research Agenda

Generated: 2026-05-08

Objective: review every current 2026 field-menu outing for within-outing efficiency opportunities: alternate parked starts, trail order, legal direction, split/re-park options, personal Strava evidence, public trail reports, current trail notes, and field-verification work still needed.

This started as a research agenda; after the corrected pass, `1A`, `4C`, `5`,
and `15A` were promoted into the regenerated field packet. The current
executable field packet is `years/2026/outputs/private/2026-outing-menu-map-data.json`
and its generated phone/GPX artifacts. Any future alternative below must be
promoted only after parking, legal access, cue text, continuous GPX, coverage,
p75/p90 time, and field-route walkthrough audits pass from the same regenerated
source.

## Evidence Used

Local route and proof artifacts:

- Reviewed synthesis document: `/Users/scott/Desktop/btc-2026-integrated-outing-efficiency-response.docx`.
- Canonical private field menu: `years/2026/outputs/private/2026-outing-menu-map-data.json`.
- Multi-start route math: `years/2026/checkpoints/multi-start-alternative-audit-2026-05-08.json` and `.md`.
- Private route-review details: `years/2026/outputs/private/multi-start-alternatives/multi-start-alternative-audit-2026-05-08.json`.
- Google Earth parking review notes for assumed road anchors: `years/2026/outputs/private/multi-start-alternatives/parking-earth-screenshots/parking-earth-review-2026-05-08.md`.
- Personal Strava-derived segment history summary: `years/2026/derived/strava/strava-segment-history-summary-2026-05-06.json`.
- Private Strava parking-anchor summary: `years/2026/inputs/personal/private/strava-parking-anchors-v1.geojson`.
- Segment crosswalk and current R2R/open-data condition tags: `years/2026/derived/segment-crosswalk/segment-crosswalk-2026-05-06.csv`.
- Elevation / grade-adjusted effort source: `years/2026/derived/elevation/segment-elevation-2026-05-06.csv`.
- Official map update decision: `years/2026/checkpoints/official-map-update-recommendation-2026-05-08.md`.

Strava parking-anchor interpretation: per `AGENTS.md`, private Strava-derived
parking anchors from the user's activity endpoint clusters mean the user has
actually parked there before. They are valid planning anchors, not merely
theoretical suggestions. Public/shareable outputs should still hide exact
coordinates and raw activity ids, and public outputs should use a public-safe
name when one is available. Do not add a generic parking-review blocker
just because the anchor came from Strava; only add one when there is specific
evidence of changed access, ambiguous/private access, or user uncertainty.
The same principle applies to user-reviewed parking anchors with a `yes`
decision: treat them as valid planning anchors unless the review itself or a
new source raises a concrete access concern.

Public/current sources checked:

- Boise Trails Challenge home/about pages for 2026 challenge dates, on-foot rules, segment credit, and app/GPS-upload guidance: `https://boisetrailschallenge.com/`, `https://boisetrailschallenge.com/about`.
- USDA Forest Service Deer Point Road, Trail, and Area Closure, Order #0402-01-117: `https://www.fs.usda.gov/r04/boise/alerts/deer-point-road-trail-and-area-closure`.
- Ridge to Rivers condition reports and interactive map entrypoints: `https://ridgetorivers.org/condition-reports/`, `https://ridgetorivers.org/trails/interactive-map/`.
- Ridge to Rivers wet-weather guidance: `https://ridgetorivers.org/trail-guide/trail-etiquette/wet-weather-and-winter-trail-use/`.
- Ridge to Rivers best-time / heat guidance: `https://ridgetorivers.org/trail-guide/best-times-to-hit-the-trails/`, `https://ridgetorivers.org/trail-guide/beat-the-heat-hikes/`.
- Ridge to Rivers trail area pages for Hillside to Hollow, Hulls/Camel's Back, Military Reserve, Polecat, Hawkins, Oregon Trail, Table Rock, and Bogus Basin.
- Ridge to Rivers 2024 map PDF for wet-weather good bets / avoid trails: `https://www.ridgetorivers.org/media/1181/r2r_2024_map.pdf`.
- Bogus Basin current trail report: `https://bogusbasin.org/your-mountain/trails-grooming/`.
- Bogus Basin operating-hours page and local reporting on June 19 summer opening / weekday facility limits: `https://bogusbasin.org/your-mountain/operating-hours/`, `https://www.kivitv.com/news/local-news/in-your-neighborhood/bogus-basin/bogus-basin-announces-summer-opening-date-music-on-the-mountain-lineup`.
- Local reporting on Bogus Basin Road and Deer Point closure windows, used only as supplemental timing context behind the Forest Service order: `https://www.kivitv.com/news/local-news/in-your-neighborhood/boise-county/closures-set-to-begin-as-logging-resumes-near-bogus-basin`, `https://idahonews.com/news/local/bogus-basin-road-closed-weekdays-for-tree-work`.
- Recreation.gov 8th Street Trailhead summary: `https://www.recreation.gov/camping/poi/255214`.
- BoiseTrails condition/report pages for key local-trail color: Lower Hulls, Polecat Loop, 36th Street Chute, Three Bears, Corrals, Sweet Connie, Around the Mountain, and the BoiseTrails area index.

## High-Yield Findings

- With the user-decision questions resolved and the ascent-preservation bug fixed, the certified promotion set is `1A`, `4C`, `5`, and `15A`. High-upside access-validation candidates remain `10A` and `19`. `13` and `17` are held because the corrected audit no longer shows a better split once ascent-only segments are preserved.
- The 2026 challenge window is a hard scheduling anchor: June 18 through July 18, 2026. The BTC app workflow is tested and confirmed for official recording, so recording method is not an open planning risk. Historical Strava remains planning evidence, not the assumed 2026 ingestion path.
- The corrected multi-start audit evaluated 24 multi-segment outing components and retained 50 alternatives for review. It found 4 promising no-blocker alternatives and 3 parking-check alternatives in the current checkpoint.
- Certified promoted alternatives:
  - `5` Polecat / Barn Owl: split Barn Owl from the Polecat core. Saves 3.97 on-foot miles and 47 elapsed minutes, with the West Hidden Springs Drive parking anchor accepted by user review.
  - `15A` Highlands / Connector / Dry Creek: split Dry Creek from Bob's/Highlands. Saves 2.41 on-foot miles and 38 elapsed minutes.
  - `1A` West Climb: split 36th Street Chute from the rest. Saves 2.46 on-foot miles but adds 32 elapsed minutes. Because slower splits are acceptable when they create bailouts, this is promoted as a real optional split rather than rejected for time alone.
  - `4C` Table Rock / Castle Rock / Tram: split Warm Springs/Tram from the Castle Rock side. Saves 0.87 on-foot miles and adds 9 elapsed minutes; promoted because it is now certified and gives a useful car-access/bailout option.
- High-upside access-validation alternatives:
  - `10A` Harlow / Hidden Springs has multiple road-parking split candidates with 2.67 to 3.38 fewer on-foot miles and 24 to 43 fewer minutes. Legal residential road starts are acceptable, so the research task is to verify public/legal/capacity/cue suitability and then keep any verified starts in the candidate set.
  - `19` Cervidae still relies on generic OSM parking/access evidence and remains a simpler parking/access verification item.
- `13` and `17` were initially ranked too high because the reverse-order split heuristic dropped non-reversible ascent-only trails. After fixing that bug, `13` and `17` splits are worse than the current baseline and should not replace the canonical routes.
- The Deer Point Forest Service closure is still a first-two-day challenge-window gate for Bogus/Stack access. Avoid affected routes on June 18 and before the relevant June 19 closure windows clear; after June 19, treat normal day-of closure/status checks as the controlling source unless a new order appears.
- Bogus lodge/facility availability is not a route blocker. The user does not need lodge support; for Bogus outings, verify trail legality, lot access, weather, and closure status, not lodge water/restrooms/food/lifts.
- Local Strava evidence is uneven. The best-covered outings by imported personal segment efforts are `13` (7 matched official segments / 27 efforts), `1A` (3 / 15), `14` (3 / 11), `15A` (5 / 11), `12` (3 / 9), and `3` (4 / 9). Several current outings have no matched Strava segment efforts and need field-test calibration or public route-report review.
- Some routes are already plausible as-is, but remain condition-sensitive. Ridge to Rivers flags wet-weather risk for Table Rock, Polecat, Sweet Connie, Cottonwood/Ridgecrest/Central Ridge spurs, Red Cliffs, and Hidden Springs area trails. The research task there is not only shorter routing; it is choosing the right day/time.

## Promotion Gates

Before any alternate route replaces a field-menu route, verify these gates from the same regenerated source:

- Legal/date/direction gate: current land-manager rules match the exact run date. Lower Hulls must be on an even on-foot day, Polecat should follow the current clockwise-through-2026 rule unless current signage changes it, Hawkins/Harrow are counter-clockwise, and Around the Mountain is counter-clockwise.
- Access gate: every start has legal, car-visible, capacity-plausible parking. Legal residential road starts are acceptable when field/source checks show the road parking is public/legal, repeatable, and cue-able from the car. Private Strava-derived anchors remain valid private planning anchors, but public artifacts need public-safe labels and must hide exact coordinates and raw activity ids.
- Closure gate: check R2R condition reports/interactive map, Boise National Forest alerts, and Bogus status before Bogus, Stack Rock, Sweet Connie, Shingle, or other closure-prone outings.
- Condition gate: muddy trail use is a hard stop; choose all-weather alternatives or turn around rather than damaging trail.
- Heat/crowd gate: default exposed summer routes to the 6-10 a.m. window when possible; use later starts only when shade/elevation/heat and hard-stop constraints justify it.
- Recording gate: official 2026 proof should use the BTC app. The user reports this workflow is tested and confirmed. Historical Strava remains planning evidence only.
- Build gate: promote only after regenerating route JSON, phone packet, GPX, p75/p90 timing, completion audit, recertification, field-tool audit, and field-route walkthrough from the same route source.

## Challenge-Window Sequencing

- June 18, 2026: do not schedule Sweet Connie, Stack Rock Connector, Pat's, Eastside, Mr. Big, Freddy's Stack Rock, DB Connector, Boise Ridge Road Trail, Ponderosa Pine Overlook, Sinker Creek, or any route needing clean upper Bogus Basin Road access during the weekday closure windows unless same-day Forest Service/R2R/Bogus checks show the closure no longer applies.
- June 19, 2026: treat affected Bogus/Stack/Sweet Connie access as constrained until the relevant closure windows clear. This is also Bogus summer-opening day, so avoid making it the first complex mountain field day unless there is a compelling reason.
- June 20 onward: the Deer Point order should no longer constrain the schedule unless a new alert replaces or extends it. Keep normal day-of alerts, trail status, and condition checks.
- Bogus Mondays/Tuesdays after June 19: acceptable as car-and-go trail days when parking/trails are open. The user does not need Bogus lodge/facilities; closed lodge amenities are not a route blocker unless a plan explicitly depends on them.

## Rejection Criteria

- Reject a split as a primary speed route if it increases elapsed time and has no explicit bailout, heat, water, family-logistics, or foot-mile-management value. Keep slower splits when those bailout/logistics benefits are the point.
- Reject or hold any start that depends on unclear road parking, road shoulders, cat tracks, gated roads, or unsigned access unless field/source evidence confirms legal, repeatable use. Legal residential road starts are acceptable once verified.
- Reject low-savings splits when the second start adds cue risk, private-label risk, or GPX fragmentation; `4C` is the model for a plausible but low-priority split.
- Reject any route that conflicts with day/date/direction rules even if the GPX is mathematically shorter.
- Demote any route with no local timing evidence until field-test or public-report calibration narrows p75/p90 uncertainty.
- Do not promote any alternate from research agenda to field packet unless route source, GPX, phone cues, p75/p90, parking labels, coverage, and field walkthrough all agree.

## Per-Outing Research Agenda

### 1A. West Climb

Current route: West Climb Trailhead; Full Sail Trail, Bob Smylie, Buena Vista Trail, 36th Street Chute. Current card: 3.86 official miles, 7.93 on-foot miles, 128 minutes p75. Strava evidence: 3 of 10 official segments matched, 15 efforts.

Audit result: promising multi-start alternative. Best retained split puts 36th Street Chute in a separate component from the lower 36th Street access area. The audit's private Strava-derived anchor is a valid planning anchor and now has a public-safe label: `Full Sail Trailhead, N 36th St Parking`. The split saves 2.46 on-foot miles, but adds 32 elapsed minutes because of the re-park. BoiseTrails describes 36th Street Chute as a loose/sandy connector with uphill effort; that supports testing downhill/order alternatives rather than blindly preserving the current loop.

Research to do:

- Treat the lower 36th Street / `Full Sail Trailhead, N 36th St Parking` start as an accepted parking/start anchor for the split. The current evidence accepts parking for this analysis.
- Test whether 36th Street Chute is better as its own short downhill/down-and-back outing, especially on hot days or when the goal is reduced foot miles rather than elapsed time.
- Keep the 36th / Full Sail split as a valid optional variant if its bailout, car-access, heat-management, or foot-mile-management value is useful; slower elapsed time alone is not a rejection reason.
- Compare current West Climb start with Hillside Park and Harrison Hollow starts for family-logistics convenience, water/restroom access, and heat exposure.
- Recheck current R2R conditions because Hillside to Hollow has usable sandy/gravel options, but non-Harrison native-surface trails still should not be used muddy.

### 1B. Harrison Hollow

Current route: Harrison Hollow Trailhead; Who Now Loop, Harrison Ridge, Harrison Hollow, Kemper's Ridge Trail, Hippie Shake Trail. Current card: 4.72 official miles, 6.36 on-foot miles, 141 minutes p75. Strava evidence: 1 of 12 official segments matched, 2 efforts.

Audit result: reviewed, no retained split. Current route is compact and already benefits from the Harrison Hollow trailhead amenities. R2R notes Harrison Hollow's sand/gravel surface and short loop options, which supports keeping this as a compact field-test / cue-quality route.

Research to do:

- Keep the main route intact unless a field test finds a confusing cue, because the multi-start math did not find a worthwhile split.
- Re-run the route in the phone packet after any cue-generator change; this is still the regression canary for named access from the car and named return back to Harrison Hollow.
- Use personal Strava/GPS from field tests to calibrate Harrison Ridge / Who Now / Kemper timing, since current matched effort coverage is sparse.
- Check R2R current conditions and dog/parking crowd context before choosing a weekend morning.

### 2. Hulls Gulch / Lower Hulls

Current route: Hulls Gulch Trailhead; Lower Hull's Gulch, Hull's Gulch Interpretive, Crestline, Red Cliffs, Kestral, Owl's Roost, Chickadee Ridge, Gold Finch, 15th St. Trail. Current card: 13.11 official miles, 17.26 on-foot miles, 340 minutes p75. Strava evidence: 2 of 25 official segments matched, 5 efforts.

Audit result: reviewed, no retained split. This is a coherent single-car route but high-exposure and rule-sensitive.

Research to do:

- Verify Lower Hulls odd/even legality against the current R2R interactive map on the exact run date; do not use stale direction/closure assumptions.
- Recheck any Owl's Roost / Grove Loop work or closures before challenge start; the R2R area page had current project language in this area.
- Compare current Hulls start with 8th Street ATV, 9th Street, and Grove starts for reducing deadhead and preserving even-day Lower Hulls legality.
- Treat Red Cliffs and other clay-sensitive pieces as wet-weather blockers; use R2R condition reports on the day.
- Consider early morning only in summer heat; R2R identifies Hulls/Lower Hulls as busy and summer mornings as the coolest window.

### 3. Military Core

Current route: Freestone Creek Trailhead; Military Reserve Connection, Mountain Cove, Central Ridge Trail, Central Ridge Spur, Ridge Crest, Cottonwood Creek, Connection (Eagle Ridge), Eagle Ridge Trail, Elephant Rock Loop, Heroes Trail. Current card: 8.31 official miles, 12.13 on-foot miles, 250 minutes p75. Strava evidence: 4 of 28 official segments matched, 9 efforts.

Audit result: reviewed, no retained split. Current routing is efficient enough in generated math, but wet-weather and trailhead choice matter.

Research to do:

- Compare Freestone, Cottonwood Creek, Mountain Cove, and Bike Park starts using current parking/cue evidence. R2R lists all three designated Military Reserve parking areas.
- Review whether Ridge Crest / Cottonwood / Central Ridge Spur should move to a drier/earlier day; R2R wet-weather guidance marks these as clay-sensitive.
- Check whether Eagle Ridge's asphalt/maintenance-road character makes it a better foul-weather or late-day anchor than the current order.
- Use local Strava efforts for Central/Mountain Cove segments to recalibrate route-finding and moving-time penalties.

### 4A. Bob's / Urban Connector

Current route: Bob's Trailhead; Bob's Trail and Urban Connector. Current card: 2.84 official miles, 4.07 on-foot miles, 97 minutes p75. Strava evidence: no matched official segment efforts.

Audit result: reviewed, no retained split. This is already a compact short outing.

Research to do:

- Keep as a short-route option unless field evidence shows Bob's Trailhead access is worse than Highlands / 8th / Upper Interpretive for the same official credit.
- Research whether this should pair with `15A` Highlands/Connector if the `15A` split from Bob's Trailhead is promoted.
- Field-test for actual pace because local Strava segment coverage is absent.

### 4B. Scott's Trail

Current route: Upper Interpretive Trailhead; Scott's Trail. Current card: 1.05 official miles, 2.01 on-foot miles, 79 minutes p75. Strava evidence: no matched official segment efforts.

Audit result: single-segment/small route, not evaluated by the multi-start split audit.

Research to do:

- Treat as a short fallback / field-tool validation route, not an optimization target.
- Confirm Upper Interpretive parking and access on the day; if 8th Street or Hulls starts are already being used nearby, research whether it is better as a same-day add-on rather than a separate outing.
- Field-test actual door-to-door time because no local Strava match exists.

### 4C. Table Rock / Castle Rock / Tram

Current route: Eagle Rock Park Parking/Trailhead; Shoshone-Paiute, Quarry Trail - Castle Rock, Table Rock Quarry Trail, Table Rock Trail, Tram Trail, Rock Garden, Rock Island. Current card: 6.60 official miles, 11.50 on-foot miles, 264 minutes p75. Strava evidence: no matched official segment efforts.

Audit result: promising in the current checkpoint, but low priority. The retained split saves 0.87 on-foot miles but adds 9 minutes, using Warm Springs Golf Course / Tram side plus a private Strava-derived Castle Rock side anchor. Because it is a Strava endpoint cluster, treat it as prior real parking by the user; the remaining publication issue is a public-safe name for the private anchor. R2R lists Old Pen and Warm Springs Golf Course trailhead parking, and says the top of Table Rock is closed to vehicles while trails remain open sunrise to sunset.

Research to do:

- Decide whether the split has enough benefit to justify two starts; current savings are small, so this is not a high-priority promotion unless the second start has a clear public-safe name.
- Compare Old Pen, Eagle Rock, and Warm Springs Golf Course starts explicitly. Table Rock's wet-weather/clay risk and sunset closure should drive day selection.
- Check current Table Rock top-vehicle-access and mesa hours before any route that assumes summit access or late-day timing.
- Field-test route order around Quarry / Table Rock / Tram because there is no local Strava segment evidence in the current imported data.

### 5. Polecat / Barn Owl

Current route: Cartwright Trailhead; Polecat Loop, Doe Ridge, Quick Draw, Barn Owl. Current card: 7.99 official miles, 13.56 on-foot miles, 282 minutes p75. Strava evidence: no matched official segment efforts.

Audit result: promising in the current checkpoint. Best retained split isolates Barn Owl from the Polecat core, saving 3.97 on-foot miles and 47 minutes, using an accepted user-reviewed West Hidden Springs Drive road-parking anchor plus Cartwright Trailhead. R2R lists Cartwright and Polecat/Collister trailheads, notes construction expected at Cartwright in 2026, says Polecat Loop users travel clockwise through 2026 with short multi-directional access sections, and requires dogs on-leash in the reserve. BoiseTrails reports reinforce Polecat's crowd and mud sensitivity.

Research to do:

- Regenerate the candidate with Polecat clockwise unless same-week R2R signage/map says the rule changed.
- Add a same-week Cartwright construction/access status check; keep Polecat/Collister as the resilient public-trailhead fallback anchor.
- Treat West Hidden Springs Drive as an accepted parking/start anchor from the user-reviewed parking decision. Focus promotion work on route design, Polecat direction, and field cue clarity; the parking anchor is already accepted.
- Compare the split against the simpler public-trailhead option from Polecat/Collister. It may save less than the road probe but avoids residential parking risk.
- Treat Polecat as a bad wet/marginal-condition candidate; use current R2R reports.

### 6. Cartwright / Peggy / Chukar

Current route: Cartwright Trailhead; Peggy's Trail, Chukar Butte Trail, Cartwright Connector, Cartwright Ridge, CHBH Connector. Current card: 13.67 official miles, 21.53 on-foot miles, 448 minutes p75. Strava evidence: 1 of 8 official segments matched, 2 efforts.

Audit result: reviewed, no retained split. Current outing is long and may be more of a field-day pressure problem than a within-outing routing error.

Research to do:

- Re-evaluate after Cartwright trailhead construction status is known; R2R says improvements are expected to start in 2026.
- Test Hidden Springs / Chukar-side alternatives only if parking is publicly verified; avoid private neighborhood assumptions.
- Look for route-order savings with `5` Polecat if the Polecat/Barn Owl split is promoted, because both use Cartwright-area logistics.
- Field-test or Strava-calibrate Peggy/Chukar timing; current matched evidence is too thin for a confident p75.

### 7. Seaman Gulch / Wild Phlox

Current route: Seamans Gulch Trailhead; Seaman Gulch Trail and Wild Phlox Trail. Current card: 2.25 official miles, 3.77 on-foot miles, 127 minutes p75. Strava evidence: no matched official segment efforts.

Audit result: reviewed, no retained split. A private review artifact includes a multi-start sample, but the public audit did not retain it as worth promoting.

Research to do:

- Keep as a compact outing unless parking/crowd pressure makes Veterans or another nearby start more practical.
- R2R best-time guidance flags Seamans as busy in the morning; research whether Friday afternoon/evening or a non-peak window is better if heat allows.
- Field-test actual pace and cue clarity because there is no local Strava segment match.

### 8A. Harris Ridge

Current route: Homestead Trail Access Point; Harris Ridge Trail. Current card: 1.72 official miles, 4.44 on-foot miles, 118 minutes p75. Strava evidence: no matched official segment efforts.

Audit result: single-segment route, not evaluated by the multi-start split audit.

Research to do:

- Prefer a bundle review with `8B` using Oregon Trail Reserve logistics. R2R lists Whitman Trailhead and Kelton Trailhead as paved public anchors, with restrooms plus water/dog-water at Whitman.
- Compare Homestead Trail Access Point with Whitman/Kelton starts where relevant, but preserve public legal parking only.
- Research whether `8A` and `8B` are better as one same-day east-side/Oregon Trail outing or separate short fallback outings.
- Field-test pace, because current timing is generic rather than Strava-backed.

### 8B. Peace Valley Overlook

Current route: Homestead Trail Access Point; Peace Valley Overlook. Current card: 0.54 official miles, 2.70 on-foot miles, 101 minutes p75. Strava evidence: no matched official segment efforts.

Audit result: no evaluated split. Current route is tiny official mileage with high connector ratio.

Research to do:

- Decide whether this should remain a standalone short card or be bundled with `8A` Harris Ridge.
- Use Whitman Trailhead as the default amenities anchor if route geometry allows; Kelton is also paved public parking but has less amenity value.
- Preserve Oregon Trail Reserve leash / historic-preservation constraints in field notes.
- Because official mileage is low, optimize for family/time window and low-friction parking rather than pure official-mile efficiency.

### 9. Veterans / Big Springs / Bike Park

Current route: Veterans Trailhead; Veterans, Big Springs, Rabbit Run, D's Chaos, REI Connection. Current card: 4.68 official miles, 5.78 on-foot miles, 180 minutes p75. Strava evidence: no matched official segment efforts.

Audit result: reviewed, no retained split. Current official-to-on-foot ratio is already strong.

Research to do:

- Keep as-is unless field testing shows Bike Park / Veterans parking or cue flow is awkward.
- Research whether this can be paired same-day with `7` Seaman Gulch as a practical west-side package; R2R lists Veterans/Big Springs/REI/Rabbit Run as alternatives when Seamans is busy.
- Add field-test timing; current local Strava coverage is absent.

### 10A. Harlow / Hidden Springs West

Current route: Harlow's / Hidden Springs west access probe; Harlow's Hollows, Harlow's Hollows Connector, Ricochet, Shooting Range, Spring Creek, Twisted Spring, Whistling Pig. Current card: 7.30 official miles, 13.62 on-foot miles, 360 minutes p75. Strava evidence: no matched official segment efforts. Segment-crosswalk review: all 13 official segments are review-required.

Audit result: high-upside access-validation candidate. Several retained alternatives save 2.67 to 3.38 on-foot miles and 24 to 43 minutes. They use assumed or manual road-parking anchors that are not field-ready yet, but legal residential road starts are acceptable after field/source verification, so this is a verification problem rather than a policy rejection.

Research to do:

- Make this the highest-priority parking/access research area: North Burnt Car Place, West Creeks Edge Drive, North Smokeys Draw Place, Cartwright Road / #20, and the existing Harlow west access all need public/legal/capacity/cue checks.
- Use Google Earth review as a starting point only; it does not make any anchor field-ready.
- Check current R2R map because the 2024 R2R map classifies all Hidden Springs area trails as wet/marginal-condition avoid routes.
- Search public Strava/route reports or manually inspect personal Strava activities for starts that correspond to these anchors without exposing private coordinates.
- If a residential road start passes verification, keep it in the candidate set; do not replace it with a worse designated trailhead solely for optics.
- Do not publish a split until parking and non-credit connector/access cues are certifiable from the car.

### 10B. Bitterbrush / Currant Creek

Current route: Dry Creek Parking Area/Trailhead; Bitterbrush Trail and Currant Creek. Current card: 2.45 official miles, 5.43 on-foot miles, 152 minutes p75. Strava evidence: no matched official segment efforts.

Audit result: reviewed, no retained split.

Research to do:

- Keep paired to Dry Creek unless a public-source alternate start clearly reduces connector burden.
- Research whether `10B`, `15B`, and parts of `16A` should be coordinated as Dry Creek / lower Bogus Basin Road logistics rather than separate mental errands.
- Check R2R conditions, because Dry Creek is a wet/marginal good bet but the higher/lateral connectors may not be.

### 11. Hawkins

Current route: Hawkins Range Reserve Trailhead; Hawkins. Current card: 5.63 official miles, 5.73 on-foot miles, 149 minutes p75. Strava evidence: 1 of 3 official segments matched, 2 efforts.

Audit result: no retained split. This is already a high-efficiency same-car route.

Research to do:

- Verify seasonal opening, dog/leash rules, and directional requirements. R2R says Hawkins access is seasonally closed December 1 to April 30, opened May 8 in 2026 due to muddy conditions, and Hawkins/Harrow are counter-clockwise directional.
- Check the dusk-to-dawn gate timing if this becomes an early or late route.
- Field-test route direction and cue language against signs before relying on generic GPX cues.

### 12. 8th Street / Sidewinder / Corrals

Current route: 8th Street ATV Parking Area; 8th Street Motorcycle Trail, Sidewinder Trail, Corrals Trail. Current card: 7.81 official miles, 12.86 on-foot miles, 262 minutes p75. Strava evidence: 3 of 10 official segments matched, 9 efforts.

Audit result: reviewed, no retained split. Recreation.gov confirms 8th Street Trailhead access to 8th Street Motorcycle, Corrals, Lower Hulls, Crestline, Sidewinder, and Hulls Interpretive, with two parking areas, trailer room, trash, OHV ramps, and a vault toilet.

Research to do:

- Compare current trail order against common Corrals counter-clockwise usage and R2R current conditions.
- Check motorized/non-motorized trail interface and any day-specific conflict concerns.
- Investigate whether this can be made weekday-safe in the p90 schedule by better timing/order; prior proof notes put this route near a weekday-bound threshold.
- Preserve 8th Street as a strong logistics anchor unless a source-verified alternative clearly improves p75.

### 13. Freestone / Three Bears / Shane's / Fat Tire

Current route: Freestone Creek Trailhead; Three Bears, Femrite's Patrol, Freestone Ridge, Two Point, Shane's Trail, Shane's Connector, Fat Tire Traverse, Curlew Connection. Current card: 14.35 official miles, 25.12 on-foot miles, 490 minutes p75. Strava evidence: 7 of 16 official segments matched, 27 efforts.

Audit result: held after corrected audit. The initial pass looked promising,
but that was a generator bug: reverse-order variants with more than three trails
could drop non-reversible ascent-only trails. After fixing that and preserving
Curlew Connection ascent segments, the retained splits are worse than the
current Freestone baseline.

Research to do:

- Keep the current Freestone single-start route unless a future manual design proves full ascent-preserving coverage with lower on-foot and p75 time.
- Keep the Fat Tire / Shane's Strava-derived starts as valid private planning evidence, but do not use them to publish a split from the current audit.
- If revisited, require explicit Curlew ascent preservation before comparing mileage/time.
- Use personal Strava effort history to recalibrate p75; this is one of the better-covered outings and should not rely on generic pace.
- Check wet/marginal condition risk for lower Military / Ridgecrest / clay-adjacent sections before scheduling.

### 14. Orchard / Five Mile / Watchman

Current route: Orchard Gulch Trail Access Point; Orchard Gulch Trail, Five Mile Gulch Trail, Watchman Trail. Current card: 8.45 official miles, 10.74 on-foot miles, 242 minutes p75. Strava evidence: 3 of 6 official segments matched, 11 efforts.

Audit result: reviewed, no retained split. Current official-to-on-foot ratio is strong.

Research to do:

- Keep current route unless field reports suggest a better Rocky Canyon / Watchman start or road connector.
- Use Strava history to calibrate Five Mile / Watchman grade effort; this is one of the better-covered routes.
- R2R says Rocky Canyon area trails can be less crowded during weekdays; treat this as a weekday candidate if heat and water are managed.
- Verify current road/trail access and any fire/closure notes near Rocky Canyon before challenge window.

### 15A. Highlands / Connector / Dry Creek

Current route: MillerGulch Parking Area/Trailhead; Connector, Highlands Trail, Dry Creek Trail. Current card: 9.33 official miles, 18.65 on-foot miles, 363 minutes p75. Strava evidence: 5 of 8 official segments matched, 11 efforts.

Audit result: promising. Retained split puts Dry Creek separate from Highlands/Connector and saves 2.41 on-foot miles and 38 minutes. R2R wet-weather guidance treats Dry Creek as a sandier good bet, while Highlands/Bob's area has different access options.

Research to do:

- Prioritize review of a two-start version: Dry Creek from Dry Creek parking and Highlands/Connector from Bob's or another verified lower foothills start.
- Check whether this split also improves water/bailout by returning to the car between components.
- Compare with `4A` Bob's / Urban Connector to avoid creating duplicate errands around Bob's Trailhead.
- Use local Strava evidence to recalibrate the Dry Creek climb/descent timing before promotion.

### 15B. Red Tail / Landslide

Current route: Dry Creek Parking Area/Trailhead; Red Tail Trail and Landslide. Current card: 4.02 official miles, 4.87 on-foot miles, 148 minutes p75. Strava evidence: no matched official segment efforts.

Audit result: reviewed, no retained split. Current route is already compact.

Research to do:

- Keep as-is unless Dry Creek condition/access changes.
- Consider same-day logistics with `10B` or `16A` only if it reduces total drive/parking friction without turning the day into a random car-hop.
- Field-test or manually inspect Strava activities for this area because current imported official-segment matches are zero.

### 16A-1. Sweet Connie

Current route: Dry Creek / Sweet Connie roadside parking; Sweet Connie Trail. Current card: 6.09 official miles, 12.20 on-foot miles, 249 minutes p75. Strava evidence: no matched official segment efforts.

Audit result: no retained split. Current route is a manual-access single-trail solution.

Research to do:

- Re-verify Dry Creek / Sweet Connie roadside parking before challenge start and after any rain. This is a manual roadside anchor, not a standard trailhead.
- Avoid Sweet Connie on June 18 and before the relevant June 19 Deer Point closure windows clear unless same-day Forest Service/R2R checks say the closure no longer applies.
- Check Sweet Connie trail reports and R2R conditions. R2R marks Sweet Connie as a wet/marginal avoid trail; BoiseTrails history also shows spring mud/maintenance concerns.
- Compare whether Sweet Connie should run as its own focused day or pair with nearby Dry Creek/Bitterbrush work only if water/heat makes sense.
- Field-test timing; current Strava official-segment matches are absent.

### 16A-2. Sheep Camp / Shingle Creek

Current route: Dry Creek / Sweet Connie roadside parking; Sheep Camp Trail and Shingle Creek Trail. Current card: 5.53 official miles, 14.96 on-foot miles, 310 minutes p75. Strava evidence: no matched official segment efforts.

Audit result: reviewed, no retained split. Prior p90 proof work identifies Shingle Creek as the hardest time/access blocker, not a simple routing mistake.

Research to do:

- Keep this as a special research item. It needs real access/time breakthrough evidence, not another synthetic shortcut.
- Re-check lower Shingle / USFS / road access sources and any field-visible gates, but do not add a connector unless it is public, legal, and physically real.
- Use a field test to calibrate Shingle / Sheep Camp moving time and route-finding burden.
- Treat wet/seasonal conditions conservatively; this area can become a route-damage and time-risk problem.

### 16B. Stack Rock Connector

Current route: Freddy's Stack Rock Trailhead; Stack Rock Connector. Current card: 3.50 official miles, 4.39 on-foot miles, 131 minutes p75. Strava evidence: no matched official segment efforts.

Audit result: small route, no retained split.

Research to do:

- Keep as-is unless Bogus/Stack Rock access conditions suggest pairing it with a Bogus day.
- Avoid June 18 and pre-clearance June 19 because Stack Rock Trailhead / Freddy's Stack Rock access is part of the Deer Point closure reporting. After June 19, verify Forest Service/R2R/Bogus status before treating it as runnable.
- Verify road access, snow, and forested trail condition before scheduling. R2R heat guidance favors higher/forested trails for later starts, but access must still be current.
- Field-test if possible because current local Strava coverage is absent.

### 17. Bogus: Sunshine / Deer Point / ATM / Face / Elk Meadows

Current route: Simplot Lodge Parking Area; Sunshine XC, Deer Point Trail, Around the Mountain Trail, The Face Trail, Elk Meadows Trail. Current card: 11.29 official miles, 15.13 on-foot miles, 388 minutes p75. Strava evidence: no matched official segment efforts.

Audit result: held after corrected audit. The initial Simplot/Pioneer split
looked like the top promote candidate, but it depended on the same ascent-trail
dropping bug. Once Sunshine and Around the Mountain ascent-only segments are
preserved, the retained splits add on-foot mileage and elapsed time. R2R still
lists Bogus parking at Nordic Lodge and Simplot Lodge, describes cooler
temperatures and challenging forested hikes, and requires Around the Mountain
counter-clockwise. Bogus status and the Forest Service Deer Point order still
make June 18 and part of June 19 poor launch-day choices for affected
Bogus/Stack access.

Research to do:

- Keep the current Simplot single-start route for now; do not promote the split from this audit.
- Treat June 18 and pre-clearance June 19 as closure-constrained for affected Deer Point / upper Bogus access; prefer June 20 or later unless a same-day alert check is clean.
- Verify Simplot, Nordic, and Pioneer parking rules. Per project policy, Bogus alternatives should use known public/lodge parking anchors rather than road shoulders or cat tracks.
- Confirm Around the Mountain direction from current signage/R2R/Bogus, even though historical/local reports strongly point counter-clockwise.
- Revisit a Deer Point / The Face / Elk Meadows split only if a future manual design preserves Sunshine/ATM ascent coverage and improves p75/on-foot mileage.
- On non-holiday Mondays/Tuesdays, closed lodge amenities are not a blocker because the user does not need Bogus lodge/facility support. Still verify trails and lots are open.
- Search public trail reports for snow/deadfall close to the challenge window because imported local Strava match coverage is zero.

### 18. Bogus: Brewers / Shindig / Tempest / Lodge / Mores

Current route: Pioneer Lodge Parking Area; Brewer's Byway Extension, Brewers Byway, Shindig, Tempest Trail, Lodge Trail, Mores Mtn Interpretive. Current card: 5.08 official miles, 11.25 on-foot miles, 320 minutes p75. Strava evidence: no matched official segment efforts.

Audit result: reviewed, no retained split. Some private parking-review artifacts cover road/cat-track probes, but current policy rejects them for promotion unless they resolve to known lodge/trailhead parking.

Research to do:

- Keep Pioneer Lodge as the default unless a current Bogus source verifies another official parking anchor.
- Check Bogus trail status and summer opening before treating this as runnable; current Bogus report has winter-season closure / summer-prep language and summer reopening points to June 19.
- On non-holiday Mondays/Tuesdays, treat this as car-and-go only if trails and lots are open; closed lodge amenities are not a blocker because the user does not need Bogus lodge/facility support.
- Review whether any pieces should pair with `17` after the `17` split research, but avoid road-shoulder/cat-track parking.
- Use public BoiseTrails reports for Shindig/Lodge/Mores only as supplemental color; field status should come from Bogus/R2R close to the run.

### 19. Cervidae Peak

Current route: Cervidae / Arrow Rock Road OSM Parking; Cervidae Peak. Current card: 2.24 official miles, 4.51 on-foot miles, 181 minutes p75. Strava evidence: no matched official segment efforts. Segment-crosswalk review: one review-required official segment.

Audit result: single-segment route, not evaluated by the multi-start split audit.

Research to do:

- Verify Arrow Rock Road parking/access in the field or from an authoritative source; do not rely on generic OSM parking alone for final field use.
- Treat this as heat/exposure sensitive; schedule early with explicit water.
- Field-test actual moving time because no local official-segment Strava match exists.
- Confirm official segment geometry and ascent/direction evidence before challenge credit; this is a singleton, so a partial traversal would waste the outing.

## Suggested Research Order

1. Treat `1A`, `4C`, `5`, and `15A` as the promoted certified replacements in the regenerated field packet.
2. Run access-verification work for `10A` and `19`. If `10A` residential/road anchors prove public, legal, repeatable, and cue-able, promote the best `10A` split into a future candidate-design pass.
3. Keep `13` and `17` on the no-promote list unless a future ascent-preserving manual design beats the current baseline.
4. Add a schedule/status gate before schedule locking: BTC dates, R2R current conditions, USFS Deer Point alerts, Bogus trail/facility status, and Hawkins/Polecat/Hulls/ATM directional rules.
5. Timing calibration for no-Strava or low-Strava outings: `4A`, `4B`, `4C`, `5`, `7`, `8A`, `8B`, `9`, `10A`, `10B`, `15B`, `16A-1`, `16A-2`, `16B`, `17`, `18`, `19`.
6. Condition-sensitive day assignment for Hulls/Table Rock/Polecat/Hidden Springs/Sweet Connie/Shingle and Bogus.
7. For any future replacement, repeat the same source-to-GPX-to-cue certification chain used here.

## Official Map Update Decision

Canonical official map route lines have been replaced for the certified `1A`,
`4C`, `5`, and `15A` alternatives. The current executable field packet is the
regenerated packet from
`years/2026/outputs/private/2026-outing-menu-map-data.json`, with public and
phone artifacts regenerated from the same source. The detailed update record is
`years/2026/checkpoints/official-map-update-recommendation-2026-05-08.md`.

## Settled User Decisions

- Slower split variants are acceptable when they provide useful bailouts, car access, heat management, foot-mile reduction, or family/work logistics.
- Legal residential road starts are acceptable after field/source verification; a designated trailhead is not required when the residential start is legal, repeatable, and cue-able.
- Bogus lodge/facilities are not required for Bogus outings. Closed lodge amenities should not block a route when trails and parking are open.
- The user will use the BTC app directly for official recording, and that workflow is tested and confirmed.

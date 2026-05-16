# Runner-Perspective Frame Shift: 4A - Bob's

## Frame Contract

- Route card: `4A` / outing `4-3`.
- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.
- Evidence used: `docs/field-packet/field-tool-data.json`, `docs/field-packet/gpx/official/4a-bob-s-bob-s-trail-urban-connector.gpx`, R2R open data, OSM connector overlay, official 2026 segment source.
- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.
- Frame decision: `needs-proof`. The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.
- Access status: known-or-mapped parking in packet data.
- Human-validity status for this audit: `needs_visual_proof`.

## Route Snapshot

- Trails: Bob's Trail, Urban Connector.
- Official miles: 2.84; on-foot miles: 4.07.
- Door-to-door: p75 97 min; p90 109 min.
- Segment count: 4; wayfinding cue count: 5.

## Start-End-Junction Frame Shifts

### Start

- Physical role: Park here and start this outing.
- Model frame: The packet proves the route has a start coordinate and a first cue.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: #30 Bob's, #1 Highlands Trail, OSM footway connector 73045, OSM path connector 11370, OSM footway connector 73044; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Bob's ~0m; #30 Bob's (path) ~1m; #1 Highlands Trail (path) ~7m; Highlands ~19m; OSM footway connector 73045 (footway) ~20m; OSM path connector 11370 (path) ~70m
- Decision as runner: Before moving, find the first signed trail or road-access line and confirm the car is parked where the return cue can actually resolve.
- Wrong-layer risk: start/finish access can fail even when route geometry passes; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: cue GPX waypoint plus local R2R/OSM overlay; no live imagery or field photo in this audit

### Cue 01: FOLLOW #30 Bob's

- Physical role: signed junction with Bob's Trail
- Model frame: The packet says `01 0.00 mi (+0.00) START/ACCESS FOLLOW #30 Bob's UNTIL signed junction with Bob's Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#1 Highlands Trail, OSM footway connector 73045, OSM path connector 11370, OSM footway connector 73044, OSM footway connector 51185); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #30 Bob's, #1 Highlands Trail, OSM footway connector 73045, OSM path connector 11370, OSM footway connector 73044; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#30 Bob's` until `signed junction with Bob's Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #30 Bob's (path) ~5m; #1 Highlands Trail (path) ~6m; Bob's ~6m; Highlands ~14m; OSM footway connector 73045 (footway) ~21m; OSM path connector 11370 (path) ~76m
- Decision as runner: Follow #30 Bob's until signed junction with Bob's Trail; target is Bob's Trail.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 02: FOLLOW Bob's Trail

- Physical role: signed junction with Urban Connector
- Model frame: The packet says `02 0.00 mi (+1.59) FOLLOW Bob's Trail UNTIL signed junction with Urban Connector.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#30 Bob's, #1 Highlands Trail, OSM footway connector 73045, OSM path connector 11370, OSM footway connector 73044); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #30 Bob's, #1 Highlands Trail, OSM footway connector 73045, OSM path connector 11370, OSM footway connector 73044; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Bob's Trail` until `signed junction with Urban Connector`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #30 Bob's (path) ~5m; #1 Highlands Trail (path) ~6m; Bob's ~6m; Highlands ~14m; OSM footway connector 73045 (footway) ~21m; OSM path connector 11370 (path) ~76m
- Decision as runner: Follow Bob's Trail until signed junction with Urban Connector; target is Urban Connector.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch; wrong-direction choice has meaningful climb penalty
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 03: FOLLOW #1 Highlands Trail / #30 Bob's / Bob's / Highlands

- Physical role: signed junction with Urban Connector
- Model frame: The packet says `03 1.59 mi (+0.66) CONNECTOR FOLLOW #1 Highlands Trail / #30 Bob's / Bob's / Highlands UNTIL signed junction with Urban Connector.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#9 Urban Connector Trail, OSM path connector 110796); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #30 Bob's, #9 Urban Connector Trail, OSM path connector 110796; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#1 Highlands Trail / #30 Bob's / Bob's / Highlands` until `signed junction with Urban Connector`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Bob's ~5m; #30 Bob's (path) ~6m; #9 Urban Connector Trail (path) ~43m; Urban Connector ~43m; Sideshow ~119m; OSM path connector 110796 (path) ~132m
- Decision as runner: Follow #1 Highlands Trail / #30 Bob's / Bob's / Highlands until signed junction with Urban Connector; target is Urban Connector.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 04: TAKE Urban Connector

- Physical role: end of Urban Connector for this route
- Model frame: The packet says `04 2.25 mi (+1.24) JCT TAKE Urban Connector UNTIL end of Urban Connector for this route.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#30 Bob's, #9 Urban Connector Trail, OSM path connector 110796); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #30 Bob's, #9 Urban Connector Trail, OSM path connector 110796; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Urban Connector` until `end of Urban Connector for this route`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Bob's ~5m; #30 Bob's (path) ~6m; #9 Urban Connector Trail (path) ~43m; Urban Connector ~43m; Sideshow ~119m; OSM path connector 110796 (path) ~132m
- Decision as runner: Follow Urban Connector until end of Urban Connector for this route; target is return to car.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 05: FOLLOW #30 Bob's / Bob's

- Physical role: parked car / trailhead
- Model frame: The packet says `05 3.49 mi (+0.47) EXIT FOLLOW #30 Bob's / Bob's UNTIL parked car / trailhead.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#9 Urban Connector Trail, OSM path connector 110796); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #30 Bob's, #9 Urban Connector Trail, OSM path connector 110796; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#30 Bob's / Bob's` until `parked car / trailhead`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Bob's ~0m; #30 Bob's (path) ~2m; Urban Connector ~13m; #9 Urban Connector Trail (path) ~14m; OSM path connector 110796 (path) ~115m; 8th Street Connection ~124m
- Decision as runner: Follow #30 Bob's / Bob's until parked car / trailhead; target is Bob's Trailhead.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Finish / return to car

- Physical role: Route endpoint / return-to-car point.
- Model frame: The packet endpoint closes the loop.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: #30 Bob's, #1 Highlands Trail, OSM footway connector 73045, OSM path connector 11370, OSM footway connector 73044; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #30 Bob's (path) ~5m; #1 Highlands Trail (path) ~6m; Bob's ~6m; Highlands ~14m; OSM footway connector 73045 (footway) ~21m; OSM path connector 11370 (path) ~76m
- Decision as runner: Do not stop the BTC recording early; finish the actual return-to-car leg and then save/upload according to the BTC workflow.
- Wrong-layer risk: start/finish access can fail even when route geometry passes; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: cue GPX return waypoint plus local R2R/OSM overlay; no official completion proof without the eventual activity geometry

## Whole-Route Frame Shift

- Original frame: a route card can be valid because coverage, GPX continuity, and cue exports exist.
- Runner frame: the route is only executable if each visible branch, road edge, return leg, overlap, and indistinct connector can be recognized while tired and moving.
- Adjacent frame checked: literal sightline proof from imagery or field photos. This audit does not have that proof, so it keeps sightline claims bounded.
- Adjacent frame checked: route-card certification. This audit reads route-card cues but does not rerun the full certification chain.
- Adjacent frame checked: day-of legality. This audit does not replace current Ridge to Rivers condition/signage checks.

## Adversarial Failure Stories

- The GPX line is correct, but the runner starts on the first official-credit label instead of the actual access leg from the car.
- A side trail or road line near a cue looks plausible in the distance, and the runner follows the visual line instead of the signed cue/active leg.
- A generic OSM connector or repeat leg has no field sign, so the runner needs the active GPX leg rather than a sign name.
- Overlap or double-back geometry causes the map to look like one line, hiding the fact that the active direction changed.
- The route answers the coverage question but misses the day-of condition question: closure, mud, heat, or signage can still block execution.

## Route-Line Nearby Features

- Bob's (~0m, feature, source=ridge_to_rivers_open_data)
- #30 Bob's (~1m, path, highway=path, source=openstreetmap)
- #1 Highlands Trail (~7m, path, highway=path, source=openstreetmap)
- Highlands (~19m, feature, source=ridge_to_rivers_open_data)
- OSM footway connector 73045 (~20m, path, highway=footway, source=openstreetmap)
- #9 Urban Connector Trail (~43m, path, highway=path, source=openstreetmap)
- Urban Connector (~43m, feature, source=ridge_to_rivers_open_data)
- OSM path connector 11370 (~70m, path, highway=path, source=openstreetmap)
- OSM footway connector 73044 (~81m, path, highway=footway, source=openstreetmap)
- OSM footway connector 51185 (~96m, path, highway=footway, source=openstreetmap)
- Sideshow (~119m, feature, source=ridge_to_rivers_open_data)
- OSM path connector 110796 (~132m, path, highway=path, source=openstreetmap)
- 8th Street Connection (~139m, feature, source=ridge_to_rivers_open_data)

## Required Next Proof

- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.
- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.
- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.

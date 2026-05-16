# Runner-Perspective Frame Shift: 12 - 8th Street ATV Parking Area

## Frame Contract

- Route card: `12` / outing `12-1`.
- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.
- Evidence used: `docs/field-packet/field-tool-data.json`, `docs/field-packet/gpx/official/12-8th-street-atv-parking-area-8th-street-motorcycle-trail-sidewinder-trail-corrals-trail.gpx`, R2R open data, OSM connector overlay, official 2026 segment source.
- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.
- Frame decision: `needs-proof`. The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.
- Access status: known-or-mapped parking in packet data.
- Human-validity status for this audit: `needs_visual_proof`.

## Route Snapshot

- Trails: 8th Street Motorcycle Trail, Sidewinder Trail, Corrals Trail.
- Official miles: 7.81; on-foot miles: 12.86.
- Door-to-door: p75 262 min; p90 294 min.
- Segment count: 10; wayfinding cue count: 7.

## Start-End-Junction Frame Shifts

### Start

- Physical role: Park here and start this outing.
- Model frame: The packet proves the route has a start coordinate and a first cue.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: #31 Corrals, #4 8th Street Motorcycle, #0 Hull's Gulch Interpretive Trail; vehicle corridor or service/residential road context: OSM service connector 12944, OSM service connector 12946, OSM service connector 12945, East Sunset Peak Road; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 12944 (service) ~2m; OSM service connector 12946 (service) ~3m; OSM service connector 12945 (service) ~6m; East Sunset Peak Road (unclassified) ~37m; #31 Corrals (path) ~40m; Corrals ~44m
- Decision as runner: Before moving, find the first signed trail or road-access line and confirm the car is parked where the return cue can actually resolve.
- Wrong-layer risk: start/finish access can fail even when route geometry passes
- Evidence boundary: cue GPX waypoint plus local R2R/OSM overlay; no live imagery or field photo in this audit

### Cue 01: FOLLOW #4 8th Street Motorcycle

- Physical role: signed junction with 8th Street Motorcycle Trail
- Model frame: The packet says `01 0.00 mi (+0.02) START/ACCESS FOLLOW #4 8th Street Motorcycle UNTIL signed junction with 8th Street Motorcycle Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#31 Corrals, #0 Hull's Gulch Interpretive Trail); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #31 Corrals, #4 8th Street Motorcycle, #0 Hull's Gulch Interpretive Trail; vehicle corridor or service/residential road context: OSM service connector 12944, OSM service connector 12945, OSM service connector 12946, East Sunset Peak Road; the branch to privilege is `#4 8th Street Motorcycle` until `signed junction with 8th Street Motorcycle Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 12944 (service) ~1m; OSM service connector 12945 (service) ~2m; OSM service connector 12946 (service) ~6m; East Sunset Peak Road (unclassified) ~38m; #31 Corrals (path) ~43m; #4 8th Street Motorcycle (path) ~46m
- Decision as runner: Follow #4 8th Street Motorcycle until signed junction with 8th Street Motorcycle Trail; target is 8th Street Motorcycle Trail.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 02: FOLLOW 8th Street Motorcycle Trail

- Physical role: signed junction with Sidewinder Trail
- Model frame: The packet says `02 0.02 mi (+1.37) FOLLOW 8th Street Motorcycle Trail UNTIL signed junction with Sidewinder Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#4 8th Street Motorcycle, #31 Corrals); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #4 8th Street Motorcycle, #31 Corrals; vehicle corridor or service/residential road context: OSM service connector 12946, OSM service connector 12944, OSM service connector 12945, East Sunset Peak Road; the branch to privilege is `8th Street Motorcycle Trail` until `signed junction with Sidewinder Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 12946 (service) ~2m; #4 8th Street Motorcycle (path) ~22m; OSM service connector 12944 (service) ~22m; OSM service connector 12945 (service) ~23m; 8th Street Motorcycle ~31m; East Sunset Peak Road (unclassified) ~31m
- Decision as runner: Follow 8th Street Motorcycle Trail until signed junction with Sidewinder Trail; target is Sidewinder Trail.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 03: FOLLOW #4 8th Street Motorcycle / 8th Street Motorcycle

- Physical role: signed junction with Sidewinder Trail
- Model frame: The packet says `03 1.39 mi (+0.13) CONNECTOR FOLLOW #4 8th Street Motorcycle / 8th Street Motorcycle UNTIL signed junction with Sidewinder Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#42 Fat Tire Traverse, #24 Sidewinder); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #4 8th Street Motorcycle, #42 Fat Tire Traverse, #24 Sidewinder; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#4 8th Street Motorcycle / 8th Street Motorcycle` until `signed junction with Sidewinder Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #4 8th Street Motorcycle (path) ~0m; 8th Street Motorcycle ~0m; Fat Tire Traverse ~41m; #42 Fat Tire Traverse (path) ~42m; #24 Sidewinder (path) ~141m; Sidewinder ~142m
- Decision as runner: Follow #4 8th Street Motorcycle / 8th Street Motorcycle until signed junction with Sidewinder Trail; target is Sidewinder Trail.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 04: BEAR RIGHT Sidewinder Trail

- Physical role: signed junction with Corrals Trail
- Model frame: The packet says `04 1.52 mi (+1.34) JCT BEAR RIGHT Sidewinder Trail UNTIL signed junction with Corrals Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#4 8th Street Motorcycle, #24 Sidewinder); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #4 8th Street Motorcycle, #24 Sidewinder; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Sidewinder Trail` until `signed junction with Corrals Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #4 8th Street Motorcycle (path) ~0m; 8th Street Motorcycle ~1m; #24 Sidewinder (path) ~20m; Sidewinder ~21m
- Decision as runner: Follow Sidewinder Trail until signed junction with Corrals Trail; target is Corrals Trail.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 05: FOLLOW #28 Crestline / #31 Corrals / #4 8th Street Motorcycle / 8th Street Motorcycle / Connector Trail / Corrals / Crestline / East Sunset Peak Road / Hull's Gulch Interpretive Trail / OSM service connector 12944 / OSM service connector 12946 / Sideshow

- Physical role: signed junction with Corrals Trail
- Model frame: The packet says `05 2.86 mi (+1.65) ROAD FOLLOW #28 Crestline / #31 Corrals / #4 8th Street Motorcycle / 8th Street Motorcycle / Connector Trail / Corrals / Crestline / East Sunset Peak Road / Hull's Gulch Interpretive Trail / OSM service connector 12944 / OSM service connector 12946 / Sideshow UNTIL signed junction with Corrals Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#24 Sidewinder); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #24 Sidewinder, #28 Crestline; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#28 Crestline / #31 Corrals / #4 8th Street Motorcycle / 8th Street Motorcycle / Connector Trail / Corrals / Crestline / East Sunset Peak Road / Hull's Gulch Interpretive Trail / OSM service connector 12944 / OSM service connector 12946 / Sideshow` until `signed junction with Corrals Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #24 Sidewinder (path) ~0m; Sidewinder ~0m; #28 Crestline (path) ~22m; Crestline ~22m
- Decision as runner: Follow #28 Crestline / #31 Corrals / #4 8th Street Motorcycle / 8th Street Motorcycle / Connector Trail / Corrals / Crestline / East Sunset Peak Road / Hull's Gulch Interpretive Trail / OSM service connector 12944 / OSM service connector 12946 / Sideshow until signed junction with Corrals Trail; target is Corrals Trail.
- Wrong-layer risk: generic OSM connector name may not exist on signs
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 06: TAKE Corrals Trail

- Physical role: end of Corrals Trail for this route
- Model frame: The packet says `06 4.51 mi (+5.09) JCT TAKE Corrals Trail UNTIL end of Corrals Trail for this route.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#31 Corrals, #4 8th Street Motorcycle); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #31 Corrals, #4 8th Street Motorcycle; vehicle corridor or service/residential road context: OSM service connector 12946, OSM service connector 12944, OSM service connector 12945, East Sunset Peak Road; the branch to privilege is `Corrals Trail` until `end of Corrals Trail for this route`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 12946 (service) ~4m; OSM service connector 12944 (service) ~12m; OSM service connector 12945 (service) ~14m; East Sunset Peak Road (unclassified) ~25m; #31 Corrals (path) ~34m; Corrals ~34m
- Decision as runner: Follow Corrals Trail until end of Corrals Trail for this route; target is return to car.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 07: FOLLOW #1 Highlands Trail / #30 Bob's / #31 Corrals / #4 8th Street Motorcycle / 8th Street Connection / 8th Street Motorcycle / Bob's / Corrals / East Sunset Peak Road / Highlands / OSM service connector 12944 / OSM service connector 12946

- Physical role: parked car / trailhead
- Model frame: The packet says `07 9.60 mi (+4.60) EXIT FOLLOW #1 Highlands Trail / #30 Bob's / #31 Corrals / #4 8th Street Motorcycle / 8th Street Connection / 8th Street Motorcycle / Bob's / Corrals / East Sunset Peak Road / Highlands / OSM service connector 12944 / OSM service connector 12946 UNTIL parked car / trailhead.`.
- Runner frame: Runner frame: the immediate job is to keep the current trail until the named junction/landmark, with no extra branch proven by local data at this checkpoint.
- Likely visual field: mapped trail/path choices near you: #31 Corrals, #30 Bob's; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#1 Highlands Trail / #30 Bob's / #31 Corrals / #4 8th Street Motorcycle / 8th Street Connection / 8th Street Motorcycle / Bob's / Corrals / East Sunset Peak Road / Highlands / OSM service connector 12944 / OSM service connector 12946` until `parked car / trailhead`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #31 Corrals (path) ~1m; Corrals ~1m; Bob's ~13m; #30 Bob's (path) ~14m
- Decision as runner: Follow #1 Highlands Trail / #30 Bob's / #31 Corrals / #4 8th Street Motorcycle / 8th Street Connection / 8th Street Motorcycle / Bob's / Corrals / East Sunset Peak Road / Highlands / OSM service connector 12944 / OSM service connector 12946 until parked car / trailhead; target is 8th Street ATV Parking Area.
- Wrong-layer risk: generic OSM connector name may not exist on signs
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Finish / return to car

- Physical role: Route endpoint / return-to-car point.
- Model frame: The packet endpoint closes the loop.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: #31 Corrals, #4 8th Street Motorcycle, #0 Hull's Gulch Interpretive Trail; vehicle corridor or service/residential road context: OSM service connector 12944, OSM service connector 12945, OSM service connector 12946, East Sunset Peak Road; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 12944 (service) ~1m; OSM service connector 12945 (service) ~2m; OSM service connector 12946 (service) ~6m; East Sunset Peak Road (unclassified) ~38m; #31 Corrals (path) ~43m; #4 8th Street Motorcycle (path) ~46m
- Decision as runner: Do not stop the BTC recording early; finish the actual return-to-car leg and then save/upload according to the BTC workflow.
- Wrong-layer risk: start/finish access can fail even when route geometry passes
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

- OSM service connector 12944 (~2m, vehicle, highway=service, source=openstreetmap)
- OSM service connector 12946 (~3m, vehicle, highway=service, source=openstreetmap)
- OSM service connector 12945 (~6m, vehicle, highway=service, source=openstreetmap)
- Bob's (~13m, feature, source=ridge_to_rivers_open_data)
- #30 Bob's (~14m, path, highway=path, source=openstreetmap)
- #28 Crestline (~22m, path, highway=path, source=openstreetmap)
- Crestline (~22m, feature, source=ridge_to_rivers_open_data)
- 8th Street Motorcycle (~31m, feature, source=ridge_to_rivers_open_data)
- East Sunset Peak Road (~37m, vehicle, highway=unclassified, source=openstreetmap)
- #31 Corrals (~40m, path, highway=path, source=openstreetmap)
- Fat Tire Traverse (~41m, feature, source=ridge_to_rivers_open_data)
- #42 Fat Tire Traverse (~42m, path, highway=path, source=openstreetmap)
- Corrals (~44m, feature, source=ridge_to_rivers_open_data)
- #4 8th Street Motorcycle (~49m, path, highway=path, source=openstreetmap)
- #0 Hull's Gulch Interpretive Trail (~50m, path, highway=path, source=openstreetmap)
- #24 Sidewinder (~141m, path, highway=path, source=openstreetmap)
- Sidewinder (~142m, feature, source=ridge_to_rivers_open_data)

## Required Next Proof

- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.
- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.
- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.

# Runner-Perspective Frame Shift: 5B - Cartwright

## Frame Contract

- Route card: `5B` / outing `5-2`.
- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.
- Evidence used: `docs/field-packet/field-tool-data.json`, `docs/field-packet/gpx/official/5b-cartwright-polecat-loop-quick-draw-doe-ridge.gpx`, R2R open data, OSM connector overlay, official 2026 segment source.
- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.
- Frame decision: `needs-proof`. The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.
- Access status: parking evidence incomplete in packet data.
- Human-validity status for this audit: `needs_visual_proof`.

## Route Snapshot

- Trails: Polecat Loop, Quick Draw, Doe Ridge.
- Official miles: 6.56; on-foot miles: 7.3.
- Door-to-door: p75 163 min; p90 183 min.
- Segment count: 9; wayfinding cue count: 7.

## Start-End-Junction Frame Shifts

### Start

- Physical role: Park here and start this outing.
- Model frame: The packet proves the route has a start coordinate and a first cue.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Vehicle movement may be audible or visible near OSM service connector 69243, North Cartwright Road, OSM track connector 12641; do not mistake the road/driveway line for the trail branch.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: #76 Peggy's, #81 Polecat Loop; vehicle corridor or service/residential road context: OSM service connector 69243, North Cartwright Road, OSM track connector 12641; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 69243 (service) ~2m; Polecat Loop ~7m; North Cartwright Road (tertiary) ~22m; #76 Peggy's (path) ~31m; Peggy's ~34m; #81 Polecat Loop (path) ~66m
- Decision as runner: Before moving, find the first signed trail or road-access line and confirm the car is parked where the return cue can actually resolve.
- Wrong-layer risk: start/finish access can fail even when route geometry passes
- Evidence boundary: cue GPX waypoint plus local R2R/OSM overlay; no live imagery or field photo in this audit

### Cue 01: FOLLOW #81 Polecat Loop

- Physical role: signed #81 Polecat Loop route / first official segment
- Model frame: The packet says `01 0.00 mi (+0.01) OFFICIAL START FOLLOW #81 Polecat Loop UNTIL signed #81 Polecat Loop route / first official segment.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#76 Peggy's); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #76 Peggy's, #81 Polecat Loop; vehicle corridor or service/residential road context: OSM service connector 69243, North Cartwright Road, OSM track connector 12641; the branch to privilege is `#81 Polecat Loop` until `signed #81 Polecat Loop route / first official segment`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 69243 (service) ~3m; Polecat Loop ~7m; North Cartwright Road (tertiary) ~23m; #76 Peggy's (path) ~34m; Peggy's ~35m; #81 Polecat Loop (path) ~66m
- Decision as runner: Follow #81 Polecat Loop until signed #81 Polecat Loop route / first official segment; target is #81 Polecat Loop.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 02: FOLLOW #81 Polecat Loop

- Physical role: signed junction with Quick Draw
- Model frame: The packet says `02 0.01 mi (+5.63) FOLLOW #81 Polecat Loop UNTIL signed junction with Quick Draw.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#76 Peggy's); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #76 Peggy's, #81 Polecat Loop; vehicle corridor or service/residential road context: OSM service connector 69243, North Cartwright Road, OSM track connector 12641; the branch to privilege is `#81 Polecat Loop` until `signed junction with Quick Draw`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Polecat Loop ~1m; OSM service connector 69243 (service) ~16m; North Cartwright Road (tertiary) ~36m; #76 Peggy's (path) ~45m; Peggy's ~49m; #81 Polecat Loop (path) ~52m
- Decision as runner: Follow #81 Polecat Loop until signed junction with Quick Draw; target is Quick Draw.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 03: FOLLOW #82 Doe Ridge / Doe Ridge

- Physical role: signed junction with Quick Draw
- Model frame: The packet says `03 5.64 mi (+0.29) CONNECTOR FOLLOW #82 Doe Ridge / Doe Ridge UNTIL signed junction with Quick Draw.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#81 Polecat Loop); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #81 Polecat Loop; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#82 Doe Ridge / Doe Ridge` until `signed junction with Quick Draw`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #81 Polecat Loop (path) ~3m; Polecat Loop (STM) ~3m; Polecat Loop ~31m
- Decision as runner: Follow #82 Doe Ridge / Doe Ridge until signed junction with Quick Draw; target is Quick Draw.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 04: TURN RIGHT Quick Draw

- Physical role: signed junction with Doe Ridge
- Model frame: The packet says `04 5.93 mi (+0.48) JCT TURN RIGHT Quick Draw UNTIL signed junction with Doe Ridge.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#81 Polecat Loop, #83 Quick Draw); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #81 Polecat Loop, #83 Quick Draw; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Quick Draw` until `signed junction with Doe Ridge`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #81 Polecat Loop (path) ~0m; Polecat Loop ~0m; Polecat Loop (STM) ~48m; Quick Draw ~48m; #83 Quick Draw (path) ~50m
- Decision as runner: Follow Quick Draw until signed junction with Doe Ridge; target is Doe Ridge.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 05: FOLLOW #83 Quick Draw / Polecat Loop (STM) / Quick Draw

- Physical role: signed junction with Doe Ridge
- Model frame: The packet says `05 6.41 mi (+0.26) CONNECTOR FOLLOW #83 Quick Draw / Polecat Loop (STM) / Quick Draw UNTIL signed junction with Doe Ridge.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#81 Polecat Loop, OSM path connector 13996); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #81 Polecat Loop, #83 Quick Draw, OSM path connector 13996; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#83 Quick Draw / Polecat Loop (STM) / Quick Draw` until `signed junction with Doe Ridge`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Polecat Loop (STM) ~2m; #81 Polecat Loop (path) ~45m; #83 Quick Draw (path) ~45m; Quick Draw ~46m; Polecat Loop ~47m; OSM path connector 13996 (path) ~164m
- Decision as runner: Follow #83 Quick Draw / Polecat Loop (STM) / Quick Draw until signed junction with Doe Ridge; target is Doe Ridge.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 06: TAKE Doe Ridge

- Physical role: end of Doe Ridge for this route
- Model frame: The packet says `06 6.67 mi (+0.46) JCT TAKE Doe Ridge UNTIL end of Doe Ridge for this route.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#81 Polecat Loop, #82 Doe Ridge); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #81 Polecat Loop, #82 Doe Ridge; vehicle corridor or service/residential road context: OSM track connector 106931; the branch to privilege is `Doe Ridge` until `end of Doe Ridge for this route`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Polecat Loop (STM) ~1m; Polecat Loop ~24m; #81 Polecat Loop (path) ~25m; #82 Doe Ridge (path) ~41m; Doe Ridge ~41m; OSM track connector 106931 (track) ~169m
- Decision as runner: Follow Doe Ridge until end of Doe Ridge for this route; target is return to car.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 07: FOLLOW #81 Polecat Loop / Polecat Loop

- Physical role: parked car / trailhead
- Model frame: The packet says `07 7.13 mi (+0.31) EXIT FOLLOW #81 Polecat Loop / Polecat Loop UNTIL parked car / trailhead.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#82 Doe Ridge); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #82 Doe Ridge, #81 Polecat Loop; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#81 Polecat Loop / Polecat Loop` until `parked car / trailhead`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Polecat Loop ~0m; #82 Doe Ridge (path) ~18m; Doe Ridge ~18m; #81 Polecat Loop (path) ~173m; Polecat Loop (STM) ~173m
- Decision as runner: Follow #81 Polecat Loop / Polecat Loop until parked car / trailhead; target is Cartwright Trailhead.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Finish / return to car

- Physical role: Route endpoint / return-to-car point.
- Model frame: The packet endpoint closes the loop.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Vehicle movement may be audible or visible near OSM service connector 69243, North Cartwright Road, OSM track connector 12641; do not mistake the road/driveway line for the trail branch.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: #76 Peggy's, #81 Polecat Loop; vehicle corridor or service/residential road context: OSM service connector 69243, North Cartwright Road, OSM track connector 12641; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 69243 (service) ~3m; Polecat Loop ~7m; North Cartwright Road (tertiary) ~23m; #76 Peggy's (path) ~34m; Peggy's ~35m; #81 Polecat Loop (path) ~66m
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

- OSM service connector 69243 (~2m, vehicle, highway=service, source=openstreetmap)
- Polecat Loop (~7m, feature, source=ridge_to_rivers_open_data)
- North Cartwright Road (~22m, vehicle, highway=tertiary, source=openstreetmap)
- #76 Peggy's (~31m, path, highway=path, source=openstreetmap)
- Peggy's (~34m, feature, source=ridge_to_rivers_open_data)
- #82 Doe Ridge (~41m, path, highway=path, source=openstreetmap)
- Doe Ridge (~41m, feature, source=ridge_to_rivers_open_data)
- Quick Draw (~48m, feature, source=ridge_to_rivers_open_data)
- #83 Quick Draw (~50m, path, highway=path, source=openstreetmap)
- #81 Polecat Loop (~66m, path, highway=path, source=openstreetmap)
- Polecat Loop (STM) (~67m, feature, source=ridge_to_rivers_open_data)
- OSM track connector 12641 (~140m, vehicle, highway=track, source=openstreetmap)
- OSM path connector 13996 (~164m, path, highway=path, source=openstreetmap)
- OSM track connector 106931 (~169m, vehicle, highway=track, source=openstreetmap)

## Required Next Proof

- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.
- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.
- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.

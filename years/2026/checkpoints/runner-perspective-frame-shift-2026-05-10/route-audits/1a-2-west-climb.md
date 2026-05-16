# Runner-Perspective Frame Shift: 1A-2 - West Climb

## Frame Contract

- Route card: `1A-2` / outing `1-2`.
- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.
- Evidence used: `docs/field-packet/field-tool-data.json`, `docs/field-packet/gpx/official/1a-2-west-climb-bob-smylie-buena-vista-trail-full-sail-trail.gpx`, R2R open data, OSM connector overlay, official 2026 segment source.
- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.
- Frame decision: `needs-proof`. The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.
- Access status: parking evidence incomplete in packet data.
- Human-validity status for this audit: `needs_visual_proof`.

## Route Snapshot

- Trails: Bob Smylie, Buena Vista Trail, Full Sail Trail.
- Official miles: 3.13; on-foot miles: 4.48.
- Door-to-door: p75 118 min; p90 133 min.
- Segment count: 8; wayfinding cue count: 7.

## Start-End-Junction Frame Shifts

### Start

- Physical role: Park here and start this outing.
- Model frame: The packet proves the route has a start coordinate and a first cue.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Vehicle movement may be audible or visible near North Ussery Street; do not mistake the road/driveway line for the trail branch.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: #55 West Climb, #56 Full Sail, OSM path connector 94808, #54 Robert Smylie; vehicle corridor or service/residential road context: North Ussery Street; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: West Climb ~0m; #55 West Climb (path) ~1m; North Ussery Street (residential) ~15m; Full Sail ~51m; #56 Full Sail (path) ~52m; OSM path connector 94808 (path) ~99m
- Decision as runner: Before moving, find the first signed trail or road-access line and confirm the car is parked where the return cue can actually resolve.
- Wrong-layer risk: start/finish access can fail even when route geometry passes; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: cue GPX waypoint plus local R2R/OSM overlay; no live imagery or field photo in this audit

### Cue 01: FOLLOW #55 West Climb

- Physical role: signed junction with Bob Smylie
- Model frame: The packet says `01 0.00 mi (+0.14) START/ACCESS FOLLOW #55 West Climb UNTIL signed junction with Bob Smylie.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#56 Full Sail, OSM path connector 94808, #54 Robert Smylie); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #55 West Climb, #56 Full Sail, OSM path connector 94808, #54 Robert Smylie; vehicle corridor or service/residential road context: North Ussery Street; the branch to privilege is `#55 West Climb` until `signed junction with Bob Smylie`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #55 West Climb (path) ~3m; West Climb ~4m; North Ussery Street (residential) ~15m; #56 Full Sail (path) ~55m; Full Sail ~55m; OSM path connector 94808 (path) ~99m
- Decision as runner: Follow #55 West Climb until signed junction with Bob Smylie; target is Bob Smylie.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 02: FOLLOW Bob Smylie

- Physical role: signed junction with #53 Buena Vista Trail
- Model frame: The packet says `02 0.14 mi (+0.80) FOLLOW Bob Smylie UNTIL signed junction with #53 Buena Vista Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#55 West Climb, #54 Robert Smylie, #53 Buena Vista, #56 Full Sail); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #55 West Climb, #54 Robert Smylie, #53 Buena Vista, #56 Full Sail; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Bob Smylie` until `signed junction with #53 Buena Vista Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: West Climb ~2m; #55 West Climb (path) ~3m; #54 Robert Smylie (path) ~15m; Robert Smylie ~15m; #53 Buena Vista (path) ~69m; Buena Vista ~70m
- Decision as runner: Follow Bob Smylie until signed junction with #53 Buena Vista Trail; target is #53 Buena Vista Trail.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 03: FOLLOW #53 Buena Vista / Buena Vista / Kemper’s Ridge #52 / #54 Robert Smylie

- Physical role: signed junction with #53 Buena Vista Trail
- Model frame: The packet says `03 0.94 mi (+0.59) CONNECTOR FOLLOW #53 Buena Vista / Buena Vista / Kemper’s Ridge #52 / #54 Robert Smylie UNTIL signed junction with #53 Buena Vista Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#55 West Climb, #56 Full Sail); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #55 West Climb, #54 Robert Smylie, #53 Buena Vista, #56 Full Sail; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#53 Buena Vista / Buena Vista / Kemper’s Ridge #52 / #54 Robert Smylie` until `signed junction with #53 Buena Vista Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: West Climb ~0m; #55 West Climb (path) ~1m; #54 Robert Smylie (path) ~24m; Robert Smylie ~24m; #53 Buena Vista (path) ~47m; Buena Vista ~48m
- Decision as runner: Follow #53 Buena Vista / Buena Vista / Kemper’s Ridge #52 / #54 Robert Smylie until signed junction with #53 Buena Vista Trail; target is #53 Buena Vista Trail.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 04: TAKE #53 Buena Vista Trail

- Physical role: signed junction with Full Sail Trail
- Model frame: The packet says `04 1.53 mi (+1.37) JCT TAKE #53 Buena Vista Trail UNTIL signed junction with Full Sail Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#56 Full Sail, #55 West Climb, #54 Robert Smylie); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #56 Full Sail, #53 Buena Vista, #55 West Climb, #54 Robert Smylie; vehicle corridor or service/residential road context: North Stone Creek Way; the branch to privilege is `#53 Buena Vista Trail` until `signed junction with Full Sail Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Full Sail ~0m; #56 Full Sail (path) ~1m; #53 Buena Vista (path) ~39m; Buena Vista ~41m; North Stone Creek Way (residential) ~156m; #55 West Climb (path) ~161m
- Decision as runner: Follow #53 Buena Vista Trail until signed junction with Full Sail Trail; target is Full Sail Trail.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 05: FOLLOW #53 Buena Vista / Buena Vista

- Physical role: signed junction with Full Sail Trail
- Model frame: The packet says `05 2.90 mi (+0.25) CONNECTOR FOLLOW #53 Buena Vista / Buena Vista UNTIL signed junction with Full Sail Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (Who Now Loop #51, Kemper’s Ridge #52, OSM path connector 91564); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #53 Buena Vista, Who Now Loop #51, Kemper’s Ridge #52, OSM path connector 91564; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#53 Buena Vista / Buena Vista` until `signed junction with Full Sail Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #53 Buena Vista (path) ~0m; Buena Vista ~0m; Kemper's Ridge ~29m; Who Now Loop #51 (path) ~40m; Kemper’s Ridge #52 (path) ~44m; Who Now Loop ~46m
- Decision as runner: Follow #53 Buena Vista / Buena Vista until signed junction with Full Sail Trail; target is Full Sail Trail.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 06: CONTINUE STRAIGHT Full Sail Trail

- Physical role: end of Full Sail Trail for this route
- Model frame: The packet says `06 3.15 mi (+0.95) JCT CONTINUE STRAIGHT Full Sail Trail UNTIL end of Full Sail Trail for this route.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#53 Buena Vista, Who Now Loop #51, Kemper’s Ridge #52, OSM path connector 91564); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #53 Buena Vista, Who Now Loop #51, Kemper’s Ridge #52, OSM path connector 91564; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Full Sail Trail` until `end of Full Sail Trail for this route`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #53 Buena Vista (path) ~0m; Buena Vista ~0m; Kemper's Ridge ~29m; Who Now Loop #51 (path) ~40m; Kemper’s Ridge #52 (path) ~44m; Who Now Loop ~46m
- Decision as runner: Follow Full Sail Trail until end of Full Sail Trail for this route; target is return to car.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 07: FOLLOW #55 West Climb / #56 Full Sail / Full Sail / West Climb

- Physical role: parked car / trailhead
- Model frame: The packet says `07 4.10 mi (+0.51) EXIT FOLLOW #55 West Climb / #56 Full Sail / Full Sail / West Climb UNTIL parked car / trailhead.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#53 Buena Vista, OSM path connector 94808, #54 Robert Smylie); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #53 Buena Vista, #56 Full Sail, #55 West Climb, OSM path connector 94808, #54 Robert Smylie; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#55 West Climb / #56 Full Sail / Full Sail / West Climb` until `parked car / trailhead`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Buena Vista ~5m; #53 Buena Vista (path) ~6m; #56 Full Sail (path) ~38m; Full Sail ~39m; #55 West Climb (path) ~121m; West Climb ~123m
- Decision as runner: Follow #55 West Climb / #56 Full Sail / Full Sail / West Climb until parked car / trailhead; target is West Climb Trailhead.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Finish / return to car

- Physical role: Route endpoint / return-to-car point.
- Model frame: The packet endpoint closes the loop.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Vehicle movement may be audible or visible near North Ussery Street; do not mistake the road/driveway line for the trail branch.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: #55 West Climb, #56 Full Sail, OSM path connector 94808, #54 Robert Smylie; vehicle corridor or service/residential road context: North Ussery Street; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #55 West Climb (path) ~3m; West Climb ~4m; North Ussery Street (residential) ~15m; #56 Full Sail (path) ~55m; Full Sail ~55m; OSM path connector 94808 (path) ~99m
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

- West Climb (~0m, feature, source=ridge_to_rivers_open_data)
- #55 West Climb (~1m, path, highway=path, source=openstreetmap)
- North Ussery Street (~15m, vehicle, highway=residential, source=openstreetmap)
- Kemper's Ridge (~29m, feature, source=ridge_to_rivers_open_data)
- Who Now Loop #51 (~40m, path, highway=path, source=openstreetmap)
- Kemper’s Ridge #52 (~44m, path, highway=path, source=openstreetmap)
- Who Now Loop (~46m, feature, source=ridge_to_rivers_open_data)
- Full Sail (~51m, feature, source=ridge_to_rivers_open_data)
- #56 Full Sail (~52m, path, highway=path, source=openstreetmap)
- #53 Buena Vista (~69m, path, highway=path, source=openstreetmap)
- Buena Vista (~70m, feature, source=ridge_to_rivers_open_data)
- OSM path connector 94808 (~99m, path, highway=path, source=openstreetmap)
- #54 Robert Smylie (~123m, path, highway=path, source=openstreetmap)
- Robert Smylie (~123m, feature, source=ridge_to_rivers_open_data)
- North Stone Creek Way (~156m, vehicle, highway=residential, source=openstreetmap)
- OSM path connector 91564 (~162m, path, highway=path, source=openstreetmap)

## Required Next Proof

- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.
- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.
- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.

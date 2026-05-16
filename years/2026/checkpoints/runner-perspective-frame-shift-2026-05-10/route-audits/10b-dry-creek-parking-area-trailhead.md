# Runner-Perspective Frame Shift: 10B - Dry Creek Parking Area/Trailhead

## Frame Contract

- Route card: `10B` / outing `10-2`.
- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.
- Evidence used: `docs/field-packet/field-tool-data.json`, `docs/field-packet/gpx/official/10b-dry-creek-parking-area-trailhead-bitterbrush-trail-currant-creek.gpx`, R2R open data, OSM connector overlay, official 2026 segment source.
- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.
- Frame decision: `needs-proof`. The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.
- Access status: known-or-mapped parking in packet data.
- Human-validity status for this audit: `needs_visual_proof`.

## Route Snapshot

- Trails: Bitterbrush Trail, Currant Creek.
- Official miles: 2.45; on-foot miles: 5.43.
- Door-to-door: p75 152 min; p90 171 min.
- Segment count: 4; wayfinding cue count: 5.

## Start-End-Junction Frame Shifts

### Start

- Physical role: Park here and start this outing.
- Model frame: The packet proves the route has a start coordinate and a first cue.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: OSM path connector 111703, OSM path connector 111704, OSM path connector 13980, OSM path connector 15043; vehicle corridor or service/residential road context: OSM service connector 67274, West Dry Creek Road, OSM service connector 16511; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 67274 (service) ~1m; OSM path connector 111703 (path) ~17m; West Dry Creek Road (tertiary) ~23m; OSM path connector 111704 (path) ~40m; OSM path connector 13980 (path) ~52m; OSM service connector 16511 (service) ~53m
- Decision as runner: Before moving, find the first signed trail or road-access line and confirm the car is parked where the return cue can actually resolve.
- Wrong-layer risk: start/finish access can fail even when route geometry passes; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: cue GPX waypoint plus local R2R/OSM overlay; no live imagery or field photo in this audit

### Cue 01: FOLLOW #71 Red Tail

- Physical role: signed junction with Bitterbrush Trail
- Model frame: The packet says `01 0.00 mi (+1.07) START/ACCESS FOLLOW #71 Red Tail UNTIL signed junction with Bitterbrush Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (OSM path connector 111703, OSM path connector 111704, OSM path connector 13980, OSM path connector 15043); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: OSM path connector 111703, OSM path connector 111704, OSM path connector 13980, OSM path connector 15043; vehicle corridor or service/residential road context: OSM service connector 67274, West Dry Creek Road, OSM service connector 16511; the branch to privilege is `#71 Red Tail` until `signed junction with Bitterbrush Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM path connector 111703 (path) ~1m; OSM service connector 67274 (service) ~1m; West Dry Creek Road (tertiary) ~22m; OSM path connector 111704 (path) ~24m; OSM path connector 13980 (path) ~37m; OSM service connector 16511 (service) ~40m
- Decision as runner: Follow #71 Red Tail until signed junction with Bitterbrush Trail; target is Bitterbrush Trail.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 02: FOLLOW Bitterbrush Trail

- Physical role: signed junction with Currant Creek
- Model frame: The packet says `02 1.07 mi (+0.66) FOLLOW Bitterbrush Trail UNTIL signed junction with Currant Creek.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#74 Chukar Butte, #73 Bitterbrush, South Bitterbrush); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #74 Chukar Butte, #73 Bitterbrush, South Bitterbrush; vehicle corridor or service/residential road context: North Cartwright Road; the branch to privilege is `Bitterbrush Trail` until `signed junction with Currant Creek`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #74 Chukar Butte (path) ~1m; #73 Bitterbrush (path) ~16m; Bitterbrush ~16m; South Bitterbrush (path) ~38m; North Cartwright Road (tertiary) ~54m; Chukar Butte (Dog On-Leash) ~99m
- Decision as runner: Follow Bitterbrush Trail until signed junction with Currant Creek; target is Currant Creek.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 03: FOLLOW #74 Chukar Butte / Chukar Butte (Dog On-Leash) / Currant Creek / W. Currant Creek / West Deerpath Drive

- Physical role: signed junction with Currant Creek
- Model frame: The packet says `03 1.73 mi (+0.37) CONNECTOR FOLLOW #74 Chukar Butte / Chukar Butte (Dog On-Leash) / Currant Creek / W. Currant Creek / West Deerpath Drive UNTIL signed junction with Currant Creek.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#73 Bitterbrush, #71 Red Tail); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #73 Bitterbrush, #71 Red Tail; vehicle corridor or service/residential road context: West Deerpath Drive, West Sage Creek Drive; the branch to privilege is `#74 Chukar Butte / Chukar Butte (Dog On-Leash) / Currant Creek / W. Currant Creek / West Deerpath Drive` until `signed junction with Currant Creek`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #73 Bitterbrush (path) ~0m; Bitterbrush ~1m; Red Tail ~30m; West Deerpath Drive (residential) ~35m; #71 Red Tail (path) ~36m; West Sage Creek Drive (residential) ~128m
- Decision as runner: Follow #74 Chukar Butte / Chukar Butte (Dog On-Leash) / Currant Creek / W. Currant Creek / West Deerpath Drive until signed junction with Currant Creek; target is Currant Creek.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 04: TAKE Currant Creek

- Physical role: end of Currant Creek for this route
- Model frame: The packet says `04 2.10 mi (+1.79) JCT TAKE Currant Creek UNTIL end of Currant Creek for this route.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#74 Chukar Butte, #75 Currant Creek); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #74 Chukar Butte, #75 Currant Creek; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Currant Creek` until `end of Currant Creek for this route`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #74 Chukar Butte (path) ~4m; Chukar Butte (Dog On-Leash) ~4m; #75 Currant Creek (path) ~12m; Currant Creek ~13m; Chukar Butte ~42m
- Decision as runner: Follow Currant Creek until end of Currant Creek for this route; target is return to car.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 05: FOLLOW #71 Red Tail

- Physical role: parked car / trailhead
- Model frame: The packet says `05 3.89 mi (+2.74) EXIT FOLLOW #71 Red Tail UNTIL parked car / trailhead.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#75 Currant Creek, W. Currant Creek, #70 Landslide Loop, S. Currant Creek, OSM path connector 106907); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #75 Currant Creek, W. Currant Creek, #70 Landslide Loop, S. Currant Creek, OSM path connector 106907; vehicle corridor or service/residential road context: West Deerpath Drive; the branch to privilege is `#71 Red Tail` until `parked car / trailhead`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #75 Currant Creek (path) ~2m; Currant Creek ~2m; W. Currant Creek (path) ~41m; #70 Landslide Loop (path) ~59m; Landslide Loop ~61m; S. Currant Creek (path) ~82m
- Decision as runner: Follow #71 Red Tail until parked car / trailhead; target is Dry Creek Parking Area/Trailhead.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Finish / return to car

- Physical role: Route endpoint / return-to-car point.
- Model frame: The packet endpoint closes the loop.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: OSM path connector 111703, OSM path connector 111704, OSM path connector 13980, OSM path connector 15043; vehicle corridor or service/residential road context: OSM service connector 67274, West Dry Creek Road, OSM service connector 16511; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM path connector 111703 (path) ~1m; OSM service connector 67274 (service) ~1m; West Dry Creek Road (tertiary) ~22m; OSM path connector 111704 (path) ~24m; OSM path connector 13980 (path) ~37m; OSM service connector 16511 (service) ~40m
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

- #74 Chukar Butte (~1m, path, highway=path, source=openstreetmap)
- OSM service connector 67274 (~1m, vehicle, highway=service, source=openstreetmap)
- #75 Currant Creek (~12m, path, highway=path, source=openstreetmap)
- Currant Creek (~13m, feature, source=ridge_to_rivers_open_data)
- #73 Bitterbrush (~16m, path, highway=path, source=openstreetmap)
- Bitterbrush (~16m, feature, source=ridge_to_rivers_open_data)
- OSM path connector 111703 (~17m, path, highway=path, source=openstreetmap)
- West Dry Creek Road (~23m, vehicle, highway=tertiary, source=openstreetmap)
- West Deerpath Drive (~35m, vehicle, highway=residential, source=openstreetmap)
- #71 Red Tail (~36m, path, highway=path, source=openstreetmap)
- South Bitterbrush (~38m, path, highway=path, source=openstreetmap)
- OSM path connector 111704 (~40m, path, highway=path, source=openstreetmap)
- W. Currant Creek (~41m, path, highway=path, source=openstreetmap)
- Chukar Butte (~42m, feature, source=ridge_to_rivers_open_data)
- OSM path connector 13980 (~52m, path, highway=path, source=openstreetmap)
- OSM service connector 16511 (~53m, vehicle, highway=service, source=openstreetmap)
- North Cartwright Road (~54m, vehicle, highway=tertiary, source=openstreetmap)
- #70 Landslide Loop (~59m, path, highway=path, source=openstreetmap)
- Landslide Loop (~61m, feature, source=ridge_to_rivers_open_data)
- OSM path connector 15043 (~76m, path, highway=path, source=openstreetmap)

## Required Next Proof

- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.
- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.
- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.

# Runner-Perspective Frame Shift: 4C-2 - Strava parking anchor 21

## Frame Contract

- Route card: `4C-2` / outing `4-2`.
- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.
- Evidence used: `docs/field-packet/field-tool-data.json`, `docs/field-packet/gpx/official/4c-2-strava-parking-anchor-21-shoshone-paiute-quarry-trail-castle-rock-table-rock-trail-ro.gpx`, R2R open data, OSM connector overlay, official 2026 segment source.
- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.
- Frame decision: `needs-proof`. The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.
- Access status: private-history parking anchor; usable as planning evidence but still public-proof limited.
- Human-validity status for this audit: `needs_visual_proof`.

## Route Snapshot

- Trails: Shoshone-Paiute, Quarry Trail - Castle Rock, Table Rock Trail, Rock Garden, Rock Island.
- Official miles: 5.08; on-foot miles: 7.47.
- Door-to-door: p75 188 min; p90 211 min.
- Segment count: 25; wayfinding cue count: 11.

## Start-End-Junction Frame Shifts

### Start

- Physical role: Park here and start this outing.
- Model frame: The packet proves the route has a start coordinate and a first cue.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars or road-edge ambiguity are plausible because the start is a road or anchor-style access point; treat exact parking legality as a separate proof.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: OSM footway connector 88078, Shoshone-Bannock Tribes Trail, #19 Shoshone-Paiute Tribes Trail, OSM path connector 15301, Shoshone-Paiute Tribes Trail; vehicle corridor or service/residential road context: East Hays Court, OSM service connector 25083; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: East Hays Court (residential) ~1m; OSM footway connector 88078 (footway) ~6m; Shoshone-Bannock Tribes Trail ~39m; #19 Shoshone-Paiute Tribes Trail (path) ~44m; OSM service connector 25083 (service) ~44m; OSM path connector 15301 (path) ~45m
- Decision as runner: Before moving, find the first signed trail or road-access line and confirm the car is parked where the return cue can actually resolve.
- Wrong-layer risk: start/finish access can fail even when route geometry passes; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: cue GPX waypoint plus local R2R/OSM overlay; no live imagery or field photo in this audit

### Cue 01: FOLLOW Shoshone-Paiute / #19A Shoshone-Bannock Tribes Trail

- Physical role: signed Shoshone-Paiute route / first official segment
- Model frame: The packet says `01 0.00 mi (+0.03) OFFICIAL START FOLLOW Shoshone-Paiute / #19A Shoshone-Bannock Tribes Trail UNTIL signed Shoshone-Paiute route / first official segment.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (OSM footway connector 88078, OSM path connector 15301, OSM footway connector 88077, #19 Shoshone-Paiute Tribes Trail, Shoshone-Paiute Tribes Trail); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: OSM footway connector 88078, OSM path connector 15301, Shoshone-Bannock Tribes Trail, OSM footway connector 88077, #19 Shoshone-Paiute Tribes Trail; vehicle corridor or service/residential road context: East Hays Court, OSM service connector 25083; the branch to privilege is `Shoshone-Paiute / #19A Shoshone-Bannock Tribes Trail` until `signed Shoshone-Paiute route / first official segment`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM footway connector 88078 (footway) ~1m; East Hays Court (residential) ~5m; OSM service connector 25083 (service) ~38m; OSM path connector 15301 (path) ~39m; Shoshone-Bannock Tribes Trail ~39m; OSM footway connector 88077 (footway) ~43m
- Decision as runner: Follow Shoshone-Paiute / #19A Shoshone-Bannock Tribes Trail until signed Shoshone-Paiute route / first official segment; target is Shoshone-Paiute.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 02: FOLLOW Shoshone-Paiute

- Physical role: signed junction with Quarry Trail - Castle Rock
- Model frame: The packet says `02 0.03 mi (+0.41) FOLLOW Shoshone-Paiute UNTIL signed junction with Quarry Trail - Castle Rock.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (OSM footway connector 88078, Shoshone-Bannock Tribes Trail, Shoshone-Paiute Tribes Trail, #19 Shoshone-Paiute Tribes Trail, OSM path connector 15301); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: OSM footway connector 88078, Shoshone-Bannock Tribes Trail, Shoshone-Paiute Tribes Trail, #19 Shoshone-Paiute Tribes Trail, OSM path connector 15301; vehicle corridor or service/residential road context: East Hays Court, OSM service connector 25083; the branch to privilege is `Shoshone-Paiute` until `signed junction with Quarry Trail - Castle Rock`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM footway connector 88078 (footway) ~1m; Shoshone-Bannock Tribes Trail ~3m; East Hays Court (residential) ~24m; Shoshone-Paiute Tribes Trail ~25m; #19 Shoshone-Paiute Tribes Trail (path) ~27m; OSM path connector 15301 (path) ~45m
- Decision as runner: Follow Shoshone-Paiute until signed junction with Quarry Trail - Castle Rock; target is Quarry Trail - Castle Rock.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 03: FOLLOW #18 Quarry / Access Trail (Quarry) / Quarry

- Physical role: signed junction with Quarry Trail - Castle Rock
- Model frame: The packet says `03 0.44 mi (+0.27) CONNECTOR FOLLOW #18 Quarry / Access Trail (Quarry) / Quarry UNTIL signed junction with Quarry Trail - Castle Rock.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (Shoshone-Paiute Tribes Trail, #19 Shoshone-Paiute Tribes Trail, OSM footway connector 80210, OSM footway connector 80216); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: Shoshone-Paiute Tribes Trail, #19 Shoshone-Paiute Tribes Trail, #18 Quarry, OSM footway connector 80210, OSM footway connector 80216; vehicle corridor or service/residential road context: East Solitude Court, North Morningside Way; the branch to privilege is `#18 Quarry / Access Trail (Quarry) / Quarry` until `signed junction with Quarry Trail - Castle Rock`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Shoshone-Paiute Tribes Trail ~0m; #19 Shoshone-Paiute Tribes Trail (path) ~2m; #18 Quarry (path) ~39m; Quarry ~43m; OSM footway connector 80210 (footway) ~89m; East Solitude Court (residential) ~104m
- Decision as runner: Follow #18 Quarry / Access Trail (Quarry) / Quarry until signed junction with Quarry Trail - Castle Rock; target is Quarry Trail - Castle Rock.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 04: BEAR LEFT Quarry Trail - Castle Rock

- Physical role: signed junction with Table Rock Trail
- Model frame: The packet says `04 0.71 mi (+0.54) JCT BEAR LEFT Quarry Trail - Castle Rock UNTIL signed junction with Table Rock Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#18 Quarry, OSM path connector 106705, Eastdale Access To Quary Trail, Access Trail (Quarry), OSM path connector 106704); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #18 Quarry, OSM path connector 106705, Eastdale Access To Quary Trail, Access Trail (Quarry), OSM path connector 106704; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Quarry Trail - Castle Rock` until `signed junction with Table Rock Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Quarry ~3m; #18 Quarry (path) ~4m; OSM path connector 106705 (path) ~33m; Eastdale Access To Quary Trail (path) ~35m; Access Trail (Quarry) ~37m; Connector ~64m
- Decision as runner: Follow Quarry Trail - Castle Rock until signed junction with Table Rock Trail; target is Table Rock Trail.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 05: FOLLOW #19 Shoshone-Paiute Tribes Trail / Quarry / Shoshone-Paiute Tribes Trail / Table Rock

- Physical role: signed junction with Table Rock Trail
- Model frame: The packet says `05 1.25 mi (+0.21) CONNECTOR FOLLOW #19 Shoshone-Paiute Tribes Trail / Quarry / Shoshone-Paiute Tribes Trail / Table Rock UNTIL signed junction with Table Rock Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#18 Quarry, Access Trail (Quarry), Eastdale Access To Quary Trail, OSM path connector 106704, OSM path connector 106732); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #18 Quarry, Access Trail (Quarry), Eastdale Access To Quary Trail, OSM path connector 106704, OSM path connector 106732; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#19 Shoshone-Paiute Tribes Trail / Quarry / Shoshone-Paiute Tribes Trail / Table Rock` until `signed junction with Table Rock Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Quarry ~0m; #18 Quarry (path) ~1m; Connector ~7m; Access Trail (Quarry) ~29m; Eastdale Access To Quary Trail (path) ~29m; OSM path connector 106704 (path) ~51m
- Decision as runner: Follow #19 Shoshone-Paiute Tribes Trail / Quarry / Shoshone-Paiute Tribes Trail / Table Rock until signed junction with Table Rock Trail; target is Table Rock Trail.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 06: TAKE Table Rock Trail

- Physical role: signed junction with Rock Garden
- Model frame: The packet says `06 1.46 mi (+1.38) JCT TAKE Table Rock Trail UNTIL signed junction with Rock Garden.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#18 Quarry, Access Trail (Quarry), Eastdale Access To Quary Trail, OSM path connector 106704, OSM path connector 106732); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #18 Quarry, Access Trail (Quarry), Eastdale Access To Quary Trail, OSM path connector 106704, OSM path connector 106732; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Table Rock Trail` until `signed junction with Rock Garden`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Quarry ~0m; #18 Quarry (path) ~1m; Connector ~7m; Access Trail (Quarry) ~29m; Eastdale Access To Quary Trail (path) ~29m; OSM path connector 106704 (path) ~51m
- Decision as runner: Follow Table Rock Trail until signed junction with Rock Garden; target is Rock Garden.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 07: FOLLOW #15 Table Rock / Connector / Rock Garden / Rock Island (Jump Line) / Table Rock

- Physical role: signed junction with Rock Garden
- Model frame: The packet says `07 2.84 mi (+0.52) CONNECTOR FOLLOW #15 Table Rock / Connector / Rock Garden / Rock Island (Jump Line) / Table Rock UNTIL signed junction with Rock Garden.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#19 Shoshone-Paiute Tribes Trail, Shoshone-Paiute Tribes Trail, #19A Shoshone-Bannock Tribes Trail); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #19 Shoshone-Paiute Tribes Trail, #15 Table Rock, Shoshone-Paiute Tribes Trail, #19A Shoshone-Bannock Tribes Trail; vehicle corridor or service/residential road context: OSM service connector 75757, OSM service connector 77762, OSM service connector 77763; the branch to privilege is `#15 Table Rock / Connector / Rock Garden / Rock Island (Jump Line) / Table Rock` until `signed junction with Rock Garden`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #19 Shoshone-Paiute Tribes Trail (path) ~3m; #15 Table Rock (path) ~36m; Table Rock ~37m; Shoshone-Paiute Tribes Trail ~42m; #19A Shoshone-Bannock Tribes Trail (path) ~43m; OSM service connector 75757 (service) ~77m
- Decision as runner: Follow #15 Table Rock / Connector / Rock Garden / Rock Island (Jump Line) / Table Rock until signed junction with Rock Garden; target is Rock Garden.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 08: CONTINUE STRAIGHT Rock Garden

- Physical role: signed junction with Rock Island
- Model frame: The packet says `08 3.36 mi (+0.69) JCT CONTINUE STRAIGHT Rock Garden UNTIL signed junction with Rock Island.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#16A Rock Garden, #16B Rock Island (Jump Line), #16B Rock Island (West), #16B Rock Island (East)); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #16A Rock Garden, #16B Rock Island (Jump Line), #16B Rock Island (West), #16B Rock Island (East); no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Rock Garden` until `signed junction with Rock Island`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Rock Garden ~0m; #16A Rock Garden (path) ~1m; #16B Rock Island (Jump Line) (path) ~32m; Rock Island (Jump Line) ~32m; #16B Rock Island (West) (path) ~42m; Rock Island (West) ~48m
- Decision as runner: Follow Rock Garden until signed junction with Rock Island; target is Rock Island.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 09: FOLLOW #16A Rock Garden / Connector / Rock Garden / Rock Island (West)

- Physical role: signed junction with Rock Island
- Model frame: The packet says `09 4.05 mi (+0.06) CONNECTOR FOLLOW #16A Rock Garden / Connector / Rock Garden / Rock Island (West) UNTIL signed junction with Rock Island.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#16B Rock Island (West), #16B Rock Island (Jump Line)); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #16A Rock Garden, #16B Rock Island (West), #16B Rock Island (Jump Line), Connector; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#16A Rock Garden / Connector / Rock Garden / Rock Island (West)` until `signed junction with Rock Island`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #16A Rock Garden (path) ~2m; Rock Island (West) ~2m; Rock Garden ~3m; #16B Rock Island (West) (path) ~36m; Rock Island (Jump Line) ~128m; #16B Rock Island (Jump Line) (path) ~133m
- Decision as runner: Follow #16A Rock Garden / Connector / Rock Garden / Rock Island (West) until signed junction with Rock Island; target is Rock Island.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 10: CONTINUE STRAIGHT Rock Island

- Physical role: end of Rock Island for this route
- Model frame: The packet says `10 4.11 mi (+2.05) JCT CONTINUE STRAIGHT Rock Island UNTIL end of Rock Island for this route.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#16A Rock Garden, #16B Rock Island (West), #16B Rock Island (Jump Line), Connector); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #16A Rock Garden, #16B Rock Island (West), #16B Rock Island (Jump Line), Connector; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Rock Island` until `end of Rock Island for this route`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #16A Rock Garden (path) ~2m; Rock Island (West) ~2m; Rock Garden ~3m; #16B Rock Island (West) (path) ~36m; Rock Island (Jump Line) ~128m; #16B Rock Island (Jump Line) (path) ~133m
- Decision as runner: Follow Rock Island until end of Rock Island for this route; target is return to car.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch; wrong-direction choice has meaningful climb penalty
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 11: FOLLOW #15 Table Rock / #15A Old Pen / #16B Rock Island (East) / #19A Shoshone-Bannock Tribes Trail / Connector / OSM path connector 106734 / Old Pen / Shoshone-Bannock Tribes Trail / Shoshone-Paiute Tribes Trail / Table Rock

- Physical role: parked car / trailhead
- Model frame: The packet says `11 6.16 mi (+1.79) EXIT FOLLOW #15 Table Rock / #15A Old Pen / #16B Rock Island (East) / #19A Shoshone-Bannock Tribes Trail / Connector / OSM path connector 106734 / Old Pen / Shoshone-Bannock Tribes Trail / Shoshone-Paiute Tribes Trail / Table Rock UNTIL parked car / trailhead.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#16B Rock Island (West), #16A Rock Garden, #16B Rock Island (Jump Line)); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #16B Rock Island (West), #16A Rock Garden, #16B Rock Island (Jump Line); no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#15 Table Rock / #15A Old Pen / #16B Rock Island (East) / #19A Shoshone-Bannock Tribes Trail / Connector / OSM path connector 106734 / Old Pen / Shoshone-Bannock Tribes Trail / Shoshone-Paiute Tribes Trail / Table Rock` until `parked car / trailhead`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #16B Rock Island (West) (path) ~0m; Rock Island (West) ~1m; #16A Rock Garden (path) ~11m; #16B Rock Island (Jump Line) (path) ~34m; Rock Island (Jump Line) ~36m; Rock Garden ~58m
- Decision as runner: Follow #15 Table Rock / #15A Old Pen / #16B Rock Island (East) / #19A Shoshone-Bannock Tribes Trail / Connector / OSM path connector 106734 / Old Pen / Shoshone-Bannock Tribes Trail / Shoshone-Paiute Tribes Trail / Table Rock until parked car / trailhead; target is Strava parking anchor 21.
- Wrong-layer risk: generic OSM connector name may not exist on signs
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Finish / return to car

- Physical role: Route endpoint / return-to-car point.
- Model frame: The packet endpoint closes the loop.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars or road-edge ambiguity are plausible because the start is a road or anchor-style access point; treat exact parking legality as a separate proof.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: OSM footway connector 88078, OSM path connector 15301, Shoshone-Bannock Tribes Trail, OSM footway connector 88077, #19 Shoshone-Paiute Tribes Trail; vehicle corridor or service/residential road context: East Hays Court, OSM service connector 25083; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM footway connector 88078 (footway) ~1m; East Hays Court (residential) ~5m; OSM service connector 25083 (service) ~38m; OSM path connector 15301 (path) ~39m; Shoshone-Bannock Tribes Trail ~39m; OSM footway connector 88077 (footway) ~43m
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

- Rock Garden (~0m, feature, source=ridge_to_rivers_open_data)
- #16A Rock Garden (~1m, path, highway=path, source=openstreetmap)
- East Hays Court (~1m, vehicle, highway=residential, source=openstreetmap)
- OSM footway connector 88078 (~6m, path, highway=footway, source=openstreetmap)
- #16B Rock Island (Jump Line) (~32m, path, highway=path, source=openstreetmap)
- Rock Island (Jump Line) (~32m, feature, source=ridge_to_rivers_open_data)
- OSM path connector 106705 (~33m, path, highway=path, source=openstreetmap)
- Eastdale Access To Quary Trail (~35m, path, highway=path, source=openstreetmap)
- #15 Table Rock (~36m, path, highway=path, source=openstreetmap)
- Access Trail (Quarry) (~37m, path, source=ridge_to_rivers_open_data)
- Table Rock (~37m, feature, source=ridge_to_rivers_open_data)
- #18 Quarry (~39m, path, highway=path, source=openstreetmap)
- Shoshone-Bannock Tribes Trail (~39m, path, source=ridge_to_rivers_open_data)
- #16B Rock Island (West) (~42m, path, highway=path, source=openstreetmap)
- #19A Shoshone-Bannock Tribes Trail (~43m, path, highway=path, source=openstreetmap)
- Quarry (~43m, feature, source=ridge_to_rivers_open_data)
- #19 Shoshone-Paiute Tribes Trail (~44m, path, highway=path, source=openstreetmap)
- OSM service connector 25083 (~44m, vehicle, highway=service, source=openstreetmap)
- OSM path connector 15301 (~45m, path, highway=path, source=openstreetmap)
- Shoshone-Paiute Tribes Trail (~45m, path, source=ridge_to_rivers_open_data)

## Required Next Proof

- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.
- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.
- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.

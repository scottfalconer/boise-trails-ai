# Runner-Perspective Frame Shift: 15A-2 - Bob's

## Frame Contract

- Route card: `15A-2` / outing `15-2`.
- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.
- Evidence used: `docs/field-packet/field-tool-data.json`, `docs/field-packet/gpx/official/15a-2-bob-s-highlands-trail-connector.gpx`, R2R open data, OSM connector overlay, official 2026 segment source.
- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.
- Frame decision: `needs-proof`. The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.
- Access status: parking evidence incomplete in packet data.
- Human-validity status for this audit: `needs_visual_proof`.

## Route Snapshot

- Trails: Highlands Trail, Connector.
- Official miles: 2.35; on-foot miles: 4.51.
- Door-to-door: p75 113 min; p90 127 min.
- Segment count: 3; wayfinding cue count: 5.

## Start-End-Junction Frame Shifts

### Start

- Physical role: Park here and start this outing.
- Model frame: The packet proves the route has a start coordinate and a first cue.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. The local route data does not prove car traffic at this point; treat cars/road noise as field-only unless a road or parking surface is visibly present.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: #30 Bob's, #1 Highlands Trail, OSM footway connector 73045, OSM path connector 11370, OSM footway connector 73044; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Bob's ~0m; #30 Bob's (path) ~1m; #1 Highlands Trail (path) ~7m; Highlands ~19m; OSM footway connector 73045 (footway) ~20m; OSM path connector 11370 (path) ~70m
- Decision as runner: Before moving, find the first signed trail or road-access line and confirm the car is parked where the return cue can actually resolve.
- Wrong-layer risk: start/finish access can fail even when route geometry passes; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: cue GPX waypoint plus local R2R/OSM overlay; no live imagery or field photo in this audit

### Cue 01: FOLLOW #30 Bob's

- Physical role: signed junction with Highlands Trail
- Model frame: The packet says `01 0.00 mi (+0.01) START/ACCESS FOLLOW #30 Bob's UNTIL signed junction with Highlands Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#1 Highlands Trail, OSM footway connector 73045, OSM path connector 11370, OSM footway connector 73044, OSM footway connector 51185); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #30 Bob's, #1 Highlands Trail, OSM footway connector 73045, OSM path connector 11370, OSM footway connector 73044; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#30 Bob's` until `signed junction with Highlands Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #30 Bob's (path) ~5m; #1 Highlands Trail (path) ~6m; Bob's ~6m; Highlands ~14m; OSM footway connector 73045 (footway) ~21m; OSM path connector 11370 (path) ~76m
- Decision as runner: Follow #30 Bob's until signed junction with Highlands Trail; target is Highlands Trail.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 02: FOLLOW Highlands Trail

- Physical role: signed junction with Connector
- Model frame: The packet says `02 0.01 mi (+1.68) FOLLOW Highlands Trail UNTIL signed junction with Connector.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#1 Highlands Trail, #30 Bob's, OSM footway connector 73045, OSM path connector 11370, OSM footway connector 73044); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #1 Highlands Trail, #30 Bob's, OSM footway connector 73045, OSM path connector 11370, OSM footway connector 73044; vehicle corridor or service/residential road context: East Hearthstone Drive; the branch to privilege is `Highlands Trail` until `signed junction with Connector`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #1 Highlands Trail (path) ~1m; Highlands ~9m; Bob's ~15m; #30 Bob's (path) ~16m; East Hearthstone Drive (residential) ~18m; OSM footway connector 73045 (footway) ~25m
- Decision as runner: Follow Highlands Trail until signed junction with Connector; target is Connector.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 03: FOLLOW #31 Corrals / Connector / Corrals / North Bogus Basin Road / OSM service connector 11395

- Physical role: signed junction with Connector
- Model frame: The packet says `03 1.69 mi (+0.71) ROAD FOLLOW #31 Corrals / Connector / Corrals / North Bogus Basin Road / OSM service connector 11395 UNTIL signed junction with Connector.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#1 Highlands Trail); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #1 Highlands Trail, #31 Corrals; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#31 Corrals / Connector / Corrals / North Bogus Basin Road / OSM service connector 11395` until `signed junction with Connector`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #1 Highlands Trail (path) ~2m; Highlands ~10m; Corrals ~38m; #31 Corrals (path) ~39m
- Decision as runner: Follow #31 Corrals / Connector / Corrals / North Bogus Basin Road / OSM service connector 11395 until signed junction with Connector; target is Connector.
- Wrong-layer risk: generic OSM connector name may not exist on signs
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 04: TAKE Connector

- Physical role: end of Connector for this route
- Model frame: The packet says `04 2.40 mi (+0.67) JCT TAKE Connector UNTIL end of Connector for this route.`.
- Runner frame: Runner frame: the immediate job is to keep the current trail until the named junction/landmark, with no extra branch proven by local data at this checkpoint.
- Likely visual field: mapped named route features near you: Connector; vehicle corridor or service/residential road context: OSM service connector 11395, North Bogus Basin Road; the branch to privilege is `Connector` until `end of Connector for this route`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 11395 (service) ~4m; North Bogus Basin Road (tertiary) ~23m; Connector ~40m
- Decision as runner: Follow Connector until end of Connector for this route; target is return to car.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 05: FOLLOW #1 Highlands Trail / #31 Corrals / Corrals / Highlands

- Physical role: parked car / trailhead
- Model frame: The packet says `05 3.07 mi (+2.17) EXIT FOLLOW #1 Highlands Trail / #31 Corrals / Corrals / Highlands UNTIL parked car / trailhead.`.
- Runner frame: Runner frame: the immediate job is to keep the current trail until the named junction/landmark, with no extra branch proven by local data at this checkpoint.
- Likely visual field: mapped trail/path choices near you: #31 Corrals; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#1 Highlands Trail / #31 Corrals / Corrals / Highlands` until `parked car / trailhead`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Connector ~1m; #31 Corrals (path) ~20m; Corrals ~21m
- Decision as runner: Follow #1 Highlands Trail / #31 Corrals / Corrals / Highlands until parked car / trailhead; target is Bob's Trailhead.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Finish / return to car

- Physical role: Route endpoint / return-to-car point.
- Model frame: The packet endpoint closes the loop.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. The local route data does not prove car traffic at this point; treat cars/road noise as field-only unless a road or parking surface is visibly present.
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
- OSM service connector 11395 (~4m, vehicle, highway=service, source=openstreetmap)
- #1 Highlands Trail (~7m, path, highway=path, source=openstreetmap)
- East Hearthstone Drive (~18m, vehicle, highway=residential, source=openstreetmap)
- Highlands (~19m, feature, source=ridge_to_rivers_open_data)
- OSM footway connector 73045 (~20m, path, highway=footway, source=openstreetmap)
- North Bogus Basin Road (~23m, vehicle, highway=tertiary, source=openstreetmap)
- Corrals (~38m, feature, source=ridge_to_rivers_open_data)
- #31 Corrals (~39m, path, highway=path, source=openstreetmap)
- Connector (~40m, feature, source=ridge_to_rivers_open_data)
- OSM path connector 11370 (~70m, path, highway=path, source=openstreetmap)
- OSM footway connector 73044 (~81m, path, highway=footway, source=openstreetmap)
- OSM footway connector 51185 (~96m, path, highway=footway, source=openstreetmap)

## Required Next Proof

- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.
- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.
- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.

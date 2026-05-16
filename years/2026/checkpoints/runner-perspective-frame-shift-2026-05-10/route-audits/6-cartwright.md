# Runner-Perspective Frame Shift: 6 - Cartwright

## Frame Contract

- Route card: `6` / outing `6-1`.
- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.
- Evidence used: `docs/field-packet/field-tool-data.json`, `docs/field-packet/gpx/official/6-cartwright-peggy-s-trail-chukar-butte-trail-cartwright-connector-cartwright-ridge-chbh-c.gpx`, R2R open data, OSM connector overlay, official 2026 segment source.
- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.
- Frame decision: `needs-proof`. The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.
- Access status: known-or-mapped parking in packet data.
- Human-validity status for this audit: `needs_visual_proof`.

## Route Snapshot

- Trails: Peggy's Trail, Chukar Butte Trail, Cartwright Connector, Cartwright Ridge, CHBH Connector.
- Official miles: 13.67; on-foot miles: 21.53.
- Door-to-door: p75 448 min; p90 502 min.
- Segment count: 8; wayfinding cue count: 10.

## Start-End-Junction Frame Shifts

### Start

- Physical role: Park here and start this outing.
- Model frame: The packet proves the route has a start coordinate and a first cue.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: #76 Peggy's, #81 Polecat Loop; vehicle corridor or service/residential road context: OSM service connector 69243, North Cartwright Road, OSM track connector 12641; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 69243 (service) ~2m; Polecat Loop ~7m; North Cartwright Road (tertiary) ~22m; #76 Peggy's (path) ~31m; Peggy's ~34m; #81 Polecat Loop (path) ~66m
- Decision as runner: Before moving, find the first signed trail or road-access line and confirm the car is parked where the return cue can actually resolve.
- Wrong-layer risk: start/finish access can fail even when route geometry passes
- Evidence boundary: cue GPX waypoint plus local R2R/OSM overlay; no live imagery or field photo in this audit

### Cue 01: FOLLOW #81 Polecat Loop

- Physical role: signed junction with Peggy's Trail
- Model frame: The packet says `01 0.00 mi (+0.17) START/ACCESS FOLLOW #81 Polecat Loop UNTIL signed junction with Peggy's Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#76 Peggy's); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #76 Peggy's, #81 Polecat Loop; vehicle corridor or service/residential road context: OSM service connector 69243, North Cartwright Road, OSM track connector 12641; the branch to privilege is `#81 Polecat Loop` until `signed junction with Peggy's Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 69243 (service) ~3m; Polecat Loop ~7m; North Cartwright Road (tertiary) ~23m; #76 Peggy's (path) ~34m; Peggy's ~35m; #81 Polecat Loop (path) ~66m
- Decision as runner: Follow #81 Polecat Loop until signed junction with Peggy's Trail; target is Peggy's Trail.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 02: FOLLOW Peggy's Trail

- Physical role: signed junction with Chukar Butte Trail
- Model frame: The packet says `02 0.17 mi (+4.56) FOLLOW Peggy's Trail UNTIL signed junction with Chukar Butte Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#76 Peggy's, OSM path connector 110670); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #76 Peggy's, OSM path connector 110670; vehicle corridor or service/residential road context: North Cartwright Road, OSM track connector 106014, OSM track connector 106931, OSM track connector 12641; the branch to privilege is `Peggy's Trail` until `signed junction with Chukar Butte Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #76 Peggy's (path) ~0m; Peggy's ~1m; North Cartwright Road (tertiary) ~18m; OSM track connector 106014 (track) ~40m; OSM path connector 110670 (path) ~42m; OSM track connector 106931 (track) ~74m
- Decision as runner: Follow Peggy's Trail until signed junction with Chukar Butte Trail; target is Chukar Butte Trail.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 03: FOLLOW #74 Chukar Butte / Chukar Butte / Chukar Butte (Dog On-Leash) / Sweet Connie

- Physical role: signed junction with Chukar Butte Trail
- Model frame: The packet says `03 4.73 mi (+2.43) CONNECTOR FOLLOW #74 Chukar Butte / Chukar Butte / Chukar Butte (Dog On-Leash) / Sweet Connie UNTIL signed junction with Chukar Butte Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#76 Peggy's, #77 Sweet Connie); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #76 Peggy's, #77 Sweet Connie; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#74 Chukar Butte / Chukar Butte / Chukar Butte (Dog On-Leash) / Sweet Connie` until `signed junction with Chukar Butte Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Peggy's ~0m; #76 Peggy's (path) ~1m; Sweet Connie ~15m; #77 Sweet Connie (path) ~24m
- Decision as runner: Follow #74 Chukar Butte / Chukar Butte / Chukar Butte (Dog On-Leash) / Sweet Connie until signed junction with Chukar Butte Trail; target is Chukar Butte Trail.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 04: TAKE Chukar Butte Trail

- Physical role: signed junction with Cartwright Connector
- Model frame: The packet says `04 7.16 mi (+4.82) JCT TAKE Chukar Butte Trail UNTIL signed junction with Cartwright Connector.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#74 Chukar Butte, #75 Currant Creek); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #74 Chukar Butte, #75 Currant Creek; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Chukar Butte Trail` until `signed junction with Cartwright Connector`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #74 Chukar Butte (path) ~1m; Chukar Butte ~3m; Currant Creek ~44m; #75 Currant Creek (path) ~45m; Chukar Butte (Dog On-Leash) ~45m
- Decision as runner: Follow Chukar Butte Trail until signed junction with Cartwright Connector; target is Cartwright Connector.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 05: FOLLOW #74 Chukar Butte / Chukar Butte (Dog On-Leash) / North Cartwright Road / West Hidden Springs Drive

- Physical role: signed junction with Cartwright Connector
- Model frame: The packet says `05 11.98 mi (+0.93) ROAD FOLLOW #74 Chukar Butte / Chukar Butte (Dog On-Leash) / North Cartwright Road / West Hidden Springs Drive UNTIL signed junction with Cartwright Connector.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#77 Sweet Connie); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #77 Sweet Connie, #74 Chukar Butte; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#74 Chukar Butte / Chukar Butte (Dog On-Leash) / North Cartwright Road / West Hidden Springs Drive` until `signed junction with Cartwright Connector`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Chukar Butte ~2m; #77 Sweet Connie (path) ~7m; Sweet Connie ~8m; #74 Chukar Butte (path) ~24m
- Decision as runner: Follow #74 Chukar Butte / Chukar Butte (Dog On-Leash) / North Cartwright Road / West Hidden Springs Drive until signed junction with Cartwright Connector; target is Cartwright Connector.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 06: CONTINUE STRAIGHT Cartwright Connector

- Physical role: signed junction with Cartwright Ridge
- Model frame: The packet says `06 12.91 mi (+1.70) JCT CONTINUE STRAIGHT Cartwright Connector UNTIL signed junction with Cartwright Ridge.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (OSM footway connector 63119, #52 Bill's, OSM footway connector 63120, OSM footway connector 95409, OSM footway connector 63132); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: OSM footway connector 63119, #52 Bill's, OSM footway connector 63120, OSM footway connector 95409, OSM footway connector 63132; vehicle corridor or service/residential road context: North Cartwright Road, West Hidden Springs Drive, West Antelope View Drive; the branch to privilege is `Cartwright Connector` until `signed junction with Cartwright Ridge`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: North Cartwright Road (tertiary) ~1m; West Hidden Springs Drive (tertiary) ~2m; OSM footway connector 63119 (footway) ~18m; #52 Bill's (path) ~20m; OSM footway connector 63120 (footway) ~20m; OSM footway connector 95409 (footway) ~21m
- Decision as runner: Follow Cartwright Connector until signed junction with Cartwright Ridge; target is Cartwright Ridge.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 07: TURN AROUND Cartwright Ridge

- Physical role: signed junction with CHBH Connector
- Model frame: The packet says `07 14.61 mi (+1.76) JCT TURN AROUND Cartwright Ridge UNTIL signed junction with CHBH Connector.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#52 Bill's, #84 Cartwright Ridge, OSM path connector 106925); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #52 Bill's, #84 Cartwright Ridge, OSM path connector 106925; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Cartwright Ridge` until `signed junction with CHBH Connector`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #52 Bill's (path) ~1m; #84 Cartwright Ridge (path) ~27m; Cartwright Ridge ~27m; OSM path connector 106925 (path) ~40m
- Decision as runner: Follow Cartwright Ridge until signed junction with CHBH Connector; target is CHBH Connector.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 08: FOLLOW #81 Polecat Loop / #82 Doe Ridge / #83 Quick Draw / #84 Cartwright Ridge / Cartwright Ridge / Doe Ridge / North Cartwright Road / OSM service connector 69243 / OSM track connector 110342 / Polecat Loop / Polecat Loop (STM) / Quick Draw

- Physical role: signed junction with CHBH Connector
- Model frame: The packet says `08 16.37 mi (+2.78) ROAD FOLLOW #81 Polecat Loop / #82 Doe Ridge / #83 Quick Draw / #84 Cartwright Ridge / Cartwright Ridge / Doe Ridge / North Cartwright Road / OSM service connector 69243 / OSM track connector 110342 / Polecat Loop / Polecat Loop (STM) / Quick Draw UNTIL signed junction with CHBH Connector.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#52 Bill's, OSM path connector 106925); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #52 Bill's, #84 Cartwright Ridge, OSM path connector 106925; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#81 Polecat Loop / #82 Doe Ridge / #83 Quick Draw / #84 Cartwright Ridge / Cartwright Ridge / Doe Ridge / North Cartwright Road / OSM service connector 69243 / OSM track connector 110342 / Polecat Loop / Polecat Loop (STM) / Quick Draw` until `signed junction with CHBH Connector`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #52 Bill's (path) ~1m; #84 Cartwright Ridge (path) ~27m; Cartwright Ridge ~27m; OSM path connector 106925 (path) ~40m
- Decision as runner: Follow #81 Polecat Loop / #82 Doe Ridge / #83 Quick Draw / #84 Cartwright Ridge / Cartwright Ridge / Doe Ridge / North Cartwright Road / OSM service connector 69243 / OSM track connector 110342 / Polecat Loop / Polecat Loop (STM) / Quick Draw until signed junction with CHBH Connector; target is CHBH Connector.
- Wrong-layer risk: generic OSM connector name may not exist on signs
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 09: TAKE CHBH Connector

- Physical role: end of CHBH Connector for this route
- Model frame: The packet says `09 19.15 mi (+0.81) JCT TAKE CHBH Connector UNTIL end of CHBH Connector for this route.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (OSM path connector 13996, #81 Polecat Loop, #83 Quick Draw); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: OSM path connector 13996, #81 Polecat Loop, #83 Quick Draw; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `CHBH Connector` until `end of CHBH Connector for this route`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Polecat Loop (STM) ~1m; OSM path connector 13996 (path) ~27m; #81 Polecat Loop (path) ~169m; #83 Quick Draw (path) ~169m; Polecat Loop ~169m; Quick Draw ~169m
- Decision as runner: Follow CHBH Connector until end of CHBH Connector for this route; target is return to car.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 10: FOLLOW #81 Polecat Loop / #82 Doe Ridge / #83 Quick Draw / Doe Ridge / OSM path connector 110670 / OSM path connector 13996 / OSM track connector 106014 / OSM track connector 106931 / Polecat Loop (STM) / Quick Draw

- Physical role: parked car / trailhead
- Model frame: The packet says `10 19.96 mi (+1.83) EXIT FOLLOW #81 Polecat Loop / #82 Doe Ridge / #83 Quick Draw / Doe Ridge / OSM path connector 110670 / OSM path connector 13996 / OSM track connector 106014 / OSM track connector 106931 / Polecat Loop (STM) / Quick Draw UNTIL parked car / trailhead.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (OSM path connector 83301, OSM path connector 13997, OSM path connector 83299, OSM path connector 83298, OSM path connector 83300); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: OSM path connector 13996, OSM path connector 83301, OSM path connector 13997, OSM path connector 83299, OSM path connector 83298; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#81 Polecat Loop / #82 Doe Ridge / #83 Quick Draw / Doe Ridge / OSM path connector 110670 / OSM path connector 13996 / OSM track connector 106014 / OSM track connector 106931 / Polecat Loop (STM) / Quick Draw` until `parked car / trailhead`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM path connector 13996 (path) ~2m; OSM path connector 83301 (path) ~5m; OSM path connector 13997 (path) ~47m; OSM path connector 83299 (path) ~76m; OSM path connector 83298 (path) ~79m; OSM path connector 83300 (path) ~133m
- Decision as runner: Follow #81 Polecat Loop / #82 Doe Ridge / #83 Quick Draw / Doe Ridge / OSM path connector 110670 / OSM path connector 13996 / OSM track connector 106014 / OSM track connector 106931 / Polecat Loop (STM) / Quick Draw until parked car / trailhead; target is Cartwright Trailhead.
- Wrong-layer risk: generic OSM connector name may not exist on signs; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Finish / return to car

- Physical role: Route endpoint / return-to-car point.
- Model frame: The packet endpoint closes the loop.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
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

- #74 Chukar Butte (~1m, path, highway=path, source=openstreetmap)
- OSM service connector 69243 (~2m, vehicle, highway=service, source=openstreetmap)
- West Hidden Springs Drive (~2m, vehicle, highway=tertiary, source=openstreetmap)
- Chukar Butte (~3m, feature, source=ridge_to_rivers_open_data)
- OSM path connector 83301 (~5m, path, highway=path, source=openstreetmap)
- Polecat Loop (~7m, feature, source=ridge_to_rivers_open_data)
- Sweet Connie (~15m, feature, source=ridge_to_rivers_open_data)
- OSM footway connector 63119 (~18m, path, highway=footway, source=openstreetmap)
- #52 Bill's (~20m, path, highway=path, source=openstreetmap)
- OSM footway connector 63120 (~20m, path, highway=footway, source=openstreetmap)
- OSM footway connector 95409 (~21m, path, highway=footway, source=openstreetmap)
- North Cartwright Road (~22m, vehicle, highway=tertiary, source=openstreetmap)
- #77 Sweet Connie (~24m, path, highway=path, source=openstreetmap)
- #84 Cartwright Ridge (~27m, path, highway=path, source=openstreetmap)
- Cartwright Ridge (~27m, feature, source=ridge_to_rivers_open_data)
- OSM path connector 13996 (~27m, path, highway=path, source=openstreetmap)
- #76 Peggy's (~31m, path, highway=path, source=openstreetmap)
- Peggy's (~34m, feature, source=ridge_to_rivers_open_data)
- OSM path connector 106925 (~40m, path, highway=path, source=openstreetmap)
- OSM track connector 106014 (~40m, vehicle, highway=track, source=openstreetmap)

## Required Next Proof

- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.
- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.
- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.

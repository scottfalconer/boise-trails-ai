# Runner-Perspective Frame Shift: 14 - Orchard Gulch

## Frame Contract

- Route card: `14` / outing `14-1`.
- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.
- Evidence used: `docs/field-packet/field-tool-data.json`, `docs/field-packet/gpx/official/14-orchard-gulch-orchard-gulch-trail-five-mile-gulch-trail-watchman-trail.gpx`, R2R open data, OSM connector overlay, official 2026 segment source.
- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.
- Frame decision: `needs-proof`. The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.
- Access status: known-or-mapped parking in packet data.
- Human-validity status for this audit: `needs_visual_proof`.

## Route Snapshot

- Trails: Orchard Gulch Trail, Five Mile Gulch Trail, Watchman Trail.
- Official miles: 8.45; on-foot miles: 10.74.
- Door-to-door: p75 242 min; p90 272 min.
- Segment count: 6; wayfinding cue count: 6.

## Start-End-Junction Frame Shifts

### Start

- Physical role: Park here and start this outing.
- Model frame: The packet proves the route has a start coordinate and a first cue.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: #7 Orchard Gulch; vehicle corridor or service/residential road context: East Shaw Mountain Road; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: East Shaw Mountain Road (unclassified) ~9m; #7 Orchard Gulch (path) ~26m; Orchard Gulch ~29m
- Decision as runner: Before moving, find the first signed trail or road-access line and confirm the car is parked where the return cue can actually resolve.
- Wrong-layer risk: start/finish access can fail even when route geometry passes
- Evidence boundary: cue GPX waypoint plus local R2R/OSM overlay; no live imagery or field photo in this audit

### Cue 01: FOLLOW #7 Orchard Gulch

- Physical role: signed junction with Orchard Gulch Trail
- Model frame: The packet says `01 0.00 mi (+0.00) START/ACCESS FOLLOW #7 Orchard Gulch UNTIL signed junction with Orchard Gulch Trail.`.
- Runner frame: Runner frame: the immediate job is to keep the current trail until the named junction/landmark, with no extra branch proven by local data at this checkpoint.
- Likely visual field: mapped trail/path choices near you: #7 Orchard Gulch; vehicle corridor or service/residential road context: East Shaw Mountain Road; the branch to privilege is `#7 Orchard Gulch` until `signed junction with Orchard Gulch Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: East Shaw Mountain Road (unclassified) ~3m; #7 Orchard Gulch (path) ~12m; Orchard Gulch ~15m
- Decision as runner: Follow #7 Orchard Gulch until signed junction with Orchard Gulch Trail; target is Orchard Gulch Trail.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 02: FOLLOW Orchard Gulch Trail

- Physical role: signed junction with Five Mile Gulch Trail
- Model frame: The packet says `02 0.00 mi (+1.58) FOLLOW Orchard Gulch Trail UNTIL signed junction with Five Mile Gulch Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#7 Orchard Gulch); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #7 Orchard Gulch; vehicle corridor or service/residential road context: East Shaw Mountain Road; the branch to privilege is `Orchard Gulch Trail` until `signed junction with Five Mile Gulch Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: East Shaw Mountain Road (unclassified) ~3m; #7 Orchard Gulch (path) ~12m; Orchard Gulch ~15m
- Decision as runner: Follow Orchard Gulch Trail until signed junction with Five Mile Gulch Trail; target is Five Mile Gulch Trail.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 03: TURN LEFT Five Mile Gulch Trail

- Physical role: signed junction with Watchman Trail
- Model frame: The packet says `03 1.58 mi (+3.38) JCT TURN LEFT Five Mile Gulch Trail UNTIL signed junction with Watchman Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#7 Orchard Gulch, #2 Five Mile Gulch); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #7 Orchard Gulch, #2 Five Mile Gulch; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Five Mile Gulch Trail` until `signed junction with Watchman Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Orchard Gulch ~1m; #7 Orchard Gulch (path) ~2m; #2 Five Mile Gulch (path) ~20m; Five Mile Gulch ~20m
- Decision as runner: Follow Five Mile Gulch Trail until signed junction with Watchman Trail; target is Watchman Trail.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 04: FOLLOW #26 Three Bears / East Shaw Mountain Road / Five Mile Gulch / Three Bears

- Physical role: signed junction with Watchman Trail
- Model frame: The packet says `04 4.96 mi (+0.75) ROAD FOLLOW #26 Three Bears / East Shaw Mountain Road / Five Mile Gulch / Three Bears UNTIL signed junction with Watchman Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#2 Five Mile Gulch, #3 Watchman); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #2 Five Mile Gulch, #3 Watchman; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#26 Three Bears / East Shaw Mountain Road / Five Mile Gulch / Three Bears` until `signed junction with Watchman Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #2 Five Mile Gulch (path) ~0m; Five Mile Gulch ~0m; #3 Watchman (path) ~45m; Watchman ~46m
- Decision as runner: Follow #26 Three Bears / East Shaw Mountain Road / Five Mile Gulch / Three Bears until signed junction with Watchman Trail; target is Watchman Trail.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 05: TAKE Watchman Trail

- Physical role: end of Watchman Trail for this route
- Model frame: The packet says `05 5.71 mi (+3.47) JCT TAKE Watchman Trail UNTIL end of Watchman Trail for this route.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#2 Five Mile Gulch, #3 Watchman); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #2 Five Mile Gulch, #3 Watchman; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Watchman Trail` until `end of Watchman Trail for this route`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Five Mile Gulch ~1m; #2 Five Mile Gulch (path) ~9m; #3 Watchman (path) ~42m; Watchman ~42m
- Decision as runner: Follow Watchman Trail until end of Watchman Trail for this route; target is return to car.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 06: FOLLOW #2 Five Mile Gulch / #7 Orchard Gulch / Five Mile Gulch / Orchard Gulch

- Physical role: parked car / trailhead
- Model frame: The packet says `06 9.18 mi (+2.09) EXIT FOLLOW #2 Five Mile Gulch / #7 Orchard Gulch / Five Mile Gulch / Orchard Gulch UNTIL parked car / trailhead.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#6 Femrite's Patrol Trail, #3 Watchman, #45 Curlew Connection); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #6 Femrite's Patrol Trail, #3 Watchman, #45 Curlew Connection; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#2 Five Mile Gulch / #7 Orchard Gulch / Five Mile Gulch / Orchard Gulch` until `parked car / trailhead`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Watchman ~0m; Femrite's Patrol ~38m; #6 Femrite's Patrol Trail (path) ~39m; Curlew Connection ~115m; #3 Watchman (path) ~122m; #45 Curlew Connection (path) ~122m
- Decision as runner: Follow #2 Five Mile Gulch / #7 Orchard Gulch / Five Mile Gulch / Orchard Gulch until parked car / trailhead; target is Orchard Gulch Trail Access Point.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Finish / return to car

- Physical role: Route endpoint / return-to-car point.
- Model frame: The packet endpoint closes the loop.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: #7 Orchard Gulch; vehicle corridor or service/residential road context: East Shaw Mountain Road; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: East Shaw Mountain Road (unclassified) ~3m; #7 Orchard Gulch (path) ~12m; Orchard Gulch ~15m
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

- East Shaw Mountain Road (~9m, vehicle, highway=unclassified, source=openstreetmap)
- #2 Five Mile Gulch (~20m, path, highway=path, source=openstreetmap)
- Five Mile Gulch (~20m, feature, source=ridge_to_rivers_open_data)
- #7 Orchard Gulch (~26m, path, highway=path, source=openstreetmap)
- Orchard Gulch (~29m, feature, source=ridge_to_rivers_open_data)
- Femrite's Patrol (~38m, feature, source=ridge_to_rivers_open_data)
- #6 Femrite's Patrol Trail (~39m, path, highway=path, source=openstreetmap)
- #3 Watchman (~45m, path, highway=path, source=openstreetmap)
- Watchman (~46m, feature, source=ridge_to_rivers_open_data)
- Curlew Connection (~115m, feature, source=ridge_to_rivers_open_data)
- #45 Curlew Connection (~122m, path, highway=path, source=openstreetmap)

## Required Next Proof

- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.
- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.
- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.

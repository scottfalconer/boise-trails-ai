# Runner-Perspective Frame Shift: 13 - Freestone Creek

## Frame Contract

- Route card: `13` / outing `13-1`.
- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.
- Evidence used: `docs/field-packet/field-tool-data.json`, `docs/field-packet/gpx/official/13-freestone-creek-three-bears-trail-femrite-s-patrol-freestone-ridge-two-point-shane-s-tr.gpx`, R2R open data, OSM connector overlay, official 2026 segment source.
- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.
- Frame decision: `needs-proof`. The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.
- Access status: known-or-mapped parking in packet data.
- Human-validity status for this audit: `needs_visual_proof`.

## Route Snapshot

- Trails: Three Bears Trail, Femrite's Patrol, Freestone Ridge, Two Point, Shane's Trail, Shane's Connector, Fat Tire Traverse, Curlew Connection.
- Official miles: 14.35; on-foot miles: 25.12.
- Door-to-door: p75 490 min; p90 549 min.
- Segment count: 16; wayfinding cue count: 15.

## Start-End-Junction Frame Shifts

### Start

- Physical role: Park here and start this outing.
- Model frame: The packet proves the route has a start coordinate and a first cue.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: #22A Access Trail (Central Ridge), #22C Mountain Cove, Access Trail (Central Ridge), #22B Freestone Creek; vehicle corridor or service/residential road context: OSM service connector 14900, North Mountain Cove Road; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 14900 (service) ~4m; #22A Access Trail (Central Ridge) (path) ~10m; #22C Mountain Cove (path) ~14m; Mountain Cove ~14m; Access Trail (Central Ridge) ~15m; North Mountain Cove Road (unclassified) ~17m
- Decision as runner: Before moving, find the first signed trail or road-access line and confirm the car is parked where the return cue can actually resolve.
- Wrong-layer risk: start/finish access can fail even when route geometry passes; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: cue GPX waypoint plus local R2R/OSM overlay; no live imagery or field photo in this audit

### Cue 01: FOLLOW #22C Mountain Cove

- Physical role: signed junction with Three Bears Trail
- Model frame: The packet says `01 0.00 mi (+0.79) START/ACCESS FOLLOW #22C Mountain Cove UNTIL signed junction with Three Bears Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#22A Access Trail (Central Ridge), Access Trail (Central Ridge), #22B Freestone Creek); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #22C Mountain Cove, #22A Access Trail (Central Ridge), Access Trail (Central Ridge), #22B Freestone Creek; vehicle corridor or service/residential road context: OSM service connector 14900, North Mountain Cove Road; the branch to privilege is `#22C Mountain Cove` until `signed junction with Three Bears Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #22C Mountain Cove (path) ~4m; Mountain Cove ~5m; OSM service connector 14900 (service) ~6m; #22A Access Trail (Central Ridge) (path) ~8m; Access Trail (Central Ridge) ~8m; North Mountain Cove Road (unclassified) ~26m
- Decision as runner: Follow #22C Mountain Cove until signed junction with Three Bears Trail; target is Three Bears Trail.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 02: FOLLOW Three Bears Trail

- Physical role: signed junction with Femrite's Patrol
- Model frame: The packet says `02 0.79 mi (+4.70) FOLLOW Three Bears Trail UNTIL signed junction with Femrite's Patrol.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (OSM path connector 11312, #20 Ridgecrest, #26 Three Bears, #22A Central Ridge Spur (North)); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: OSM path connector 11312, #20 Ridgecrest, #26 Three Bears, #22A Central Ridge Spur (North); vehicle corridor or service/residential road context: OSM unclassified connector 11309, #26 Three Bears; the branch to privilege is `Three Bears Trail` until `signed junction with Femrite's Patrol`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM path connector 11312 (path) ~1m; OSM unclassified connector 11309 (unclassified) ~8m; Ridge Crest ~33m; #20 Ridgecrest (path) ~34m; Three Bears ~40m; #26 Three Bears (track) ~42m
- Decision as runner: Follow Three Bears Trail until signed junction with Femrite's Patrol; target is Femrite's Patrol.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 03: FOLLOW #26 Three Bears / #3 Watchman / Three Bears / Watchman

- Physical role: signed junction with Femrite's Patrol
- Model frame: The packet says `03 5.49 mi (+1.10) CONNECTOR FOLLOW #26 Three Bears / #3 Watchman / Three Bears / Watchman UNTIL signed junction with Femrite's Patrol.`.
- Runner frame: Runner frame: the immediate job is to keep the current trail until the named junction/landmark, with no extra branch proven by local data at this checkpoint.
- Likely visual field: mapped trail/path choices near you: #26 Three Bears; vehicle corridor or service/residential road context: East Shaw Mountain Road; the branch to privilege is `#26 Three Bears / #3 Watchman / Three Bears / Watchman` until `signed junction with Femrite's Patrol`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #26 Three Bears (path) ~1m; Three Bears ~1m; East Shaw Mountain Road (unclassified) ~18m
- Decision as runner: Follow #26 Three Bears / #3 Watchman / Three Bears / Watchman until signed junction with Femrite's Patrol; target is Femrite's Patrol.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 04: TAKE Femrite's Patrol

- Physical role: signed junction with Freestone Ridge
- Model frame: The packet says `04 6.59 mi (+0.06) JCT TAKE Femrite's Patrol UNTIL signed junction with Freestone Ridge.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#6 Femrite's Patrol Trail, #3 Watchman, #45 Curlew Connection); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #6 Femrite's Patrol Trail, #3 Watchman, #45 Curlew Connection; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Femrite's Patrol` until `signed junction with Freestone Ridge`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Watchman ~3m; Femrite's Patrol ~39m; #6 Femrite's Patrol Trail (path) ~40m; Curlew Connection ~118m; #3 Watchman (path) ~124m; #45 Curlew Connection (path) ~124m
- Decision as runner: Follow Femrite's Patrol until signed junction with Freestone Ridge; target is Freestone Ridge.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 05: FOLLOW #6 Femrite's Patrol Trail / Curlew Connection / Femrite's Patrol

- Physical role: signed junction with Freestone Ridge
- Model frame: The packet says `05 6.65 mi (+1.01) CONNECTOR FOLLOW #6 Femrite's Patrol Trail / Curlew Connection / Femrite's Patrol UNTIL signed junction with Freestone Ridge.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#3 Watchman, #45 Curlew Connection); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #6 Femrite's Patrol Trail, #3 Watchman, #45 Curlew Connection; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#6 Femrite's Patrol Trail / Curlew Connection / Femrite's Patrol` until `signed junction with Freestone Ridge`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Watchman ~0m; Femrite's Patrol ~1m; #6 Femrite's Patrol Trail (path) ~2m; Curlew Connection ~43m; #3 Watchman (path) ~50m; #45 Curlew Connection (path) ~50m
- Decision as runner: Follow #6 Femrite's Patrol Trail / Curlew Connection / Femrite's Patrol until signed junction with Freestone Ridge; target is Freestone Ridge.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 06: TAKE Freestone Ridge

- Physical role: signed junction with Two Point
- Model frame: The packet says `06 7.66 mi (+2.01) JCT TAKE Freestone Ridge UNTIL signed junction with Two Point.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#45 Curlew Connection, #6 Femrite's Patrol Trail, #5 Freestone Ridge); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #45 Curlew Connection, #6 Femrite's Patrol Trail, #5 Freestone Ridge; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Freestone Ridge` until `signed junction with Two Point`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #45 Curlew Connection (path) ~2m; Curlew Connection ~2m; #6 Femrite's Patrol Trail (path) ~35m; Femrite's Patrol ~37m; #5 Freestone Ridge (path) ~43m; Freestone Ridge ~45m
- Decision as runner: Follow Freestone Ridge until signed junction with Two Point; target is Two Point.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 07: FOLLOW #20 Ridgecrest / #26 Three Bears / OSM path connector 11312 / OSM unclassified connector 11309 / Ridge Crest / Three Bears

- Physical role: signed junction with Two Point
- Model frame: The packet says `07 9.67 mi (+0.67) ROAD FOLLOW #20 Ridgecrest / #26 Three Bears / OSM path connector 11312 / OSM unclassified connector 11309 / Ridge Crest / Three Bears UNTIL signed junction with Two Point.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#5 Freestone Ridge, #42 Fat Tire Traverse, #45 Curlew Connection); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #5 Freestone Ridge, #42 Fat Tire Traverse, #45 Curlew Connection; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#20 Ridgecrest / #26 Three Bears / OSM path connector 11312 / OSM unclassified connector 11309 / Ridge Crest / Three Bears` until `signed junction with Two Point`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #5 Freestone Ridge (path) ~2m; Freestone Ridge ~2m; Curlew Connection ~39m; Fat Tire Traverse ~39m; #42 Fat Tire Traverse (path) ~42m; #45 Curlew Connection (path) ~42m
- Decision as runner: Follow #20 Ridgecrest / #26 Three Bears / OSM path connector 11312 / OSM unclassified connector 11309 / Ridge Crest / Three Bears until signed junction with Two Point; target is Two Point.
- Wrong-layer risk: generic OSM connector name may not exist on signs
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 08: TAKE Two Point

- Physical role: signed junction with Shane's Trail
- Model frame: The packet says `08 10.34 mi (+1.20) JCT TAKE Two Point UNTIL signed junction with Shane's Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#20 Ridgecrest, #44 Two Point, #22A Central Ridge Spur (North), OSM path connector 11312); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #20 Ridgecrest, #44 Two Point, #22A Central Ridge Spur (North), OSM path connector 11312; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Two Point` until `signed junction with Shane's Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #20 Ridgecrest (path) ~3m; Bucktail ~35m; #44 Two Point (footway) ~40m; Two Point ~40m; Central Ridge Spur (North) ~104m; #22A Central Ridge Spur (North) (path) ~106m
- Decision as runner: Follow Two Point until signed junction with Shane's Trail; target is Shane's Trail.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 09: TURN LEFT Shane's Trail

- Physical role: signed junction with Shane's Connector
- Model frame: The packet says `09 11.54 mi (+1.84) JCT TURN LEFT Shane's Trail UNTIL signed junction with Shane's Connector.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#44 Two Point, #22 Central Ridge, #26A Shane's); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #44 Two Point, #22 Central Ridge, #26A Shane's; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Shane's Trail` until `signed junction with Shane's Connector`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Two Point ~0m; #44 Two Point (footway) ~1m; Bucktail ~20m; #22 Central Ridge (path) ~26m; Central Ridge ~27m; Shane's ~37m
- Decision as runner: Follow Shane's Trail until signed junction with Shane's Connector; target is Shane's Connector.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; wrong-direction choice has meaningful climb penalty
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 10: TAKE Shane's Connector

- Physical role: signed junction with Fat Tire Traverse
- Model frame: The packet says `10 13.38 mi (+0.44) JCT TAKE Shane's Connector UNTIL signed junction with Fat Tire Traverse.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#26A Shane's); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #26A Shane's; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Shane's Connector` until `signed junction with Fat Tire Traverse`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Shane's ~38m; #26A Shane's (path) ~44m
- Decision as runner: Follow Shane's Connector until signed junction with Fat Tire Traverse; target is Fat Tire Traverse.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 11: FOLLOW #26 Three Bears / #26A Shane's / #5 Freestone Ridge / Freestone Ridge / Shane's / Three Bears

- Physical role: signed junction with Fat Tire Traverse
- Model frame: The packet says `11 13.82 mi (+3.22) ROAD FOLLOW #26 Three Bears / #26A Shane's / #5 Freestone Ridge / Freestone Ridge / Shane's / Three Bears UNTIL signed junction with Fat Tire Traverse.`.
- Runner frame: Runner frame: the immediate job is to keep the current trail until the named junction/landmark, with no extra branch proven by local data at this checkpoint.
- Likely visual field: mapped trail/path choices near you: #26A Shane's; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#26 Three Bears / #26A Shane's / #5 Freestone Ridge / Freestone Ridge / Shane's / Three Bears` until `signed junction with Fat Tire Traverse`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #26A Shane's (path) ~1m; Shane's ~37m
- Decision as runner: Follow #26 Three Bears / #26A Shane's / #5 Freestone Ridge / Freestone Ridge / Shane's / Three Bears until signed junction with Fat Tire Traverse; target is Fat Tire Traverse.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 12: TAKE Fat Tire Traverse

- Physical role: signed junction with Curlew Connection
- Model frame: The packet says `12 17.04 mi (+1.20) JCT TAKE Fat Tire Traverse UNTIL signed junction with Curlew Connection.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#5 Freestone Ridge, #42 Fat Tire Traverse, #45 Curlew Connection); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #5 Freestone Ridge, #42 Fat Tire Traverse, #45 Curlew Connection; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Fat Tire Traverse` until `signed junction with Curlew Connection`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #5 Freestone Ridge (path) ~2m; Freestone Ridge ~2m; Curlew Connection ~40m; Fat Tire Traverse ~40m; #42 Fat Tire Traverse (path) ~42m; #45 Curlew Connection (path) ~42m
- Decision as runner: Follow Fat Tire Traverse until signed junction with Curlew Connection; target is Curlew Connection.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 13: FOLLOW #42 Fat Tire Traverse / 8th Street Motorcycle / Curlew Connection / Fat Tire Traverse

- Physical role: signed junction with Curlew Connection
- Model frame: The packet says `13 18.24 mi (+0.69) CONNECTOR FOLLOW #42 Fat Tire Traverse / 8th Street Motorcycle / Curlew Connection / Fat Tire Traverse UNTIL signed junction with Curlew Connection.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#4 8th Street Motorcycle); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #42 Fat Tire Traverse, #4 8th Street Motorcycle; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#42 Fat Tire Traverse / 8th Street Motorcycle / Curlew Connection / Fat Tire Traverse` until `signed junction with Curlew Connection`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #42 Fat Tire Traverse (path) ~2m; Fat Tire Traverse ~2m; #4 8th Street Motorcycle (path) ~41m; 8th Street Motorcycle ~43m
- Decision as runner: Follow #42 Fat Tire Traverse / 8th Street Motorcycle / Curlew Connection / Fat Tire Traverse until signed junction with Curlew Connection; target is Curlew Connection.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 14: TURN RIGHT Curlew Connection

- Physical role: end of Curlew Connection for this route
- Model frame: The packet says `14 18.93 mi (+2.90) JCT TURN RIGHT Curlew Connection UNTIL end of Curlew Connection for this route.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#42 Fat Tire Traverse, #4 8th Street Motorcycle); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #42 Fat Tire Traverse, #4 8th Street Motorcycle; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Curlew Connection` until `end of Curlew Connection for this route`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #42 Fat Tire Traverse (path) ~2m; Fat Tire Traverse ~2m; #4 8th Street Motorcycle (path) ~41m; 8th Street Motorcycle ~43m
- Decision as runner: Follow Curlew Connection until end of Curlew Connection for this route; target is return to car.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 15: FOLLOW #26 Three Bears / #45 Curlew Connection / #5 Freestone Ridge / Curlew Connection / Freestone Ridge / Three Bears

- Physical role: parked car / trailhead
- Model frame: The packet says `15 21.83 mi (+3.74) EXIT FOLLOW #26 Three Bears / #45 Curlew Connection / #5 Freestone Ridge / Curlew Connection / Freestone Ridge / Three Bears UNTIL parked car / trailhead.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#42 Fat Tire Traverse); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #42 Fat Tire Traverse, #5 Freestone Ridge, #45 Curlew Connection; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#26 Three Bears / #45 Curlew Connection / #5 Freestone Ridge / Curlew Connection / Freestone Ridge / Three Bears` until `parked car / trailhead`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #42 Fat Tire Traverse (path) ~1m; Fat Tire Traverse ~2m; #5 Freestone Ridge (path) ~11m; Freestone Ridge ~14m; #45 Curlew Connection (path) ~29m; Curlew Connection ~31m
- Decision as runner: Follow #26 Three Bears / #45 Curlew Connection / #5 Freestone Ridge / Curlew Connection / Freestone Ridge / Three Bears until parked car / trailhead; target is Freestone Creek Trailhead.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Finish / return to car

- Physical role: Route endpoint / return-to-car point.
- Model frame: The packet endpoint closes the loop.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: #22C Mountain Cove, #22A Access Trail (Central Ridge), Access Trail (Central Ridge), #22B Freestone Creek; vehicle corridor or service/residential road context: OSM service connector 14900, North Mountain Cove Road; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #22C Mountain Cove (path) ~4m; Mountain Cove ~5m; OSM service connector 14900 (service) ~6m; #22A Access Trail (Central Ridge) (path) ~8m; Access Trail (Central Ridge) ~8m; North Mountain Cove Road (unclassified) ~26m
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

- OSM path connector 11312 (~1m, path, highway=path, source=openstreetmap)
- Watchman (~3m, feature, source=ridge_to_rivers_open_data)
- OSM service connector 14900 (~4m, vehicle, highway=service, source=openstreetmap)
- OSM unclassified connector 11309 (~8m, vehicle, highway=unclassified, source=openstreetmap)
- #22A Access Trail (Central Ridge) (~10m, path, highway=path, source=openstreetmap)
- #22C Mountain Cove (~14m, path, highway=path, source=openstreetmap)
- Mountain Cove (~14m, feature, source=ridge_to_rivers_open_data)
- Access Trail (Central Ridge) (~15m, path, source=ridge_to_rivers_open_data)
- North Mountain Cove Road (~17m, vehicle, highway=unclassified, source=openstreetmap)
- East Shaw Mountain Road (~18m, vehicle, highway=unclassified, source=openstreetmap)
- #22 Central Ridge (~26m, path, highway=path, source=openstreetmap)
- Central Ridge (~27m, feature, source=ridge_to_rivers_open_data)
- Ridge Crest (~33m, feature, source=ridge_to_rivers_open_data)
- #20 Ridgecrest (~34m, path, highway=path, source=openstreetmap)
- Bucktail (~35m, feature, source=ridge_to_rivers_open_data)
- Shane's (~37m, feature, source=ridge_to_rivers_open_data)
- #26A Shane's (~38m, path, highway=path, source=openstreetmap)
- Fat Tire Traverse (~39m, feature, source=ridge_to_rivers_open_data)
- Femrite's Patrol (~39m, feature, source=ridge_to_rivers_open_data)
- #44 Two Point (~40m, path, highway=footway, source=openstreetmap)

## Required Next Proof

- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.
- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.
- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.

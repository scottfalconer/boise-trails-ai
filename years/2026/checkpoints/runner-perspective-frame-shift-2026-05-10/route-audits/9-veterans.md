# Runner-Perspective Frame Shift: 9 - Veterans

## Frame Contract

- Route card: `9` / outing `9-1`.
- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.
- Evidence used: `docs/field-packet/field-tool-data.json`, `docs/field-packet/gpx/official/9-veterans-veterans-big-springs-rabbit-run-d-s-chaos-rei-connection.gpx`, R2R open data, OSM connector overlay, official 2026 segment source.
- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.
- Frame decision: `needs-proof`. The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.
- Access status: known-or-mapped parking in packet data.
- Human-validity status for this audit: `needs_visual_proof`.

## Route Snapshot

- Trails: Veterans, Big Springs, Rabbit Run, D's Chaos, REI Connection.
- Official miles: 4.68; on-foot miles: 5.78.
- Door-to-door: p75 180 min; p90 202 min.
- Segment count: 13; wayfinding cue count: 10.

## Start-End-Junction Frame Shifts

### Start

- Physical role: Park here and start this outing.
- Model frame: The packet proves the route has a start coordinate and a first cue.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: Veteran's Trail; vehicle corridor or service/residential road context: OSM service connector 12725, OSM service connector 12966, North Dry Creek Cemetery Road, OSM service connector 229; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 12725 (service) ~1m; Veteran's Trail (path) ~5m; Veterans (Dog On-Leash) ~7m; OSM service connector 12966 (service) ~51m; North Dry Creek Cemetery Road (unclassified) ~62m; OSM service connector 229 (service) ~144m
- Decision as runner: Before moving, find the first signed trail or road-access line and confirm the car is parked where the return cue can actually resolve.
- Wrong-layer risk: start/finish access can fail even when route geometry passes
- Evidence boundary: cue GPX waypoint plus local R2R/OSM overlay; no live imagery or field photo in this audit

### Cue 01: FOLLOW #114 Veterans (Dog On-Leash)

- Physical role: signed junction with Veterans
- Model frame: The packet says `01 0.00 mi (+0.00) START/ACCESS FOLLOW #114 Veterans (Dog On-Leash) UNTIL signed junction with Veterans.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (Veteran's Trail); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: Veteran's Trail; vehicle corridor or service/residential road context: OSM service connector 12725, OSM service connector 12966, North Dry Creek Cemetery Road, OSM service connector 229; the branch to privilege is `#114 Veterans (Dog On-Leash)` until `signed junction with Veterans`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 12725 (service) ~1m; Veterans (Dog On-Leash) ~8m; Veteran's Trail (path) ~9m; OSM service connector 12966 (service) ~51m; North Dry Creek Cemetery Road (unclassified) ~62m; OSM service connector 229 (service) ~149m
- Decision as runner: Follow #114 Veterans (Dog On-Leash) until signed junction with Veterans; target is Veterans.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 02: FOLLOW Veterans

- Physical role: signed junction with Big Springs
- Model frame: The packet says `02 0.00 mi (+1.02) FOLLOW Veterans UNTIL signed junction with Big Springs.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (Veteran's Trail); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: Veteran's Trail; vehicle corridor or service/residential road context: OSM service connector 12725, OSM service connector 12966, North Dry Creek Cemetery Road, OSM service connector 229; the branch to privilege is `Veterans` until `signed junction with Big Springs`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 12725 (service) ~1m; Veterans (Dog On-Leash) ~8m; Veteran's Trail (path) ~9m; OSM service connector 12966 (service) ~51m; North Dry Creek Cemetery Road (unclassified) ~62m; OSM service connector 229 (service) ~149m
- Decision as runner: Follow Veterans until signed junction with Big Springs; target is Big Springs.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 03: TURN LEFT Big Springs

- Physical role: signed junction with Rabbit Run
- Model frame: The packet says `03 1.02 mi (+0.43) JCT TURN LEFT Big Springs UNTIL signed junction with Rabbit Run.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (Veteran's Trail, Access Trail (#113 Big Springs Loop), OSM path connector 38069, Rabbit Run, REI Connector); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: Veteran's Trail, Access Trail (#113 Big Springs Loop), OSM path connector 38069, Rabbit Run, REI Connector; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Big Springs` until `signed junction with Rabbit Run`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Veteran's Trail (path) ~1m; Veterans ~1m; Big Springs Loop ~43m; Access Trail (#113 Big Springs Loop) ~58m; OSM path connector 38069 (path) ~58m; Rabbit Run (path) ~60m
- Decision as runner: Follow Big Springs until signed junction with Rabbit Run; target is Rabbit Run.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 04: FOLLOW #113 Access Trail ( Big Springs Loop) / Access Trail (#113 Big Springs Loop) / Big Springs Loop / Rabbit Run / Veteran's Trail / Veterans

- Physical role: signed junction with Rabbit Run
- Model frame: The packet says `04 1.45 mi (+0.22) CONNECTOR FOLLOW #113 Access Trail ( Big Springs Loop) / Access Trail (#113 Big Springs Loop) / Big Springs Loop / Rabbit Run / Veteran's Trail / Veterans UNTIL signed junction with Rabbit Run.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (OSM footway connector 66290, OSM path connector 66289); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: Veteran's Trail, OSM footway connector 66290, OSM path connector 66289; vehicle corridor or service/residential road context: OSM service connector 76779, OSM service connector 76777; the branch to privilege is `#113 Access Trail ( Big Springs Loop) / Access Trail (#113 Big Springs Loop) / Big Springs Loop / Rabbit Run / Veteran's Trail / Veterans` until `signed junction with Rabbit Run`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Big Springs Loop ~1m; Veterans ~14m; Veteran's Trail (path) ~94m; Veterans (Dog On-Leash) ~97m; OSM footway connector 66290 (footway) ~130m; OSM service connector 76779 (service) ~136m
- Decision as runner: Follow #113 Access Trail ( Big Springs Loop) / Access Trail (#113 Big Springs Loop) / Big Springs Loop / Rabbit Run / Veteran's Trail / Veterans until signed junction with Rabbit Run; target is Rabbit Run.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 05: CONTINUE STRAIGHT Rabbit Run

- Physical role: signed junction with D's Chaos
- Model frame: The packet says `05 1.67 mi (+1.67) JCT CONTINUE STRAIGHT Rabbit Run UNTIL signed junction with D's Chaos.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (Access Trail (#113 Big Springs Loop), OSM path connector 38069, Veteran's Trail, OSM footway connector 38037, REI Connector); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: Access Trail (#113 Big Springs Loop), OSM path connector 38069, Veteran's Trail, Rabbit Run, OSM footway connector 38037; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Rabbit Run` until `signed junction with D's Chaos`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Big Springs Loop ~1m; Access Trail (#113 Big Springs Loop) ~19m; OSM path connector 38069 (path) ~19m; Veteran's Trail (path) ~20m; Veterans ~20m; Rabbit Run (path) ~29m
- Decision as runner: Follow Rabbit Run until signed junction with D's Chaos; target is D's Chaos.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 06: FOLLOW Concrete Jungle / Concrete Jungle Alt. / OSM path connector 74487 / OSM service connector 12726 / Rabbit Run / Rolling Thunder / #XC9 Treasure View Traverse

- Physical role: signed junction with D's Chaos
- Model frame: The packet says `06 3.34 mi (+0.08) ROAD FOLLOW Concrete Jungle / Concrete Jungle Alt. / OSM path connector 74487 / OSM service connector 12726 / Rabbit Run / Rolling Thunder / #XC9 Treasure View Traverse UNTIL signed junction with D's Chaos.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (REI Connector, OSM path connector 38069, Access Trail (#113 Big Springs Loop), OSM footway connector 38045); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: REI Connector, Rabbit Run, OSM path connector 38069, Access Trail (#113 Big Springs Loop), OSM footway connector 38045; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Concrete Jungle / Concrete Jungle Alt. / OSM path connector 74487 / OSM service connector 12726 / Rabbit Run / Rolling Thunder / #XC9 Treasure View Traverse` until `signed junction with D's Chaos`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: REI Connector (path) ~3m; REI Connection ~4m; Rabbit Run (path) ~38m; Rabbit Run (Dog Off-Leash) ~40m; Big Springs Loop ~51m; OSM path connector 38069 (path) ~54m
- Decision as runner: Follow Concrete Jungle / Concrete Jungle Alt. / OSM path connector 74487 / OSM service connector 12726 / Rabbit Run / Rolling Thunder / #XC9 Treasure View Traverse until signed junction with D's Chaos; target is D's Chaos.
- Wrong-layer risk: generic OSM connector name may not exist on signs; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 07: CONTINUE STRAIGHT D's Chaos

- Physical role: signed junction with REI Connection
- Model frame: The packet says `07 3.42 mi (+1.04) JCT CONTINUE STRAIGHT D's Chaos UNTIL signed junction with REI Connection.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (OSM path connector 111857, OSM path connector 110777); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: OSM path connector 111857, OSM path connector 110777; vehicle corridor or service/residential road context: OSM service connector 41568; the branch to privilege is `D's Chaos` until `signed junction with REI Connection`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: D's Chaos ~4m; OSM path connector 111857 (path) ~23m; Rolly Pollie ~23m; Hell Mary ~45m; OSM path connector 110777 (path) ~47m; OSM service connector 41568 (service) ~47m
- Decision as runner: Follow D's Chaos until signed junction with REI Connection; target is REI Connection.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 08: FOLLOW D's Chaos / REI Connection / #XC12 Stormin Mormon / #XC10 Shake n' Bake / #XC1 Junk Yard / #XC11 Flow Trail

- Physical role: signed junction with REI Connection
- Model frame: The packet says `08 4.46 mi (+0.06) CONNECTOR FOLLOW D's Chaos / REI Connection / #XC12 Stormin Mormon / #XC10 Shake n' Bake / #XC1 Junk Yard / #XC11 Flow Trail UNTIL signed junction with REI Connection.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (OSM path connector 74487, OSM path connector 110735); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: OSM path connector 74487, OSM path connector 110735; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `D's Chaos / REI Connection / #XC12 Stormin Mormon / #XC10 Shake n' Bake / #XC1 Junk Yard / #XC11 Flow Trail` until `signed junction with REI Connection`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Rabbit Run ~3m; D's Chaos ~51m; OSM path connector 74487 (path) ~60m; Rolling Thunder ~67m; Free Ride Connector ~70m; Junk Yard ~70m
- Decision as runner: Follow D's Chaos / REI Connection / #XC12 Stormin Mormon / #XC10 Shake n' Bake / #XC1 Junk Yard / #XC11 Flow Trail until signed junction with REI Connection; target is REI Connection.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 09: TAKE REI Connection

- Physical role: end of REI Connection for this route
- Model frame: The packet says `09 4.52 mi (+0.52) JCT TAKE REI Connection UNTIL end of REI Connection for this route.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (REI Connector, Rabbit Run, OSM path connector 110789); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: REI Connector, Rabbit Run, OSM path connector 110789; vehicle corridor or service/residential road context: OSM track connector 74310; the branch to privilege is `REI Connection` until `end of REI Connection for this route`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: D's Chaos ~4m; REI Connector (path) ~20m; REI Connection ~21m; OSM track connector 74310 (track) ~93m; Rabbit Run (path) ~99m; Rabbit Run (Dog Off-Leash) ~100m
- Decision as runner: Follow REI Connection until end of REI Connection for this route; target is return to car.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 10: FOLLOW Access Trail (#113 Big Springs Loop) / Big Springs Loop / REI Connector / Veteran's Trail / Veterans / Veterans (Dog On-Leash)

- Physical role: parked car / trailhead
- Model frame: The packet says `10 5.04 mi (+1.18) EXIT FOLLOW Access Trail (#113 Big Springs Loop) / Big Springs Loop / REI Connector / Veteran's Trail / Veterans / Veterans (Dog On-Leash) UNTIL parked car / trailhead.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (Rabbit Run, OSM path connector 38069, OSM footway connector 38045); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: REI Connector, Rabbit Run, OSM path connector 38069, Access Trail (#113 Big Springs Loop), OSM footway connector 38045; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Access Trail (#113 Big Springs Loop) / Big Springs Loop / REI Connector / Veteran's Trail / Veterans / Veterans (Dog On-Leash)` until `parked car / trailhead`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: REI Connection ~1m; REI Connector (path) ~1m; Rabbit Run (path) ~42m; Rabbit Run (Dog Off-Leash) ~43m; Big Springs Loop ~55m; OSM path connector 38069 (path) ~59m
- Decision as runner: Follow Access Trail (#113 Big Springs Loop) / Big Springs Loop / REI Connector / Veteran's Trail / Veterans / Veterans (Dog On-Leash) until parked car / trailhead; target is Veterans Trailhead.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Finish / return to car

- Physical role: Route endpoint / return-to-car point.
- Model frame: The packet endpoint closes the loop.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: Veteran's Trail; vehicle corridor or service/residential road context: OSM service connector 12725, OSM service connector 12966, North Dry Creek Cemetery Road, OSM service connector 229; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 12725 (service) ~1m; Veterans (Dog On-Leash) ~8m; Veteran's Trail (path) ~9m; OSM service connector 12966 (service) ~51m; North Dry Creek Cemetery Road (unclassified) ~62m; OSM service connector 229 (service) ~149m
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

- OSM service connector 12725 (~1m, vehicle, highway=service, source=openstreetmap)
- Veterans (~1m, feature, source=ridge_to_rivers_open_data)
- D's Chaos (~4m, feature, source=ridge_to_rivers_open_data)
- Veteran's Trail (~5m, path, highway=path, source=openstreetmap)
- Veterans (Dog On-Leash) (~7m, feature, source=ridge_to_rivers_open_data)
- OSM path connector 111857 (~23m, path, highway=path, source=openstreetmap)
- Rolly Pollie (~23m, feature, source=ridge_to_rivers_open_data)
- Rabbit Run (Dog Off-Leash) (~40m, feature, source=ridge_to_rivers_open_data)
- OSM footway connector 38037 (~41m, path, highway=footway, source=openstreetmap)
- Big Springs Loop (~43m, feature, source=ridge_to_rivers_open_data)
- Hell Mary (~45m, feature, source=ridge_to_rivers_open_data)
- OSM path connector 110777 (~47m, path, highway=path, source=openstreetmap)
- OSM service connector 41568 (~47m, vehicle, highway=service, source=openstreetmap)
- Connector (~48m, feature, source=ridge_to_rivers_open_data)
- Return Ridge (~49m, feature, source=ridge_to_rivers_open_data)
- OSM service connector 12966 (~51m, vehicle, highway=service, source=openstreetmap)
- Access Trail (#113 Big Springs Loop) (~58m, path, source=ridge_to_rivers_open_data)
- OSM path connector 38069 (~58m, path, highway=path, source=openstreetmap)
- OSM path connector 74487 (~60m, path, highway=path, source=openstreetmap)
- Rabbit Run (~60m, path, highway=path, source=openstreetmap)

## Required Next Proof

- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.
- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.
- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.

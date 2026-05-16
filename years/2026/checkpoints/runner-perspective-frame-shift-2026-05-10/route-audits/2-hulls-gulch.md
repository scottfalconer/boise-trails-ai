# Runner-Perspective Frame Shift: 2 - Hulls Gulch

## Frame Contract

- Route card: `2` / outing `2-1`.
- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.
- Evidence used: `docs/field-packet/field-tool-data.json`, `docs/field-packet/gpx/official/2-hulls-gulch-lower-hull-s-gulch-trail-hull-s-gulch-interpretive-crestline-trail-red-cliff.gpx`, R2R open data, OSM connector overlay, official 2026 segment source.
- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.
- Frame decision: `needs-proof`. The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.
- Access status: known-or-mapped parking in packet data.
- Human-validity status for this audit: `needs_visual_proof`.

## Route Snapshot

- Trails: Lower Hull's Gulch Trail, Hull's Gulch Interpretive, Crestline Trail, Red Cliffs, Kestral Trail, Owl's Roost, Chickadee Ridge Trail, Gold Finch, 15th St. Trail.
- Official miles: 13.11; on-foot miles: 17.26.
- Door-to-door: p75 340 min; p90 381 min.
- Segment count: 25; wayfinding cue count: 19.

## Start-End-Junction Frame Shifts

### Start

- Physical role: Park here and start this outing.
- Model frame: The packet proves the route has a start coordinate and a first cue.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: #39A Kestrel, OSM path connector 19870, OSM path connector 83479; vehicle corridor or service/residential road context: OSM service connector 17124, North Sunset Peak Road, OSM service connector 19868; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 17124 (service) ~1m; Kestrel ~15m; Lower Hull's Gulch ~17m; North Sunset Peak Road (unclassified) ~19m; #39A Kestrel (path) ~36m; OSM path connector 19870 (path) ~46m
- Decision as runner: Before moving, find the first signed trail or road-access line and confirm the car is parked where the return cue can actually resolve.
- Wrong-layer risk: start/finish access can fail even when route geometry passes
- Evidence boundary: cue GPX waypoint plus local R2R/OSM overlay; no live imagery or field photo in this audit

### Cue 01: FOLLOW #39A Kestrel

- Physical role: signed junction with Lower Hull's Gulch Trail
- Model frame: The packet says `01 0.00 mi (+0.04) START/ACCESS FOLLOW #39A Kestrel UNTIL signed junction with Lower Hull's Gulch Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (OSM path connector 19870, OSM path connector 83479); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #39A Kestrel, OSM path connector 19870, OSM path connector 83479; vehicle corridor or service/residential road context: OSM service connector 17124, North Sunset Peak Road, OSM service connector 19868; the branch to privilege is `#39A Kestrel` until `signed junction with Lower Hull's Gulch Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 17124 (service) ~3m; Kestrel ~5m; Lower Hull's Gulch ~8m; #39A Kestrel (path) ~25m; North Sunset Peak Road (unclassified) ~27m; OSM path connector 19870 (path) ~37m
- Decision as runner: Follow #39A Kestrel until signed junction with Lower Hull's Gulch Trail; target is Lower Hull's Gulch Trail.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 02: FOLLOW Lower Hull's Gulch Trail

- Physical role: signed junction with Hull's Gulch Interpretive
- Model frame: The packet says `02 0.04 mi (+2.17) FOLLOW Lower Hull's Gulch Trail UNTIL signed junction with Hull's Gulch Interpretive.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#29 Lower Hulls Gulch, OSM path connector 111709); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #29 Lower Hulls Gulch, OSM path connector 111709; vehicle corridor or service/residential road context: North Sunset Peak Road, OSM service connector 19868, OSM service connector 17124; the branch to privilege is `Lower Hull's Gulch Trail` until `signed junction with Hull's Gulch Interpretive`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Lower Hull's Gulch ~2m; North Sunset Peak Road (unclassified) ~11m; OSM service connector 19868 (service) ~12m; #29 Lower Hulls Gulch (path) ~20m; Red Fox (Dog Off-Leash) ~25m; Kestrel ~34m
- Decision as runner: Follow Lower Hull's Gulch Trail until signed junction with Hull's Gulch Interpretive; target is Hull's Gulch Interpretive.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 03: FOLLOW #0 Hull's Gulch Interpretive Trail / #4 8th Street Motorcycle / 8th Street Motorcycle / Connector Trail / Hull's Gulch Interpretive Trail / OSM service connector 12945 / OSM service connector 12946

- Physical role: signed junction with Hull's Gulch Interpretive
- Model frame: The packet says `03 2.21 mi (+0.41) ROAD FOLLOW #0 Hull's Gulch Interpretive Trail / #4 8th Street Motorcycle / 8th Street Motorcycle / Connector Trail / Hull's Gulch Interpretive Trail / OSM service connector 12945 / OSM service connector 12946 UNTIL signed junction with Hull's Gulch Interpretive.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#29 Lower Hulls Gulch, #4 8th Street Motorcycle Trail); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #29 Lower Hulls Gulch, #4 8th Street Motorcycle, #4 8th Street Motorcycle Trail, Connector Trail, Hull's Gulch Interpretive Trail; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#0 Hull's Gulch Interpretive Trail / #4 8th Street Motorcycle / 8th Street Motorcycle / Connector Trail / Hull's Gulch Interpretive Trail / OSM service connector 12945 / OSM service connector 12946` until `signed junction with Hull's Gulch Interpretive`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #29 Lower Hulls Gulch (path) ~0m; Lower Hull's Gulch (STM) ~1m; 8th Street Motorcycle ~18m; #4 8th Street Motorcycle (path) ~46m; #4 8th Street Motorcycle Trail (path) ~48m; Connector Trail (path) ~61m
- Decision as runner: Follow #0 Hull's Gulch Interpretive Trail / #4 8th Street Motorcycle / 8th Street Motorcycle / Connector Trail / Hull's Gulch Interpretive Trail / OSM service connector 12945 / OSM service connector 12946 until signed junction with Hull's Gulch Interpretive; target is Hull's Gulch Interpretive.
- Wrong-layer risk: generic OSM connector name may not exist on signs; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 04: TURN LEFT Hull's Gulch Interpretive

- Physical role: signed junction with Crestline Trail
- Model frame: The packet says `04 2.62 mi (+5.07) JCT TURN LEFT Hull's Gulch Interpretive UNTIL signed junction with Crestline Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#29 Lower Hulls Gulch, #4 8th Street Motorcycle, #4 8th Street Motorcycle Trail, Connector Trail, Hull's Gulch Interpretive Trail); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #29 Lower Hulls Gulch, #4 8th Street Motorcycle, #4 8th Street Motorcycle Trail, Connector Trail, Hull's Gulch Interpretive Trail; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Hull's Gulch Interpretive` until `signed junction with Crestline Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #29 Lower Hulls Gulch (path) ~0m; Lower Hull's Gulch (STM) ~1m; 8th Street Motorcycle ~18m; #4 8th Street Motorcycle (path) ~46m; #4 8th Street Motorcycle Trail (path) ~48m; Connector Trail (path) ~61m
- Decision as runner: Follow Hull's Gulch Interpretive until signed junction with Crestline Trail; target is Crestline Trail.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 05: FOLLOW #28 Crestline / #4 8th Street Motorcycle / 8th Street Motorcycle / Connector Trail / Crestline / Hull's Gulch Interpretive Trail

- Physical role: signed junction with Crestline Trail
- Model frame: The packet says `05 7.69 mi (+1.18) CONNECTOR FOLLOW #28 Crestline / #4 8th Street Motorcycle / 8th Street Motorcycle / Connector Trail / Crestline / Hull's Gulch Interpretive Trail UNTIL signed junction with Crestline Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (Hull's Gulch Interpretive Loop); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: Hull's Gulch Interpretive Loop, Hull's Gulch Interpretive Trail; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#28 Crestline / #4 8th Street Motorcycle / 8th Street Motorcycle / Connector Trail / Crestline / Hull's Gulch Interpretive Trail` until `signed junction with Crestline Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Hull's Gulch Interpretive Loop (path) ~0m; Hull's Gulch Interpretive Trail ~41m
- Decision as runner: Follow #28 Crestline / #4 8th Street Motorcycle / 8th Street Motorcycle / Connector Trail / Crestline / Hull's Gulch Interpretive Trail until signed junction with Crestline Trail; target is Crestline Trail.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 06: TAKE Crestline Trail

- Physical role: signed junction with Red Cliffs
- Model frame: The packet says `06 8.87 mi (+1.83) JCT TAKE Crestline Trail UNTIL signed junction with Red Cliffs.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#29 Lower Hulls Gulch, #28 Crestline, #4 8th Street Motorcycle); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #29 Lower Hulls Gulch, #28 Crestline, #4 8th Street Motorcycle; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Crestline Trail` until `signed junction with Red Cliffs`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: 8th Street Motorcycle ~4m; #29 Lower Hulls Gulch (path) ~43m; Lower Hull's Gulch (STM) ~43m; Crestline ~47m; #28 Crestline (path) ~48m; #4 8th Street Motorcycle (path) ~145m
- Decision as runner: Follow Crestline Trail until signed junction with Red Cliffs; target is Red Cliffs.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 07: FOLLOW Red Cliffs

- Physical role: signed junction with Red Cliffs
- Model frame: The packet says `07 10.70 mi (+0.10) CONNECTOR FOLLOW Red Cliffs UNTIL signed junction with Red Cliffs.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#28 Crestline, #4 8th Street Motorcycle, #29 Lower Hulls Gulch); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #28 Crestline, #4 8th Street Motorcycle, #29 Lower Hulls Gulch; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Red Cliffs` until `signed junction with Red Cliffs`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #28 Crestline (path) ~3m; Crestline ~4m; #4 8th Street Motorcycle (path) ~45m; 8th Street Motorcycle ~47m; Lower Hull's Gulch (STM) ~55m; #29 Lower Hulls Gulch (path) ~56m
- Decision as runner: Follow Red Cliffs until signed junction with Red Cliffs; target is Red Cliffs.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 08: CONTINUE STRAIGHT Red Cliffs

- Physical role: signed junction with Kestral Trail
- Model frame: The packet says `08 10.80 mi (+1.27) JCT CONTINUE STRAIGHT Red Cliffs UNTIL signed junction with Kestral Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#39 Red Cliffs, #28 Crestline); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #39 Red Cliffs, #28 Crestline; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Red Cliffs` until `signed junction with Kestral Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #39 Red Cliffs (path) ~7m; Red Cliffs ~37m; #28 Crestline (path) ~50m; Crestline ~51m
- Decision as runner: Follow Red Cliffs until signed junction with Kestral Trail; target is Kestral Trail.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; wrong-direction choice has meaningful climb penalty
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 09: DOUBLE BACK #29 Lower Hulls Gulch / #39A Kestrel / Kestrel / Lower Hull's Gulch

- Physical role: signed junction with Kestral Trail
- Model frame: The packet says `09 12.07 mi (+0.29) OVERLAP DOUBLE BACK #29 Lower Hulls Gulch / #39A Kestrel / Kestrel / Lower Hull's Gulch UNTIL signed junction with Kestral Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#39 Red Cliffs, #29 Lower Hulls Gulch Trail, OSM path connector 18145, OSM path connector 73086); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #39 Red Cliffs, #29 Lower Hulls Gulch Trail, #29 Lower Hulls Gulch, OSM path connector 18145, OSM path connector 73086; vehicle corridor or service/residential road context: North Sunset Peak Road; the branch to privilege is `#29 Lower Hulls Gulch / #39A Kestrel / Kestrel / Lower Hull's Gulch` until `signed junction with Kestral Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #39 Red Cliffs (path) ~0m; Red Cliffs ~1m; #29 Lower Hulls Gulch Trail (path) ~40m; Lower Hull's Gulch ~40m; #29 Lower Hulls Gulch (path) ~41m; OSM path connector 18145 (path) ~41m
- Decision as runner: Follow #29 Lower Hulls Gulch / #39A Kestrel / Kestrel / Lower Hull's Gulch until signed junction with Kestral Trail; target is Kestral Trail.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 10: TURN RIGHT Kestral Trail

- Physical role: signed junction with Owl's Roost
- Model frame: The packet says `10 12.36 mi (+0.75) JCT TURN RIGHT Kestral Trail UNTIL signed junction with Owl's Roost.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#39A Kestrel, #39 Owls Roost, Foothills Learning Center Interpretive Trail); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #39A Kestrel, #39 Owls Roost, Foothills Learning Center Interpretive Trail; vehicle corridor or service/residential road context: OSM service connector 17126, OSM service connector 17125, OSM service connector 17124; the branch to privilege is `Kestral Trail` until `signed junction with Owl's Roost`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Kestrel ~1m; #39A Kestrel (path) ~2m; OSM service connector 17126 (service) ~22m; #39 Owls Roost (path) ~33m; Owl's Roost ~36m; OSM service connector 17125 (service) ~54m
- Decision as runner: Follow Kestral Trail until signed junction with Owl's Roost; target is Owl's Roost.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 11: FOLLOW #39A Kestrel / Kestrel

- Physical role: signed junction with Owl's Roost
- Model frame: The packet says `11 13.11 mi (+0.61) CONNECTOR FOLLOW #39A Kestrel / Kestrel UNTIL signed junction with Owl's Roost.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#28 Crestline); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #39A Kestrel, #28 Crestline; vehicle corridor or service/residential road context: OSM service connector 11384; the branch to privilege is `#39A Kestrel / Kestrel` until `signed junction with Owl's Roost`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #39A Kestrel (path) ~2m; Kestrel ~38m; #28 Crestline (path) ~102m; Crestline ~103m; OSM service connector 11384 (service) ~135m
- Decision as runner: Follow #39A Kestrel / Kestrel until signed junction with Owl's Roost; target is Owl's Roost.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 12: TAKE Owl's Roost

- Physical role: signed junction with Chickadee Ridge Trail
- Model frame: The packet says `12 13.72 mi (+0.64) JCT TAKE Owl's Roost UNTIL signed junction with Chickadee Ridge Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#39A Kestrel, #28 Crestline); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #39A Kestrel, #28 Crestline; vehicle corridor or service/residential road context: OSM service connector 11384; the branch to privilege is `Owl's Roost` until `signed junction with Chickadee Ridge Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #39A Kestrel (path) ~2m; Kestrel ~38m; #28 Crestline (path) ~102m; Crestline ~103m; OSM service connector 11384 (service) ~135m
- Decision as runner: Follow Owl's Roost until signed junction with Chickadee Ridge Trail; target is Chickadee Ridge Trail.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 13: FOLLOW #36A Chickadee Ridge / #35 Gold Finch / Chickadee Ridge #36A / Connector / Gold Finch / Gold Finch #35 / OSM service connector 19608

- Physical role: signed junction with Chickadee Ridge Trail
- Model frame: The packet says `13 14.36 mi (+0.09) ROAD FOLLOW #36A Chickadee Ridge / #35 Gold Finch / Chickadee Ridge #36A / Connector / Gold Finch / Gold Finch #35 / OSM service connector 19608 UNTIL signed junction with Chickadee Ridge Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#39A Kestrel, #39 Owls Roost, #39 Red Cliffs); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #39A Kestrel, #39 Owls Roost, #39 Red Cliffs; vehicle corridor or service/residential road context: OSM service connector 17126, OSM service connector 17125; the branch to privilege is `#36A Chickadee Ridge / #35 Gold Finch / Chickadee Ridge #36A / Connector / Gold Finch / Gold Finch #35 / OSM service connector 19608` until `signed junction with Chickadee Ridge Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #39A Kestrel (path) ~0m; Kestrel ~0m; Owl's Roost ~45m; #39 Owls Roost (path) ~46m; OSM service connector 17126 (service) ~80m; #39 Red Cliffs (path) ~130m
- Decision as runner: Follow #36A Chickadee Ridge / #35 Gold Finch / Chickadee Ridge #36A / Connector / Gold Finch / Gold Finch #35 / OSM service connector 19608 until signed junction with Chickadee Ridge Trail; target is Chickadee Ridge Trail.
- Wrong-layer risk: generic OSM connector name may not exist on signs
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 14: TAKE Chickadee Ridge Trail

- Physical role: signed junction with Gold Finch
- Model frame: The packet says `14 14.45 mi (+0.60) JCT TAKE Chickadee Ridge Trail UNTIL signed junction with Gold Finch.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#35A Red-Winged Blackbird (Middle), Chickadee Ridge #36A, #35A Red-Winged Blackbird (Connector)); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #35A Red-Winged Blackbird (Middle), Chickadee Ridge #36A, #35A Red-Winged Blackbird (Connector); vehicle corridor or service/residential road context: North Sunset Peak Road; the branch to privilege is `Chickadee Ridge Trail` until `signed junction with Gold Finch`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #35A Red-Winged Blackbird (Middle) (path) ~9m; Red-Winged Blackbird (Middle) ~10m; Connector ~32m; Chickadee Ridge ~35m; Chickadee Ridge #36A (path) ~35m; Red-Winged Blackbird (Connector) ~35m
- Decision as runner: Follow Chickadee Ridge Trail until signed junction with Gold Finch; target is Gold Finch.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 15: FOLLOW #35A Red-Winged Blackbird / #35A Red-Winged Blackbird (Middle) / #36A Chickadee Ridge / #34 Hulls Pond / Chickadee Ridge #36A / Hull's Pond / Hulls Pond #34 / OSM path connector 85288 / Red-Winged Blackbird / Red-Winged Blackbird (Connector) / Red-Winged Blackbird (Middle) / Red-Winged Blackbird Trail #35A

- Physical role: signed junction with Gold Finch
- Model frame: The packet says `15 15.05 mi (+0.23) CONNECTOR FOLLOW #35A Red-Winged Blackbird / #35A Red-Winged Blackbird (Middle) / #36A Chickadee Ridge / #34 Hulls Pond / Chickadee Ridge #36A / Hull's Pond / Hulls Pond #34 / OSM path connector 85288 / Red-Winged Blackbird / Red-Winged Blackbird (Connector) / Red-Winged Blackbird (Middle) / Red-Winged Blackbird Trail #35A UNTIL signed junction with Gold Finch.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (OSM path connector 111709, OSM path connector 111710); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: Chickadee Ridge #36A, OSM path connector 111709, OSM path connector 111710; vehicle corridor or service/residential road context: OSM service connector 19868, North Sunset Peak Road, OSM service connector 17124; the branch to privilege is `#35A Red-Winged Blackbird / #35A Red-Winged Blackbird (Middle) / #36A Chickadee Ridge / #34 Hulls Pond / Chickadee Ridge #36A / Hull's Pond / Hulls Pond #34 / OSM path connector 85288 / Red-Winged Blackbird / Red-Winged Blackbird (Connector) / Red-Winged Blackbird (Middle) / Red-Winged Blackbird Trail #35A` until `signed junction with Gold Finch`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Chickadee Ridge ~0m; Chickadee Ridge #36A (path) ~1m; OSM service connector 19868 (service) ~11m; Red Fox (Dog Off-Leash) ~11m; North Sunset Peak Road (unclassified) ~127m; OSM path connector 111709 (path) ~131m
- Decision as runner: Follow #35A Red-Winged Blackbird / #35A Red-Winged Blackbird (Middle) / #36A Chickadee Ridge / #34 Hulls Pond / Chickadee Ridge #36A / Hull's Pond / Hulls Pond #34 / OSM path connector 85288 / Red-Winged Blackbird / Red-Winged Blackbird (Connector) / Red-Winged Blackbird (Middle) / Red-Winged Blackbird Trail #35A until signed junction with Gold Finch; target is Gold Finch.
- Wrong-layer risk: generic OSM connector name may not exist on signs
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 16: TAKE Gold Finch

- Physical role: signed junction with 15th St. Trail
- Model frame: The packet says `16 15.28 mi (+0.34) JCT TAKE Gold Finch UNTIL signed junction with 15th St. Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (Hulls Pond #34, #35A Red-Winged Blackbird, OSM path connector 83482, OSM path connector 85288, OSM path connector 83481); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: Hulls Pond #34, #35A Red-Winged Blackbird, OSM path connector 83482, OSM path connector 85288, OSM path connector 83481; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Gold Finch` until `signed junction with 15th St. Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Hulls Pond #34 (path) ~5m; Hull's Pond ~6m; #35A Red-Winged Blackbird (path) ~9m; Red-Winged Blackbird ~10m; OSM path connector 83482 (path) ~13m; OSM path connector 85288 (path) ~13m
- Decision as runner: Follow Gold Finch until signed junction with 15th St. Trail; target is 15th St. Trail.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 17: FOLLOW #34 Hulls Pond / Hull's Pond / Hulls Pond #34 / Red Fox (Dog On-Leash)

- Physical role: signed junction with 15th St. Trail
- Model frame: The packet says `17 15.62 mi (+0.12) CONNECTOR FOLLOW #34 Hulls Pond / Hull's Pond / Hulls Pond #34 / Red Fox (Dog On-Leash) UNTIL signed junction with 15th St. Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (Gold Finch #35, OSM footway connector 19606, #39 Owls Roost); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: Gold Finch #35, OSM footway connector 19606, #39 Owls Roost; vehicle corridor or service/residential road context: North Sunset Peak Road, OSM service connector 19608; the branch to privilege is `#34 Hulls Pond / Hull's Pond / Hulls Pond #34 / Red Fox (Dog On-Leash)` until `signed junction with 15th St. Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Gold Finch ~5m; Gold Finch #35 (path) ~6m; North Sunset Peak Road (unclassified) ~21m; Connector ~33m; OSM service connector 19608 (service) ~37m; OSM footway connector 19606 (footway) ~40m
- Decision as runner: Follow #34 Hulls Pond / Hull's Pond / Hulls Pond #34 / Red Fox (Dog On-Leash) until signed junction with 15th St. Trail; target is 15th St. Trail.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 18: TAKE 15th St. Trail

- Physical role: end of 15th St. Trail for this route
- Model frame: The packet says `18 15.74 mi (+0.45) JCT TAKE 15th St. Trail UNTIL end of 15th St. Trail for this route.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (Red Fox #36, 15th Street #41, Camelsback Trails #40); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: Red Fox #36, 15th St. Trail, 15th Street #41, Camelsback Trails #40; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `15th St. Trail` until `end of 15th St. Trail for this route`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Red Fox (Dog On-Leash) ~0m; Red Fox #36 (path) ~12m; Red Fox (Dog Off-Leash) ~19m; 15th St. Trail ~26m; 15th Street #41 (path) ~26m; Camelsback Trails #40 (path) ~55m
- Decision as runner: Follow 15th St. Trail until end of 15th St. Trail for this route; target is return to car.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 19: FOLLOW #29 Lower Hulls Gulch / 15th St. Trail / 15th Street #41 / Chickadee Ridge / Chickadee Ridge #36A / North Sunset Peak Road / OSM service connector 19868 / Red Fox (Dog Off-Leash)

- Physical role: parked car / trailhead
- Model frame: The packet says `19 16.19 mi (+1.05) EXIT FOLLOW #29 Lower Hulls Gulch / 15th St. Trail / 15th Street #41 / Chickadee Ridge / Chickadee Ridge #36A / North Sunset Peak Road / OSM service connector 19868 / Red Fox (Dog Off-Leash) UNTIL parked car / trailhead.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (OSM footway connector 114199, OSM footway connector 51993, OSM footway connector 51990, OSM footway connector 114197); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: 15th St. Trail, 15th Street #41, OSM footway connector 114199, OSM footway connector 51993, OSM footway connector 51990; vehicle corridor or service/residential road context: North 15th Street, OSM service connector 16945; the branch to privilege is `#29 Lower Hulls Gulch / 15th St. Trail / 15th Street #41 / Chickadee Ridge / Chickadee Ridge #36A / North Sunset Peak Road / OSM service connector 19868 / Red Fox (Dog Off-Leash)` until `parked car / trailhead`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: 15th St. Trail ~0m; 15th Street #41 (path) ~1m; OSM footway connector 114199 (footway) ~31m; OSM footway connector 51993 (footway) ~33m; North 15th Street (tertiary) ~37m; OSM service connector 16945 (service) ~38m
- Decision as runner: Follow #29 Lower Hulls Gulch / 15th St. Trail / 15th Street #41 / Chickadee Ridge / Chickadee Ridge #36A / North Sunset Peak Road / OSM service connector 19868 / Red Fox (Dog Off-Leash) until parked car / trailhead; target is Hulls Gulch Trailhead.
- Wrong-layer risk: generic OSM connector name may not exist on signs; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Finish / return to car

- Physical role: Route endpoint / return-to-car point.
- Model frame: The packet endpoint closes the loop.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: #39A Kestrel, OSM path connector 19870, OSM path connector 83479; vehicle corridor or service/residential road context: OSM service connector 17124, North Sunset Peak Road, OSM service connector 19868; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 17124 (service) ~3m; Kestrel ~5m; Lower Hull's Gulch ~8m; #39A Kestrel (path) ~25m; North Sunset Peak Road (unclassified) ~27m; OSM path connector 19870 (path) ~37m
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

- Hull's Gulch Interpretive Loop (~0m, path, highway=path, source=openstreetmap)
- Red Fox (Dog On-Leash) (~0m, feature, source=ridge_to_rivers_open_data)
- Lower Hull's Gulch (STM) (~1m, feature, source=ridge_to_rivers_open_data)
- OSM service connector 17124 (~1m, vehicle, highway=service, source=openstreetmap)
- Gold Finch (~5m, feature, source=ridge_to_rivers_open_data)
- Hulls Pond #34 (~5m, path, highway=path, source=openstreetmap)
- Gold Finch #35 (~6m, path, highway=path, source=openstreetmap)
- Hull's Pond (~6m, feature, source=ridge_to_rivers_open_data)
- #39 Red Cliffs (~7m, path, highway=path, source=openstreetmap)
- #35A Red-Winged Blackbird (~9m, path, highway=path, source=openstreetmap)
- #35A Red-Winged Blackbird (Middle) (~9m, path, highway=path, source=openstreetmap)
- Red-Winged Blackbird (~10m, feature, source=ridge_to_rivers_open_data)
- Red-Winged Blackbird (Middle) (~10m, feature, source=ridge_to_rivers_open_data)
- Red Fox #36 (~12m, path, highway=path, source=openstreetmap)
- OSM path connector 83482 (~13m, path, highway=path, source=openstreetmap)
- OSM path connector 85288 (~13m, path, highway=path, source=openstreetmap)
- Kestrel (~15m, feature, source=ridge_to_rivers_open_data)
- Lower Hull's Gulch (~17m, feature, source=ridge_to_rivers_open_data)
- 8th Street Motorcycle (~18m, feature, source=ridge_to_rivers_open_data)
- North Sunset Peak Road (~19m, vehicle, highway=unclassified, source=openstreetmap)

## Required Next Proof

- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.
- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.
- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.

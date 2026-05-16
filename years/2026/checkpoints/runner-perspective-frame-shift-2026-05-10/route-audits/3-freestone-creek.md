# Runner-Perspective Frame Shift: 3 - Freestone Creek

## Frame Contract

- Route card: `3` / outing `3-1`.
- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.
- Evidence used: `docs/field-packet/field-tool-data.json`, `docs/field-packet/gpx/official/3-freestone-creek-military-reserve-connection-mountain-cove-central-ridge-trail-central-ri.gpx`, R2R open data, OSM connector overlay, official 2026 segment source.
- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.
- Frame decision: `needs-proof`. The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.
- Access status: known-or-mapped parking in packet data.
- Human-validity status for this audit: `needs_visual_proof`.

## Route Snapshot

- Trails: Military Reserve Connection, Mountain Cove, Central Ridge Trail, Central Ridge Spur, Ridge Crest, Cottonwood Creek Trail, Connection (Eagle Ridge), Eagle Ridge Trail, Elephant Rock Loop, Heroes Trail.
- Official miles: 8.31; on-foot miles: 12.13.
- Door-to-door: p75 250 min; p90 280 min.
- Segment count: 28; wayfinding cue count: 20.

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

- Physical role: signed junction with Military Reserve Connection
- Model frame: The packet says `01 0.00 mi (+0.27) START/ACCESS FOLLOW #22C Mountain Cove UNTIL signed junction with Military Reserve Connection.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#22A Access Trail (Central Ridge), Access Trail (Central Ridge), #22B Freestone Creek); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #22C Mountain Cove, #22A Access Trail (Central Ridge), Access Trail (Central Ridge), #22B Freestone Creek; vehicle corridor or service/residential road context: OSM service connector 14900, North Mountain Cove Road; the branch to privilege is `#22C Mountain Cove` until `signed junction with Military Reserve Connection`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #22C Mountain Cove (path) ~4m; Mountain Cove ~5m; OSM service connector 14900 (service) ~6m; #22A Access Trail (Central Ridge) (path) ~8m; Access Trail (Central Ridge) ~8m; North Mountain Cove Road (unclassified) ~26m
- Decision as runner: Follow #22C Mountain Cove until signed junction with Military Reserve Connection; target is Military Reserve Connection.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 02: FOLLOW Military Reserve Connection

- Physical role: signed junction with Mountain Cove
- Model frame: The packet says `02 0.27 mi (+0.63) FOLLOW Military Reserve Connection UNTIL signed junction with Mountain Cove.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#22C Mountain Cove, #23 Military Reserve Connection); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #22C Mountain Cove, #23 Military Reserve Connection; vehicle corridor or service/residential road context: North Mountain Cove Road; the branch to privilege is `Military Reserve Connection` until `signed junction with Mountain Cove`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #22C Mountain Cove (path) ~1m; Mountain Cove ~1m; Military Reserve Connection ~45m; North Mountain Cove Road (unclassified) ~45m; #23 Military Reserve Connection (path) ~50m
- Decision as runner: Follow Military Reserve Connection until signed junction with Mountain Cove; target is Mountain Cove.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 03: DOUBLE BACK #23 Military Reserve Connection / Military Reserve Connection / Mountain Cove

- Physical role: signed junction with Mountain Cove
- Model frame: The packet says `03 0.90 mi (+0.59) OVERLAP DOUBLE BACK #23 Military Reserve Connection / Military Reserve Connection / Mountain Cove UNTIL signed junction with Mountain Cove.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#28 Crestline, OSM path connector 26442, #39A Kestrel); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #23 Military Reserve Connection, #28 Crestline, OSM path connector 26442, #39A Kestrel; vehicle corridor or service/residential road context: North Claremont Drive, OSM service connector 11384; the branch to privilege is `#23 Military Reserve Connection / Military Reserve Connection / Mountain Cove` until `signed junction with Mountain Cove`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #23 Military Reserve Connection (path) ~1m; Military Reserve Connection ~1m; #28 Crestline (path) ~45m; Crestline ~45m; OSM path connector 26442 (path) ~65m; North Claremont Drive (residential) ~86m
- Decision as runner: Follow #23 Military Reserve Connection / Military Reserve Connection / Mountain Cove until signed junction with Mountain Cove; target is Mountain Cove.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 04: BEAR LEFT Mountain Cove

- Physical role: signed junction with Central Ridge Trail
- Model frame: The packet says `04 1.49 mi (+0.96) JCT BEAR LEFT Mountain Cove UNTIL signed junction with Central Ridge Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#23 Military Reserve Connection, #22C Mountain Cove); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #23 Military Reserve Connection, #22C Mountain Cove; vehicle corridor or service/residential road context: North Mountain Cove Road; the branch to privilege is `Mountain Cove` until `signed junction with Central Ridge Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #23 Military Reserve Connection (path) ~1m; North Mountain Cove Road (unclassified) ~6m; Mountain Cove ~28m; #22C Mountain Cove (path) ~30m; Military Reserve Connection ~30m
- Decision as runner: Follow Mountain Cove until signed junction with Central Ridge Trail; target is Central Ridge Trail.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 05: FOLLOW #22C Mountain Cove / #27A Toll Road Trail / Mountain Cove / Toll Road Trail

- Physical role: signed junction with Central Ridge Trail
- Model frame: The packet says `05 2.45 mi (+0.07) CONNECTOR FOLLOW #22C Mountain Cove / #27A Toll Road Trail / Mountain Cove / Toll Road Trail UNTIL signed junction with Central Ridge Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#23 Military Reserve Connection); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #22C Mountain Cove, #23 Military Reserve Connection; vehicle corridor or service/residential road context: North Mountain Cove Road; the branch to privilege is `#22C Mountain Cove / #27A Toll Road Trail / Mountain Cove / Toll Road Trail` until `signed junction with Central Ridge Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Mountain Cove ~0m; #22C Mountain Cove (path) ~2m; North Mountain Cove Road (unclassified) ~25m; #23 Military Reserve Connection (path) ~38m; Military Reserve Connection ~43m
- Decision as runner: Follow #22C Mountain Cove / #27A Toll Road Trail / Mountain Cove / Toll Road Trail until signed junction with Central Ridge Trail; target is Central Ridge Trail.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 06: BEAR RIGHT Central Ridge Trail

- Physical role: signed junction with Central Ridge Spur
- Model frame: The packet says `06 2.52 mi (+1.86) JCT BEAR RIGHT Central Ridge Trail UNTIL signed junction with Central Ridge Spur.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#22C Mountain Cove, #22A Central Ridge Spur (South), #27A Toll Road Trail, Toll Road Trail, OSM path connector 11316); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #22C Mountain Cove, #22A Central Ridge Spur (South), #27A Toll Road Trail, Toll Road Trail, OSM path connector 11316; vehicle corridor or service/residential road context: North Mountain Cove Road; the branch to privilege is `Central Ridge Trail` until `signed junction with Central Ridge Spur`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #22C Mountain Cove (path) ~4m; Mountain Cove ~4m; North Mountain Cove Road (unclassified) ~22m; Central Ridge Spur (South) ~25m; #22A Central Ridge Spur (South) (path) ~27m; #27A Toll Road Trail (path) ~27m
- Decision as runner: Follow Central Ridge Trail until signed junction with Central Ridge Spur; target is Central Ridge Spur.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 07: FOLLOW #20 Ridgecrest / #26A Shane's / #20A Bucktail / Bucktail / Ridge Crest / Two Point

- Physical role: signed junction with Central Ridge Spur
- Model frame: The packet says `07 4.38 mi (+0.63) CONNECTOR FOLLOW #20 Ridgecrest / #26A Shane's / #20A Bucktail / Bucktail / Ridge Crest / Two Point UNTIL signed junction with Central Ridge Spur.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#22A Central Ridge Spur (South), #22 Central Ridge, Access Trail (Central Ridge), #22A Access Trail (Central Ridge), #22B Freestone Creek); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #22A Central Ridge Spur (South), #22 Central Ridge, Access Trail (Central Ridge), #22A Access Trail (Central Ridge), #22B Freestone Creek; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#20 Ridgecrest / #26A Shane's / #20A Bucktail / Bucktail / Ridge Crest / Two Point` until `signed junction with Central Ridge Spur`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #22A Central Ridge Spur (South) (path) ~4m; Central Ridge Spur (South) ~4m; #22 Central Ridge (path) ~36m; Central Ridge ~38m; Access Trail (Central Ridge) ~67m; #22A Access Trail (Central Ridge) (path) ~69m
- Decision as runner: Follow #20 Ridgecrest / #26A Shane's / #20A Bucktail / Bucktail / Ridge Crest / Two Point until signed junction with Central Ridge Spur; target is Central Ridge Spur.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 08: CONTINUE STRAIGHT Central Ridge Spur

- Physical role: signed junction with Ridge Crest
- Model frame: The packet says `08 5.01 mi (+0.35) JCT CONTINUE STRAIGHT Central Ridge Spur UNTIL signed junction with Ridge Crest.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#22 Central Ridge, #22A Central Ridge Spur (North), #22A Central Ridge Spur (South)); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #22 Central Ridge, #22A Central Ridge Spur (North), #22A Central Ridge Spur (South); no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Central Ridge Spur` until `signed junction with Ridge Crest`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #22 Central Ridge (path) ~1m; Central Ridge ~1m; #22A Central Ridge Spur (North) (path) ~37m; Central Ridge Spur (North) ~40m; Central Ridge Spur (South) ~170m; #22A Central Ridge Spur (South) (path) ~171m
- Decision as runner: Follow Central Ridge Spur until signed junction with Ridge Crest; target is Ridge Crest.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 09: FOLLOW #20 Ridgecrest / #22A Central Ridge Spur (North) / Central Ridge Spur (North) / Ridge Crest

- Physical role: signed junction with Ridge Crest
- Model frame: The packet says `09 5.36 mi (+0.30) CONNECTOR FOLLOW #20 Ridgecrest / #22A Central Ridge Spur (North) / Central Ridge Spur (North) / Ridge Crest UNTIL signed junction with Ridge Crest.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#22 Central Ridge); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #22 Central Ridge, #22A Central Ridge Spur (North); no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#20 Ridgecrest / #22A Central Ridge Spur (North) / Central Ridge Spur (North) / Ridge Crest` until `signed junction with Ridge Crest`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #22 Central Ridge (path) ~3m; Central Ridge ~3m; #22A Central Ridge Spur (North) (path) ~27m; Central Ridge Spur (North) ~27m
- Decision as runner: Follow #20 Ridgecrest / #22A Central Ridge Spur (North) / Central Ridge Spur (North) / Ridge Crest until signed junction with Ridge Crest; target is Ridge Crest.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 10: TAKE Ridge Crest

- Physical role: signed junction with Cottonwood Creek Trail
- Model frame: The packet says `10 5.66 mi (+1.11) JCT TAKE Ridge Crest UNTIL signed junction with Cottonwood Creek Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#22 Central Ridge, OSM path connector 26422, OSM path connector 26559, #20 Ridgecrest, #20 Ridge Crest); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #22 Central Ridge, OSM path connector 26422, OSM path connector 26559, #20 Ridgecrest, #20 Ridge Crest; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Ridge Crest` until `signed junction with Cottonwood Creek Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #22 Central Ridge (path) ~1m; Central Ridge ~1m; OSM path connector 26422 (path) ~9m; OSM path connector 26559 (path) ~28m; #20 Ridgecrest (path) ~32m; Ridge Crest ~100m
- Decision as runner: Follow Ridge Crest until signed junction with Cottonwood Creek Trail; target is Cottonwood Creek Trail.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 11: FOLLOW #20 Ridge Crest / #27 Cottonwood Creek / Cottonwood Creek / Ridge Crest / Toll Road Trail

- Physical role: signed junction with Cottonwood Creek Trail
- Model frame: The packet says `11 6.77 mi (+0.41) CONNECTOR FOLLOW #20 Ridge Crest / #27 Cottonwood Creek / Cottonwood Creek / Ridge Crest / Toll Road Trail UNTIL signed junction with Cottonwood Creek Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#44 Two Point, #20 Ridgecrest, #22 Central Ridge); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #44 Two Point, #20 Ridgecrest, #22 Central Ridge; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#20 Ridge Crest / #27 Cottonwood Creek / Cottonwood Creek / Ridge Crest / Toll Road Trail` until `signed junction with Cottonwood Creek Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #44 Two Point (footway) ~0m; Two Point ~1m; Bucktail ~9m; #20 Ridgecrest (path) ~21m; Ridge Crest ~21m; Central Ridge ~172m
- Decision as runner: Follow #20 Ridge Crest / #27 Cottonwood Creek / Cottonwood Creek / Ridge Crest / Toll Road Trail until signed junction with Cottonwood Creek Trail; target is Cottonwood Creek Trail.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 12: TAKE Cottonwood Creek Trail

- Physical role: signed junction with Connection (Eagle Ridge)
- Model frame: The packet says `12 7.18 mi (+0.76) JCT TAKE Cottonwood Creek Trail UNTIL signed junction with Connection (Eagle Ridge).`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#20 Ridge Crest, Toll Road Trail, #27 Cottonwood Creek, #27A Toll Road Trail); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #20 Ridge Crest, Toll Road Trail, #27 Cottonwood Creek, #27A Toll Road Trail; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Cottonwood Creek Trail` until `signed junction with Connection (Eagle Ridge)`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #20 Ridge Crest (path) ~1m; Ridge Crest ~2m; Cottonwood Creek ~25m; Toll Road Trail ~25m; #27 Cottonwood Creek (path) ~33m; #27A Toll Road Trail (path) ~33m
- Decision as runner: Follow Cottonwood Creek Trail until signed junction with Connection (Eagle Ridge); target is Connection (Eagle Ridge).
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 13: FOLLOW #25A Eagle Ridge Loop / Connector / Cottonwood Creek / Eagle Ridge Loop (AWT)

- Physical role: signed junction with Connection (Eagle Ridge)
- Model frame: The packet says `13 7.94 mi (+0.22) CONNECTOR FOLLOW #25A Eagle Ridge Loop / Connector / Cottonwood Creek / Eagle Ridge Loop (AWT) UNTIL signed junction with Connection (Eagle Ridge).`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#20 Ridge Crest, Toll Road Trail, #27 Cottonwood Creek, #27A Toll Road Trail); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #20 Ridge Crest, Toll Road Trail, #27 Cottonwood Creek, #27A Toll Road Trail; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#25A Eagle Ridge Loop / Connector / Cottonwood Creek / Eagle Ridge Loop (AWT)` until `signed junction with Connection (Eagle Ridge)`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #20 Ridge Crest (path) ~5m; Ridge Crest ~6m; Cottonwood Creek ~24m; Toll Road Trail ~24m; #27 Cottonwood Creek (path) ~32m; #27A Toll Road Trail (path) ~32m
- Decision as runner: Follow #25A Eagle Ridge Loop / Connector / Cottonwood Creek / Eagle Ridge Loop (AWT) until signed junction with Connection (Eagle Ridge); target is Connection (Eagle Ridge).
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 14: BEAR LEFT Connection (Eagle Ridge)

- Physical role: signed junction with Eagle Ridge Trail
- Model frame: The packet says `14 8.16 mi (+0.58) JCT BEAR LEFT Connection (Eagle Ridge) UNTIL signed junction with Eagle Ridge Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#25A Eagle Ridge Loop, #25 Eagle Ridge, OSM footway connector 53482, OSM footway connector 53483); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #25A Eagle Ridge Loop, #25 Eagle Ridge, OSM footway connector 53482, OSM footway connector 53483; vehicle corridor or service/residential road context: North Knights Drive; the branch to privilege is `Connection (Eagle Ridge)` until `signed junction with Eagle Ridge Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #25A Eagle Ridge Loop (path) ~2m; Eagle Ridge Loop (AWT) ~2m; #25 Eagle Ridge (path) ~9m; Eagle Ridge (AWT) ~9m; Eagle Ridge ~33m; OSM footway connector 53482 (footway) ~69m
- Decision as runner: Follow Connection (Eagle Ridge) until signed junction with Eagle Ridge Trail; target is Eagle Ridge Trail.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 15: BEAR LEFT Eagle Ridge Trail

- Physical role: signed junction with Elephant Rock Loop
- Model frame: The packet says `15 8.74 mi (+0.72) JCT BEAR LEFT Eagle Ridge Trail UNTIL signed junction with Elephant Rock Loop.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#25A Eagle Ridge Loop, #25 Eagle Ridge, OSM path connector 11316, OSM path connector 14902, OSM steps connector 42714); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #25A Eagle Ridge Loop, #25 Eagle Ridge, OSM path connector 11316, OSM path connector 14902, OSM steps connector 42714; vehicle corridor or service/residential road context: #25 Eagle Ridge; the branch to privilege is `Eagle Ridge Trail` until `signed junction with Elephant Rock Loop`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Eagle Ridge Loop (AWT) ~1m; #25A Eagle Ridge Loop (path) ~7m; Eagle Ridge (AWT) ~28m; #25 Eagle Ridge (service) ~35m; Cottonwood Creek ~94m; OSM path connector 11316 (path) ~104m
- Decision as runner: Follow Eagle Ridge Trail until signed junction with Elephant Rock Loop; target is Elephant Rock Loop.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 16: FOLLOW #22C Mountain Cove / #27 Cottonwood Creek / #27A Toll Road Trail / Cottonwood Creek / Elephant Rock Loop / Mountain Cove / OSM path connector 11316

- Physical role: signed junction with Elephant Rock Loop
- Model frame: The packet says `16 9.46 mi (+0.68) CONNECTOR FOLLOW #22C Mountain Cove / #27 Cottonwood Creek / #27A Toll Road Trail / Cottonwood Creek / Elephant Rock Loop / Mountain Cove / OSM path connector 11316 UNTIL signed junction with Elephant Rock Loop.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#25A Eagle Ridge Loop, #25 Eagle Ridge, OSM path connector 14902); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #25A Eagle Ridge Loop, #25 Eagle Ridge, OSM path connector 14902, #27 Cottonwood Creek; vehicle corridor or service/residential road context: #25 Eagle Ridge, OSM service connector 2482, OSM track connector 101224; the branch to privilege is `#22C Mountain Cove / #27 Cottonwood Creek / #27A Toll Road Trail / Cottonwood Creek / Elephant Rock Loop / Mountain Cove / OSM path connector 11316` until `signed junction with Elephant Rock Loop`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Eagle Ridge Loop (AWT) ~2m; #25A Eagle Ridge Loop (path) ~6m; #25 Eagle Ridge (service) ~14m; Eagle Ridge (AWT) ~19m; OSM path connector 14902 (path) ~101m; OSM service connector 2482 (service) ~106m
- Decision as runner: Follow #22C Mountain Cove / #27 Cottonwood Creek / #27A Toll Road Trail / Cottonwood Creek / Elephant Rock Loop / Mountain Cove / OSM path connector 11316 until signed junction with Elephant Rock Loop; target is Elephant Rock Loop.
- Wrong-layer risk: generic OSM connector name may not exist on signs; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 17: CONTINUE STRAIGHT Elephant Rock Loop

- Physical role: signed junction with Heroes Trail
- Model frame: The packet says `17 10.14 mi (+0.50) JCT CONTINUE STRAIGHT Elephant Rock Loop UNTIL signed junction with Heroes Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#22C Mountain Cove, #22A Central Ridge Spur (South), #22B Freestone Creek); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #22C Mountain Cove, #22A Central Ridge Spur (South), #22B Freestone Creek; vehicle corridor or service/residential road context: North Mountain Cove Road; the branch to privilege is `Elephant Rock Loop` until `signed junction with Heroes Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #22C Mountain Cove (path) ~1m; Mountain Cove ~1m; North Mountain Cove Road (unclassified) ~6m; Elephant Rock Loop ~43m; Central Ridge Spur (South) ~44m; #22A Central Ridge Spur (South) (path) ~46m
- Decision as runner: Follow Elephant Rock Loop until signed junction with Heroes Trail; target is Heroes Trail.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 18: FOLLOW Elephant Rock Loop

- Physical role: signed junction with Heroes Trail
- Model frame: The packet says `18 10.64 mi (+0.08) CONNECTOR FOLLOW Elephant Rock Loop UNTIL signed junction with Heroes Trail.`.
- Runner frame: Runner frame: the immediate job is to keep the current trail until the named junction/landmark, with no extra branch proven by local data at this checkpoint.
- Likely visual field: mapped named route features near you: Elephant Rock Loop; vehicle corridor or service/residential road context: North Mountain Cove Road; the branch to privilege is `Elephant Rock Loop` until `signed junction with Heroes Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Elephant Rock Loop ~20m; North Mountain Cove Road (unclassified) ~174m
- Decision as runner: Follow Elephant Rock Loop until signed junction with Heroes Trail; target is Heroes Trail.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 19: BEAR RIGHT Heroes Trail

- Physical role: end of Heroes Trail for this route
- Model frame: The packet says `19 10.72 mi (+0.85) JCT BEAR RIGHT Heroes Trail UNTIL end of Heroes Trail for this route.`.
- Runner frame: Runner frame: the immediate job is to keep the current trail until the named junction/landmark, with no extra branch proven by local data at this checkpoint.
- Likely visual field: mapped named route features near you: Elephant Rock Loop; vehicle corridor or service/residential road context: North Mountain Cove Road; the branch to privilege is `Heroes Trail` until `end of Heroes Trail for this route`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Elephant Rock Loop ~20m; North Mountain Cove Road (unclassified) ~174m
- Decision as runner: Follow Heroes Trail until end of Heroes Trail for this route; target is return to car.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 20: FOLLOW #22C Mountain Cove / Elephant Rock Loop / Military Reserve Connection / Mountain Cove / OSM service connector 14900

- Physical role: parked car / trailhead
- Model frame: The packet says `20 11.57 mi (+1.33) EXIT FOLLOW #22C Mountain Cove / Elephant Rock Loop / Military Reserve Connection / Mountain Cove / OSM service connector 14900 UNTIL parked car / trailhead.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#8 Heroes Trail, Heroes Trail); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #8 Heroes Trail, Heroes Trail; vehicle corridor or service/residential road context: OSM service connector 17147, OSM service connector 17148, OSM service connector 4697, West Circle Way Drive; the branch to privilege is `#22C Mountain Cove / Elephant Rock Loop / Military Reserve Connection / Mountain Cove / OSM service connector 14900` until `parked car / trailhead`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #8 Heroes Trail (path) ~5m; Heroes Trail ~6m; OSM service connector 17147 (service) ~40m; OSM service connector 17148 (service) ~51m; OSM service connector 4697 (service) ~54m; West Circle Way Drive (residential) ~70m
- Decision as runner: Follow #22C Mountain Cove / Elephant Rock Loop / Military Reserve Connection / Mountain Cove / OSM service connector 14900 until parked car / trailhead; target is Freestone Creek Trailhead.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; generic OSM connector name may not exist on signs
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

- #44 Two Point (~0m, path, highway=footway, source=openstreetmap)
- #25A Eagle Ridge Loop (~2m, path, highway=path, source=openstreetmap)
- Eagle Ridge Loop (AWT) (~2m, feature, source=ridge_to_rivers_open_data)
- OSM service connector 14900 (~4m, vehicle, highway=service, source=openstreetmap)
- #8 Heroes Trail (~5m, path, highway=path, source=openstreetmap)
- Heroes Trail (~6m, path, source=ridge_to_rivers_open_data)
- #25 Eagle Ridge (~9m, path, highway=path, source=openstreetmap)
- Bucktail (~9m, feature, source=ridge_to_rivers_open_data)
- Eagle Ridge (AWT) (~9m, feature, source=ridge_to_rivers_open_data)
- OSM path connector 26422 (~9m, path, highway=path, source=openstreetmap)
- #22A Access Trail (Central Ridge) (~10m, path, highway=path, source=openstreetmap)
- #22C Mountain Cove (~14m, path, highway=path, source=openstreetmap)
- Mountain Cove (~14m, feature, source=ridge_to_rivers_open_data)
- Access Trail (Central Ridge) (~15m, path, source=ridge_to_rivers_open_data)
- North Mountain Cove Road (~17m, vehicle, highway=unclassified, source=openstreetmap)
- Central Ridge Spur (South) (~25m, feature, source=ridge_to_rivers_open_data)
- Cottonwood Creek (~25m, feature, source=ridge_to_rivers_open_data)
- #22A Central Ridge Spur (South) (~27m, path, highway=path, source=openstreetmap)
- #27A Toll Road Trail (~27m, path, highway=path, source=openstreetmap)
- OSM path connector 26559 (~28m, path, highway=path, source=openstreetmap)

## Required Next Proof

- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.
- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.
- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.

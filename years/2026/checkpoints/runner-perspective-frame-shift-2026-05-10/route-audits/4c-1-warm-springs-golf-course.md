# Runner-Perspective Frame Shift: 4C-1 - Warm Springs Golf Course

## Frame Contract

- Route card: `4C-1` / outing `4-1`.
- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.
- Evidence used: `docs/field-packet/field-tool-data.json`, `docs/field-packet/gpx/official/4c-1-warm-springs-golf-course-tram-trail-table-rock-quarry-trail.gpx`, R2R open data, OSM connector overlay, official 2026 segment source.
- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.
- Frame decision: `needs-proof`. The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.
- Access status: parking evidence incomplete in packet data.
- Human-validity status for this audit: `needs_visual_proof`.

## Route Snapshot

- Trails: Tram Trail, Table Rock Quarry Trail.
- Official miles: 1.52; on-foot miles: 3.45.
- Door-to-door: p75 102 min; p90 115 min.
- Segment count: 4; wayfinding cue count: 5.

## Start-End-Junction Frame Shifts

### Start

- Physical role: Park here and start this outing.
- Model frame: The packet proves the route has a start coordinate and a first cue.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Vehicle movement may be audible or visible near OSM service connector 23778, OSM service connector 23784, OSM service connector 23783; do not mistake the road/driveway line for the trail branch.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: OSM footway connector 43960, OSM footway connector 43961; vehicle corridor or service/residential road context: OSM service connector 23778, OSM service connector 23784, OSM service connector 23783, OSM service connector 23779; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 23778 (service) ~1m; OSM service connector 23784 (service) ~3m; OSM service connector 23783 (service) ~17m; OSM service connector 23779 (service) ~23m; East Warm Springs Avenue (tertiary) ~26m; OSM footway connector 43960 (footway) ~34m
- Decision as runner: Before moving, find the first signed trail or road-access line and confirm the car is parked where the return cue can actually resolve.
- Wrong-layer risk: start/finish access can fail even when route geometry passes
- Evidence boundary: cue GPX waypoint plus local R2R/OSM overlay; no live imagery or field photo in this audit

### Cue 01: FOLLOW #584366 Greenbelt - North - Broadway to Warm Springs Golf Course

- Physical role: signed junction with Tram Trail
- Model frame: The packet says `01 0.00 mi (+0.05) START/ACCESS FOLLOW #584366 Greenbelt - North - Broadway to Warm Springs Golf Course UNTIL signed junction with Tram Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (OSM footway connector 43964); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: OSM footway connector 43964; vehicle corridor or service/residential road context: OSM service connector 23778, OSM service connector 23784, OSM service connector 23783, East Warm Springs Avenue; the branch to privilege is `#584366 Greenbelt - North - Broadway to Warm Springs Golf Course` until `signed junction with Tram Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 23778 (service) ~2m; OSM service connector 23784 (service) ~5m; OSM service connector 23783 (service) ~15m; East Warm Springs Avenue (tertiary) ~25m; OSM service connector 23779 (service) ~25m; OSM service connector 23777 (service) ~33m
- Decision as runner: Follow #584366 Greenbelt - North - Broadway to Warm Springs Golf Course until signed junction with Tram Trail; target is Tram Trail.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 02: FOLLOW Tram Trail

- Physical role: signed junction with Table Rock Quarry Trail
- Model frame: The packet says `02 0.05 mi (+0.50) FOLLOW Tram Trail UNTIL signed junction with Table Rock Quarry Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#14 Tram, OSM footway connector 43964); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #14 Tram, OSM footway connector 43964; vehicle corridor or service/residential road context: East Warm Springs Avenue, OSM service connector 23777, OSM service connector 23778, OSM service connector 23781; the branch to privilege is `Tram Trail` until `signed junction with Table Rock Quarry Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #14 Tram (path) ~4m; Tram ~4m; OSM footway connector 43964 (footway) ~5m; East Warm Springs Avenue (tertiary) ~10m; OSM service connector 23777 (service) ~17m; OSM service connector 23778 (service) ~33m
- Decision as runner: Follow Tram Trail until signed junction with Table Rock Quarry Trail; target is Table Rock Quarry Trail.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 03: FOLLOW connector/access

- Physical role: signed junction with Table Rock Quarry Trail
- Model frame: The packet says `03 0.55 mi (+0.99) CONNECTOR FOLLOW connector/access UNTIL signed junction with Table Rock Quarry Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#14 Tram, #16B Rock Island (West), OSM footway connector 43973, OSM path connector 83637, OSM path connector 106703); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #14 Tram, #16B Rock Island (West), OSM footway connector 43973, OSM path connector 83637, OSM path connector 106703; vehicle corridor or service/residential road context: East Windsong Drive; the branch to privilege is `connector/access` until `signed junction with Table Rock Quarry Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #14 Tram (path) ~1m; Tram ~3m; #16B Rock Island (West) (path) ~46m; Rock Island (West) ~47m; OSM footway connector 43973 (footway) ~52m; OSM path connector 83637 (path) ~68m
- Decision as runner: Follow connector/access until signed junction with Table Rock Quarry Trail; target is Table Rock Quarry Trail.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 04: TAKE Table Rock Quarry Trail

- Physical role: end of Table Rock Quarry Trail for this route
- Model frame: The packet says `04 1.54 mi (+1.02) JCT TAKE Table Rock Quarry Trail UNTIL end of Table Rock Quarry Trail for this route.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#15 Table Rock, #17 Table Rock Quarry, #16B Rock Island (East)); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #15 Table Rock, #17 Table Rock Quarry, #16B Rock Island (East); no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Table Rock Quarry Trail` until `end of Table Rock Quarry Trail for this route`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Table Rock ~2m; #15 Table Rock (path) ~3m; #17 Table Rock Quarry (path) ~45m; Table Rock Quarry ~51m; #16B Rock Island (East) (path) ~132m; Rock Island (East) ~132m
- Decision as runner: Follow Table Rock Quarry Trail until end of Table Rock Quarry Trail for this route; target is return to car.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 05: FOLLOW #14 Tram / #16A Rock Garden / #16B Rock Island (Deer Path) / #16B Rock Island (East) / #16B Rock Island (West) / #17 Table Rock Quarry / Connector / Rock Garden / Rock Island (East) / Rock Island (West) / Table Rock Quarry / Tram

- Physical role: parked car / trailhead
- Model frame: The packet says `05 2.56 mi (+1.76) EXIT FOLLOW #14 Tram / #16A Rock Garden / #16B Rock Island (Deer Path) / #16B Rock Island (East) / #16B Rock Island (West) / #17 Table Rock Quarry / Connector / Rock Garden / Rock Island (East) / Rock Island (West) / Table Rock Quarry / Tram UNTIL parked car / trailhead.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (OSM path connector 74683, OSM path connector 74671); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #17 Table Rock Quarry, OSM path connector 74683, OSM path connector 74671; vehicle corridor or service/residential road context: East Table Rock Road, OSM service connector 74679, OSM service connector 10732, OSM service connector 74680; the branch to privilege is `#14 Tram / #16A Rock Garden / #16B Rock Island (Deer Path) / #16B Rock Island (East) / #16B Rock Island (West) / #17 Table Rock Quarry / Connector / Rock Garden / Rock Island (East) / Rock Island (West) / Table Rock Quarry / Tram` until `parked car / trailhead`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Table Rock Quarry ~2m; #17 Table Rock Quarry (path) ~3m; OSM path connector 74683 (path) ~79m; East Table Rock Road (service) ~81m; OSM path connector 74671 (path) ~83m; OSM service connector 74679 (service) ~83m
- Decision as runner: Follow #14 Tram / #16A Rock Garden / #16B Rock Island (Deer Path) / #16B Rock Island (East) / #16B Rock Island (West) / #17 Table Rock Quarry / Connector / Rock Garden / Rock Island (East) / Rock Island (West) / Table Rock Quarry / Tram until parked car / trailhead; target is Warm Springs Golf Course Parking/Trailhead.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Finish / return to car

- Physical role: Route endpoint / return-to-car point.
- Model frame: The packet endpoint closes the loop.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Vehicle movement may be audible or visible near OSM service connector 23778, OSM service connector 23784, OSM service connector 23783; do not mistake the road/driveway line for the trail branch.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: OSM footway connector 43964; vehicle corridor or service/residential road context: OSM service connector 23778, OSM service connector 23784, OSM service connector 23783, East Warm Springs Avenue; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 23778 (service) ~2m; OSM service connector 23784 (service) ~5m; OSM service connector 23783 (service) ~15m; East Warm Springs Avenue (tertiary) ~25m; OSM service connector 23779 (service) ~25m; OSM service connector 23777 (service) ~33m
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

- OSM service connector 23778 (~1m, vehicle, highway=service, source=openstreetmap)
- Table Rock (~2m, feature, source=ridge_to_rivers_open_data)
- #15 Table Rock (~3m, path, highway=path, source=openstreetmap)
- OSM service connector 23784 (~3m, vehicle, highway=service, source=openstreetmap)
- #14 Tram (~4m, path, highway=path, source=openstreetmap)
- Tram (~4m, feature, source=ridge_to_rivers_open_data)
- OSM service connector 23783 (~17m, vehicle, highway=service, source=openstreetmap)
- OSM service connector 23779 (~23m, vehicle, highway=service, source=openstreetmap)
- East Warm Springs Avenue (~26m, vehicle, highway=tertiary, source=openstreetmap)
- OSM service connector 23777 (~33m, vehicle, highway=service, source=openstreetmap)
- OSM service connector 23781 (~33m, vehicle, highway=service, source=openstreetmap)
- OSM service connector 23782 (~33m, vehicle, highway=service, source=openstreetmap)
- OSM footway connector 43960 (~34m, path, highway=footway, source=openstreetmap)
- OSM footway connector 43961 (~34m, path, highway=footway, source=openstreetmap)
- OSM footway connector 43964 (~34m, path, highway=footway, source=openstreetmap)
- OSM service connector 23780 (~34m, vehicle, highway=service, source=openstreetmap)
- #17 Table Rock Quarry (~45m, path, highway=path, source=openstreetmap)
- #16B Rock Island (West) (~46m, path, highway=path, source=openstreetmap)
- Rock Island (West) (~47m, feature, source=ridge_to_rivers_open_data)
- Table Rock Quarry (~51m, feature, source=ridge_to_rivers_open_data)

## Required Next Proof

- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.
- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.
- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.

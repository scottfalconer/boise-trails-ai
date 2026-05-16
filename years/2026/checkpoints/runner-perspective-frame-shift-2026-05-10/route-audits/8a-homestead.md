# Runner-Perspective Frame Shift: 8A - Homestead

## Frame Contract

- Route card: `8A` / outing `8-1`.
- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.
- Evidence used: `docs/field-packet/field-tool-data.json`, `docs/field-packet/gpx/official/8a-homestead-harris-ridge-trail.gpx`, R2R open data, OSM connector overlay, official 2026 segment source.
- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.
- Frame decision: `needs-proof`. The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.
- Access status: known-or-mapped parking in packet data.
- Human-validity status for this audit: `needs_visual_proof`.

## Route Snapshot

- Trails: Harris Ridge Trail.
- Official miles: 1.72; on-foot miles: 4.44.
- Door-to-door: p75 118 min; p90 133 min.
- Segment count: 1; wayfinding cue count: 3.

## Start-End-Junction Frame Shifts

### Start

- Physical role: Park here and start this outing.
- Model frame: The packet proves the route has a start coordinate and a first cue.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: OSM footway connector 113828, OSM footway connector 47524, OSM footway connector 47525, OSM footway connector 47523, OSM footway connector 47526; vehicle corridor or service/residential road context: South Council Spring Road, OSM service connector 21895; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: South Council Spring Road (residential) ~10m; OSM footway connector 113828 (footway) ~17m; OSM footway connector 47524 (footway) ~20m; OSM footway connector 47525 (footway) ~20m; OSM footway connector 47523 (footway) ~21m; OSM service connector 21895 (service) ~22m
- Decision as runner: Before moving, find the first signed trail or road-access line and confirm the car is parked where the return cue can actually resolve.
- Wrong-layer risk: start/finish access can fail even when route geometry passes; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: cue GPX waypoint plus local R2R/OSM overlay; no live imagery or field photo in this audit

### Cue 01: FOLLOW #12 Homestead / South Council Spring Road

- Physical role: signed junction with Harris Ridge Trail
- Model frame: The packet says `01 0.00 mi (+0.38) START/ACCESS FOLLOW #12 Homestead / South Council Spring Road UNTIL signed junction with Harris Ridge Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (OSM footway connector 113828, OSM footway connector 47524, OSM footway connector 47525, OSM footway connector 47523, OSM footway connector 47526); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: OSM footway connector 113828, OSM footway connector 47524, OSM footway connector 47525, OSM footway connector 47523, OSM footway connector 47526; vehicle corridor or service/residential road context: South Council Spring Road, OSM service connector 21895; the branch to privilege is `#12 Homestead / South Council Spring Road` until `signed junction with Harris Ridge Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: South Council Spring Road (residential) ~2m; OSM footway connector 113828 (footway) ~8m; OSM footway connector 47524 (footway) ~12m; OSM footway connector 47525 (footway) ~12m; OSM footway connector 47523 (footway) ~19m; OSM footway connector 47526 (footway) ~23m
- Decision as runner: Follow #12 Homestead / South Council Spring Road until signed junction with Harris Ridge Trail; target is Harris Ridge Trail.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 02: FOLLOW Harris Ridge Trail

- Physical role: end of Harris Ridge Trail for this route
- Model frame: The packet says `02 0.38 mi (+1.72) FOLLOW Harris Ridge Trail UNTIL end of Harris Ridge Trail for this route.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (OSM path connector 42419, #108 Harris Ridge Trail, #12 Homestead, OSM path connector 113149); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: OSM path connector 42419, #108 Harris Ridge Trail, Harris Ridge Trail, #12 Homestead, OSM path connector 113149; vehicle corridor or service/residential road context: OSM service connector 42418, OSM service connector 111324; the branch to privilege is `Harris Ridge Trail` until `end of Harris Ridge Trail for this route`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 42418 (service) ~3m; OSM path connector 42419 (path) ~25m; #108 Harris Ridge Trail (path) ~27m; Harris Ridge Trail ~27m; Homestead ~27m; #12 Homestead (path) ~55m
- Decision as runner: Follow Harris Ridge Trail until end of Harris Ridge Trail for this route; target is return to car.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 03: FOLLOW #108 Harris Ridge Trail / Harris Ridge Trail / OSM service connector 42418

- Physical role: parked car / trailhead
- Model frame: The packet says `03 2.10 mi (+0.72) EXIT FOLLOW #108 Harris Ridge Trail / Harris Ridge Trail / OSM service connector 42418 UNTIL parked car / trailhead.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#109 Peace Valley Overlook Trail, OSM path connector 104148, OSM path connector 104149); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: Harris Ridge Trail, #108 Harris Ridge Trail, #109 Peace Valley Overlook Trail, OSM path connector 104148, OSM path connector 104149; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#108 Harris Ridge Trail / Harris Ridge Trail / OSM service connector 42418` until `parked car / trailhead`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Harris Ridge Trail ~0m; #108 Harris Ridge Trail (path) ~2m; Peace Valley Overlook ~10m; #109 Peace Valley Overlook Trail (path) ~11m; OSM path connector 104148 (path) ~19m; OSM path connector 104149 (path) ~32m
- Decision as runner: Follow #108 Harris Ridge Trail / Harris Ridge Trail / OSM service connector 42418 until parked car / trailhead; target is Homestead Trail Access Point.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; generic OSM connector name may not exist on signs; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Finish / return to car

- Physical role: Route endpoint / return-to-car point.
- Model frame: The packet endpoint closes the loop.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: OSM footway connector 113828, OSM footway connector 47524, OSM footway connector 47525, OSM footway connector 47523, OSM footway connector 47526; vehicle corridor or service/residential road context: South Council Spring Road, OSM service connector 21895; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: South Council Spring Road (residential) ~2m; OSM footway connector 113828 (footway) ~8m; OSM footway connector 47524 (footway) ~12m; OSM footway connector 47525 (footway) ~12m; OSM footway connector 47523 (footway) ~19m; OSM footway connector 47526 (footway) ~23m
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

- OSM service connector 42418 (~3m, vehicle, highway=service, source=openstreetmap)
- Peace Valley Overlook (~10m, feature, source=ridge_to_rivers_open_data)
- South Council Spring Road (~10m, vehicle, highway=residential, source=openstreetmap)
- #109 Peace Valley Overlook Trail (~11m, path, highway=path, source=openstreetmap)
- OSM footway connector 113828 (~17m, path, highway=footway, source=openstreetmap)
- OSM path connector 104148 (~19m, path, highway=path, source=openstreetmap)
- OSM footway connector 47524 (~20m, path, highway=footway, source=openstreetmap)
- OSM footway connector 47525 (~20m, path, highway=footway, source=openstreetmap)
- OSM footway connector 47523 (~21m, path, highway=footway, source=openstreetmap)
- OSM service connector 21895 (~22m, vehicle, highway=service, source=openstreetmap)
- OSM footway connector 47526 (~25m, path, highway=footway, source=openstreetmap)
- OSM path connector 42419 (~25m, path, highway=path, source=openstreetmap)
- OSM footway connector 47527 (~26m, path, highway=footway, source=openstreetmap)
- #108 Harris Ridge Trail (~27m, path, highway=path, source=openstreetmap)
- Harris Ridge Trail (~27m, path, source=ridge_to_rivers_open_data)
- Homestead (~27m, feature, source=ridge_to_rivers_open_data)
- OSM path connector 104149 (~32m, path, highway=path, source=openstreetmap)
- #12 Homestead (~55m, path, highway=path, source=openstreetmap)
- OSM service connector 111324 (~132m, vehicle, highway=service, source=openstreetmap)
- OSM path connector 113149 (~153m, path, highway=path, source=openstreetmap)

## Required Next Proof

- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.
- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.
- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.

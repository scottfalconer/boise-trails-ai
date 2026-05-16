# Runner-Perspective Frame Shift: 7 - Seamans Gulch

## Frame Contract

- Route card: `7` / outing `7-1`.
- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.
- Evidence used: `docs/field-packet/field-tool-data.json`, `docs/field-packet/gpx/official/7-seamans-gulch-seaman-gulch-trail-wild-phlox-trail.gpx`, R2R open data, OSM connector overlay, official 2026 segment source.
- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.
- Frame decision: `needs-proof`. The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.
- Access status: known-or-mapped parking in packet data.
- Human-validity status for this audit: `needs_visual_proof`.

## Route Snapshot

- Trails: Seaman Gulch Trail, Wild Phlox Trail.
- Official miles: 2.25; on-foot miles: 3.77.
- Door-to-door: p75 127 min; p90 143 min.
- Segment count: 6; wayfinding cue count: 5.

## Start-End-Junction Frame Shifts

### Start

- Physical role: Park here and start this outing.
- Model frame: The packet proves the route has a start coordinate and a first cue.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: Seamans Gulch, Wild Phlox Trail, Seamans Gulch Trail; vehicle corridor or service/residential road context: OSM service connector 78374, OSM service connector 78377, North Seaman Gulch Road; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 78374 (service) ~1m; Seamans Gulch (path) ~2m; OSM service connector 78377 (service) ~18m; North Seaman Gulch Road (tertiary) ~30m; Wild Phlox Trail (path) ~44m; Wild Phlox ~47m
- Decision as runner: Before moving, find the first signed trail or road-access line and confirm the car is parked where the return cue can actually resolve.
- Wrong-layer risk: start/finish access can fail even when route geometry passes
- Evidence boundary: cue GPX waypoint plus local R2R/OSM overlay; no live imagery or field photo in this audit

### Cue 01: FOLLOW #110 Seaman Gulch / Seamans Gulch

- Physical role: signed junction with Seaman Gulch Trail
- Model frame: The packet says `01 0.00 mi (+0.02) START/ACCESS FOLLOW #110 Seaman Gulch / Seamans Gulch UNTIL signed junction with Seaman Gulch Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (Wild Phlox Trail, Seamans Gulch Trail); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: Seamans Gulch, Wild Phlox Trail, Seamans Gulch Trail; vehicle corridor or service/residential road context: OSM service connector 78374, OSM service connector 78377, North Seaman Gulch Road; the branch to privilege is `#110 Seaman Gulch / Seamans Gulch` until `signed junction with Seaman Gulch Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Seamans Gulch (path) ~3m; OSM service connector 78374 (service) ~5m; OSM service connector 78377 (service) ~18m; North Seaman Gulch Road (tertiary) ~36m; Wild Phlox Trail (path) ~46m; Wild Phlox ~49m
- Decision as runner: Follow #110 Seaman Gulch / Seamans Gulch until signed junction with Seaman Gulch Trail; target is Seaman Gulch Trail.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 02: FOLLOW Seaman Gulch Trail

- Physical role: signed junction with Wild Phlox Trail
- Model frame: The packet says `02 0.02 mi (+1.49) FOLLOW Seaman Gulch Trail UNTIL signed junction with Wild Phlox Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (Seamans Gulch, Seamans Gulch Trail, Wild Phlox Trail); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: Seamans Gulch, Seamans Gulch Trail, Wild Phlox Trail; vehicle corridor or service/residential road context: OSM service connector 78377, OSM service connector 78374, North Seaman Gulch Road; the branch to privilege is `Seaman Gulch Trail` until `signed junction with Wild Phlox Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Seamans Gulch (path) ~5m; OSM service connector 78377 (service) ~8m; OSM service connector 78374 (service) ~28m; North Seaman Gulch Road (tertiary) ~33m; Seaman Gulch ~45m; Seamans Gulch Trail (path) ~45m
- Decision as runner: Follow Seaman Gulch Trail until signed junction with Wild Phlox Trail; target is Wild Phlox Trail.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 03: FOLLOW Seaman Gulch / Seamans Gulch Trail / Wild Phlox / Wild Phlox Trail

- Physical role: signed junction with Wild Phlox Trail
- Model frame: The packet says `03 1.51 mi (+0.41) CONNECTOR FOLLOW Seaman Gulch / Seamans Gulch Trail / Wild Phlox / Wild Phlox Trail UNTIL signed junction with Wild Phlox Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (Valley View Trail, Access Trail (#110 Seaman Gulch), Seamans Gulch Access Trail); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: Seamans Gulch Trail, Valley View Trail, Access Trail (#110 Seaman Gulch), Seamans Gulch Access Trail; vehicle corridor or service/residential road context: North Seaman Gulch Road, OSM service connector 2498; the branch to privilege is `Seaman Gulch / Seamans Gulch Trail / Wild Phlox / Wild Phlox Trail` until `signed junction with Wild Phlox Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Seaman Gulch ~2m; Seamans Gulch Trail (path) ~2m; Valley View Trail (path) ~37m; Valley View ~38m; Access Trail (#110 Seaman Gulch) ~73m; Seamans Gulch Access Trail (path) ~74m
- Decision as runner: Follow Seaman Gulch / Seamans Gulch Trail / Wild Phlox / Wild Phlox Trail until signed junction with Wild Phlox Trail; target is Wild Phlox Trail.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 04: TURN AROUND Wild Phlox Trail

- Physical role: end of Wild Phlox Trail for this route
- Model frame: The packet says `04 1.92 mi (+0.77) JCT TURN AROUND Wild Phlox Trail UNTIL end of Wild Phlox Trail for this route.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (Seamans Gulch Trail); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: Seamans Gulch Trail, Wild Phlox Trail; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Wild Phlox Trail` until `end of Wild Phlox Trail for this route`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Seaman Gulch ~1m; Seamans Gulch Trail (path) ~1m; Wild Phlox ~46m; Wild Phlox Trail (path) ~47m
- Decision as runner: Follow Wild Phlox Trail until end of Wild Phlox Trail for this route; target is return to car.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 05: FOLLOW Seaman Gulch / Seamans Gulch Trail / Wild Phlox / Wild Phlox Trail

- Physical role: parked car / trailhead
- Model frame: The packet says `05 2.69 mi (+0.36) EXIT FOLLOW Seaman Gulch / Seamans Gulch Trail / Wild Phlox / Wild Phlox Trail UNTIL parked car / trailhead.`.
- Runner frame: Runner frame: the immediate job is to keep the current trail until the named junction/landmark, with no extra branch proven by local data at this checkpoint.
- Likely visual field: mapped trail/path choices near you: Seamans Gulch Trail, Wild Phlox Trail; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Seaman Gulch / Seamans Gulch Trail / Wild Phlox / Wild Phlox Trail` until `parked car / trailhead`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Seaman Gulch ~1m; Seamans Gulch Trail (path) ~1m; Wild Phlox ~46m; Wild Phlox Trail (path) ~47m
- Decision as runner: Follow Seaman Gulch / Seamans Gulch Trail / Wild Phlox / Wild Phlox Trail until parked car / trailhead; target is Seamans Gulch Trailhead.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Finish / return to car

- Physical role: Route endpoint / return-to-car point.
- Model frame: The packet endpoint closes the loop.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: Seamans Gulch, Wild Phlox Trail, Seamans Gulch Trail; vehicle corridor or service/residential road context: OSM service connector 78374, OSM service connector 78377, North Seaman Gulch Road; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Seamans Gulch (path) ~3m; OSM service connector 78374 (service) ~5m; OSM service connector 78377 (service) ~18m; North Seaman Gulch Road (tertiary) ~36m; Wild Phlox Trail (path) ~46m; Wild Phlox ~49m
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

- OSM service connector 78374 (~1m, vehicle, highway=service, source=openstreetmap)
- Seamans Gulch (~2m, path, highway=path, source=openstreetmap)
- OSM service connector 78377 (~18m, vehicle, highway=service, source=openstreetmap)
- North Seaman Gulch Road (~30m, vehicle, highway=tertiary, source=openstreetmap)
- Valley View Trail (~37m, path, highway=path, source=openstreetmap)
- Valley View (~38m, feature, source=ridge_to_rivers_open_data)
- Wild Phlox Trail (~44m, path, highway=path, source=openstreetmap)
- Wild Phlox (~47m, feature, source=ridge_to_rivers_open_data)
- Seaman Gulch (~72m, feature, source=ridge_to_rivers_open_data)
- Access Trail (#110 Seaman Gulch) (~73m, path, source=ridge_to_rivers_open_data)
- Seamans Gulch Trail (~73m, path, highway=path, source=openstreetmap)
- Seamans Gulch Access Trail (~74m, path, highway=path, source=openstreetmap)
- OSM service connector 2498 (~153m, vehicle, highway=service, source=openstreetmap)

## Required Next Proof

- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.
- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.
- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.

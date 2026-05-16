# Runner-Perspective Frame Shift: 16A-2 - Dry Creek / Sweet Connie roadside parking

## Frame Contract

- Route card: `16A-2` / outing `16-2`.
- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.
- Evidence used: `docs/field-packet/field-tool-data.json`, `docs/field-packet/gpx/official/16a-2-dry-creek-sweet-connie-roadside-parking-sheep-camp-trail-shingle-creek-trail.gpx`, R2R open data, OSM connector overlay, official 2026 segment source.
- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.
- Frame decision: `needs-proof`. The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.
- Access status: parking/access proof-sensitive road or probe anchor.
- Human-validity status for this audit: `needs_visual_proof`.

## Route Snapshot

- Trails: Sheep Camp Trail, Shingle Creek Trail.
- Official miles: 5.53; on-foot miles: 14.96.
- Door-to-door: p75 310 min; p90 348 min.
- Segment count: 2; wayfinding cue count: 5.

## Start-End-Junction Frame Shifts

### Start

- Physical role: Park here and start this outing.
- Model frame: The packet proves the route has a start coordinate and a first cue.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: #77 Sweet Connie, #78 Dry Creek; vehicle corridor or service/residential road context: North Bogus Basin Road; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: North Bogus Basin Road (tertiary) ~9m; Sweet Connie ~10m; #77 Sweet Connie (path) ~12m; Dry Creek ~27m; #78 Dry Creek (path) ~31m
- Decision as runner: Before moving, find the first signed trail or road-access line and confirm the car is parked where the return cue can actually resolve.
- Wrong-layer risk: start/finish access can fail even when route geometry passes
- Evidence boundary: cue GPX waypoint plus local R2R/OSM overlay; no live imagery or field photo in this audit

### Cue 01: FOLLOW Shingle Creek Trail

- Physical role: signed Shingle Creek Trail route / first official segment
- Model frame: The packet says `01 0.00 mi (+2.42) START/ACCESS FOLLOW Shingle Creek Trail UNTIL signed Shingle Creek Trail route / first official segment.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#77 Sweet Connie, #78 Dry Creek); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #77 Sweet Connie, #78 Dry Creek; vehicle corridor or service/residential road context: North Bogus Basin Road; the branch to privilege is `Shingle Creek Trail` until `signed Shingle Creek Trail route / first official segment`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Sweet Connie ~1m; #77 Sweet Connie (path) ~4m; North Bogus Basin Road (tertiary) ~15m; Dry Creek ~24m; #78 Dry Creek (path) ~27m
- Decision as runner: Follow Shingle Creek Trail until signed Shingle Creek Trail route / first official segment; target is Shingle Creek Trail.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 02: FOLLOW Shingle Creek Trail

- Physical role: signed junction with Sheep Camp Trail
- Model frame: The packet says `02 2.42 mi (+4.76) FOLLOW Shingle Creek Trail UNTIL signed junction with Sheep Camp Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#78 Dry Creek, #79 Shingle Creek); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #78 Dry Creek, #79 Shingle Creek; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Shingle Creek Trail` until `signed junction with Sheep Camp Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #78 Dry Creek (path) ~1m; #79 Shingle Creek (path) ~38m; Dry Creek ~41m; Shingle Creek ~41m
- Decision as runner: Follow Shingle Creek Trail until signed junction with Sheep Camp Trail; target is Sheep Camp Trail.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 03: DOUBLE BACK #78 Dry Creek / #79 Shingle Creek / #80 Sheep Camp / Dry Creek / Shingle Creek

- Physical role: signed junction with Sheep Camp Trail
- Model frame: The packet says `03 7.18 mi (+4.47) OVERLAP DOUBLE BACK #78 Dry Creek / #79 Shingle Creek / #80 Sheep Camp / Dry Creek / Shingle Creek UNTIL signed junction with Sheep Camp Trail.`.
- Runner frame: Runner frame: the immediate job is to keep the current trail until the named junction/landmark, with no extra branch proven by local data at this checkpoint.
- Likely visual field: mapped trail/path choices near you: #79 Shingle Creek, #78 Dry Creek; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#78 Dry Creek / #79 Shingle Creek / #80 Sheep Camp / Dry Creek / Shingle Creek` until `signed junction with Sheep Camp Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #79 Shingle Creek (path) ~0m; Shingle Creek ~1m; Dry Creek ~24m; #78 Dry Creek (path) ~25m
- Decision as runner: Follow #78 Dry Creek / #79 Shingle Creek / #80 Sheep Camp / Dry Creek / Shingle Creek until signed junction with Sheep Camp Trail; target is Sheep Camp Trail.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 04: TAKE Sheep Camp Trail

- Physical role: end of Sheep Camp Trail for this route
- Model frame: The packet says `04 11.65 mi (+0.77) JCT TAKE Sheep Camp Trail UNTIL end of Sheep Camp Trail for this route.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#78 Dry Creek, #80 Sheep Camp); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #78 Dry Creek, #80 Sheep Camp; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Sheep Camp Trail` until `end of Sheep Camp Trail for this route`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #78 Dry Creek (path) ~3m; Dry Creek ~3m; Sheep Camp ~42m; #80 Sheep Camp (path) ~47m
- Decision as runner: Follow Sheep Camp Trail until end of Sheep Camp Trail for this route; target is return to car.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 05: FOLLOW Sheep Camp Trail

- Physical role: parked car / trailhead
- Model frame: The packet says `05 12.42 mi (+7.76) EXIT FOLLOW Sheep Camp Trail UNTIL parked car / trailhead.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#80 Sheep Camp, #33 Hard Guy); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #80 Sheep Camp, #33 Hard Guy; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Sheep Camp Trail` until `parked car / trailhead`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Sheep Camp ~1m; #80 Sheep Camp (path) ~2m; #33 Hard Guy (path) ~46m; Hard Guy ~46m
- Decision as runner: Follow Sheep Camp Trail until parked car / trailhead; target is Dry Creek / Sweet Connie roadside parking.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Finish / return to car

- Physical role: Route endpoint / return-to-car point.
- Model frame: The packet endpoint closes the loop.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: #77 Sweet Connie, #78 Dry Creek; vehicle corridor or service/residential road context: North Bogus Basin Road; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Sweet Connie ~1m; #77 Sweet Connie (path) ~4m; North Bogus Basin Road (tertiary) ~15m; Dry Creek ~24m; #78 Dry Creek (path) ~27m
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

- North Bogus Basin Road (~9m, vehicle, highway=tertiary, source=openstreetmap)
- Sweet Connie (~10m, feature, source=ridge_to_rivers_open_data)
- #77 Sweet Connie (~12m, path, highway=path, source=openstreetmap)
- Dry Creek (~27m, feature, source=ridge_to_rivers_open_data)
- #78 Dry Creek (~31m, path, highway=path, source=openstreetmap)
- #79 Shingle Creek (~38m, path, highway=path, source=openstreetmap)
- Shingle Creek (~41m, feature, source=ridge_to_rivers_open_data)
- Sheep Camp (~42m, feature, source=ridge_to_rivers_open_data)
- #33 Hard Guy (~46m, path, highway=path, source=openstreetmap)
- Hard Guy (~46m, feature, source=ridge_to_rivers_open_data)
- #80 Sheep Camp (~47m, path, highway=path, source=openstreetmap)

## Required Next Proof

- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.
- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.
- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.

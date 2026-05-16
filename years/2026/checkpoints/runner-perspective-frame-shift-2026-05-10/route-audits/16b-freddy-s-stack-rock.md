# Runner-Perspective Frame Shift: 16B - Freddy's Stack Rock

## Frame Contract

- Route card: `16B` / outing `16-3`.
- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.
- Evidence used: `docs/field-packet/field-tool-data.json`, `docs/field-packet/gpx/official/16b-freddy-s-stack-rock-stack-rock-connector.gpx`, R2R open data, OSM connector overlay, official 2026 segment source.
- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.
- Frame decision: `needs-proof`. The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.
- Access status: known-or-mapped parking in packet data.
- Human-validity status for this audit: `needs_visual_proof`.

## Route Snapshot

- Trails: Stack Rock Connector.
- Official miles: 3.5; on-foot miles: 4.39.
- Door-to-door: p75 131 min; p90 147 min.
- Segment count: 2; wayfinding cue count: 3.

## Start-End-Junction Frame Shifts

### Start

- Physical role: Park here and start this outing.
- Model frame: The packet proves the route has a start coordinate and a first cue.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: Freddys Stack Rock Trail, #125 Freddys Stack Rock Trail; vehicle corridor or service/residential road context: OSM service connector 40457, North Bogus Basin Road, OSM service connector 112257; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 40457 (service) ~1m; North Bogus Basin Road (tertiary) ~52m; OSM service connector 112257 (service) ~58m; Freddys Stack Rock Trail ~59m; #125 Freddys Stack Rock Trail (path) ~64m
- Decision as runner: Before moving, find the first signed trail or road-access line and confirm the car is parked where the return cue can actually resolve.
- Wrong-layer risk: start/finish access can fail even when route geometry passes
- Evidence boundary: cue GPX waypoint plus local R2R/OSM overlay; no live imagery or field photo in this audit

### Cue 01: FOLLOW #125 Freddys Stack Rock Trail

- Physical role: signed junction with Stack Rock Connector
- Model frame: The packet says `01 0.00 mi (+0.07) START/ACCESS FOLLOW #125 Freddys Stack Rock Trail UNTIL signed junction with Stack Rock Connector.`.
- Runner frame: Runner frame: the immediate job is to keep the current trail until the named junction/landmark, with no extra branch proven by local data at this checkpoint.
- Likely visual field: mapped trail/path choices near you: Freddys Stack Rock Trail, #125 Freddys Stack Rock Trail; vehicle corridor or service/residential road context: OSM service connector 40457, North Bogus Basin Road, OSM service connector 112257; the branch to privilege is `#125 Freddys Stack Rock Trail` until `signed junction with Stack Rock Connector`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 40457 (service) ~5m; North Bogus Basin Road (tertiary) ~40m; Freddys Stack Rock Trail ~73m; #125 Freddys Stack Rock Trail (path) ~80m; OSM service connector 112257 (service) ~105m
- Decision as runner: Follow #125 Freddys Stack Rock Trail until signed junction with Stack Rock Connector; target is Stack Rock Connector.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 02: FOLLOW Stack Rock Connector

- Physical role: end of Stack Rock Connector for this route
- Model frame: The packet says `02 0.07 mi (+3.49) FOLLOW Stack Rock Connector UNTIL end of Stack Rock Connector for this route.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#125 Freddys Stack Rock Trail, Freddys Stack Rock Trail); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #125 Freddys Stack Rock Trail, Freddys Stack Rock Trail; vehicle corridor or service/residential road context: OSM service connector 112257, OSM service connector 40457, North Bogus Basin Road; the branch to privilege is `Stack Rock Connector` until `end of Stack Rock Connector for this route`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 112257 (service) ~1m; OSM service connector 40457 (service) ~13m; #125 Freddys Stack Rock Trail (path) ~22m; Freddys Stack Rock Trail ~27m; North Bogus Basin Road (tertiary) ~67m
- Decision as runner: Follow Stack Rock Connector until end of Stack Rock Connector for this route; target is return to car.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 03: FOLLOW #125 Freddys Stack Rock Trail / Freddys Stack Rock Trail

- Physical role: parked car / trailhead
- Model frame: The packet says `03 3.56 mi (+0.90) EXIT FOLLOW #125 Freddys Stack Rock Trail / Freddys Stack Rock Trail UNTIL parked car / trailhead.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#77 Sweet Connie, #120 Eastside, #126 Big-Stack Cutoff); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #77 Sweet Connie, Freddys Stack Rock Trail, #120 Eastside, #125 Freddys Stack Rock Trail, #126 Big-Stack Cutoff; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#125 Freddys Stack Rock Trail / Freddys Stack Rock Trail` until `parked car / trailhead`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #77 Sweet Connie (path) ~36m; Sweet Connie ~37m; Freddys Stack Rock Trail ~64m; #120 Eastside (path) ~68m; #125 Freddys Stack Rock Trail (path) ~68m; Mr. Big ~76m
- Decision as runner: Follow #125 Freddys Stack Rock Trail / Freddys Stack Rock Trail until parked car / trailhead; target is Freddy's Stack Rock Trailhead.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Finish / return to car

- Physical role: Route endpoint / return-to-car point.
- Model frame: The packet endpoint closes the loop.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: Freddys Stack Rock Trail, #125 Freddys Stack Rock Trail; vehicle corridor or service/residential road context: OSM service connector 40457, North Bogus Basin Road, OSM service connector 112257; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 40457 (service) ~5m; North Bogus Basin Road (tertiary) ~40m; Freddys Stack Rock Trail ~73m; #125 Freddys Stack Rock Trail (path) ~80m; OSM service connector 112257 (service) ~105m
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

- OSM service connector 40457 (~1m, vehicle, highway=service, source=openstreetmap)
- #77 Sweet Connie (~36m, path, highway=path, source=openstreetmap)
- Sweet Connie (~37m, feature, source=ridge_to_rivers_open_data)
- North Bogus Basin Road (~52m, vehicle, highway=tertiary, source=openstreetmap)
- OSM service connector 112257 (~58m, vehicle, highway=service, source=openstreetmap)
- Freddys Stack Rock Trail (~59m, path, source=ridge_to_rivers_open_data)
- #125 Freddys Stack Rock Trail (~64m, path, highway=path, source=openstreetmap)
- #120 Eastside (~68m, path, highway=path, source=openstreetmap)
- Mr. Big (~76m, feature, source=ridge_to_rivers_open_data)
- #126 Big-Stack Cutoff (~77m, path, highway=path, source=openstreetmap)
- Eastside (~81m, feature, source=ridge_to_rivers_open_data)

## Required Next Proof

- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.
- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.
- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.

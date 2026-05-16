# Runner-Perspective Frame Shift: 4B - Upper Interpretive

## Frame Contract

- Route card: `4B` / outing `4-4`.
- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.
- Evidence used: `docs/field-packet/field-tool-data.json`, `docs/field-packet/gpx/official/4b-upper-interpretive-scott-s-trail.gpx`, R2R open data, OSM connector overlay, official 2026 segment source.
- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.
- Frame decision: `needs-proof`. The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.
- Access status: known-or-mapped parking in packet data.
- Human-validity status for this audit: `needs_visual_proof`.

## Route Snapshot

- Trails: Scott's Trail.
- Official miles: 1.05; on-foot miles: 2.01.
- Door-to-door: p75 79 min; p90 89 min.
- Segment count: 1; wayfinding cue count: 3.

## Start-End-Junction Frame Shifts

### Start

- Physical role: Park here and start this outing.
- Model frame: The packet proves the route has a start coordinate and a first cue.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: Hull's Gulch Interpretive Trail; vehicle corridor or service/residential road context: East Sunset Peak Road, OSM track connector 112589; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Hull's Gulch Interpretive Trail ~3m; East Sunset Peak Road (unclassified) ~5m; OSM track connector 112589 (track) ~23m
- Decision as runner: Before moving, find the first signed trail or road-access line and confirm the car is parked where the return cue can actually resolve.
- Wrong-layer risk: start/finish access can fail even when route geometry passes
- Evidence boundary: cue GPX waypoint plus local R2R/OSM overlay; no live imagery or field photo in this audit

### Cue 01: FOLLOW #0 Hull's Gulch Interpretive Trail / East Sunset Peak Road

- Physical role: signed junction with Scott's Trail
- Model frame: The packet says `01 0.00 mi (+0.13) START/ACCESS FOLLOW #0 Hull's Gulch Interpretive Trail / East Sunset Peak Road UNTIL signed junction with Scott's Trail.`.
- Runner frame: Runner frame: the immediate job is to keep the current trail until the named junction/landmark, with no extra branch proven by local data at this checkpoint.
- Likely visual field: mapped trail/path choices near you: Hull's Gulch Interpretive Trail; vehicle corridor or service/residential road context: East Sunset Peak Road, OSM track connector 112589; the branch to privilege is `#0 Hull's Gulch Interpretive Trail / East Sunset Peak Road` until `signed junction with Scott's Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: East Sunset Peak Road (unclassified) ~3m; Hull's Gulch Interpretive Trail ~7m; OSM track connector 112589 (track) ~18m
- Decision as runner: Follow #0 Hull's Gulch Interpretive Trail / East Sunset Peak Road until signed junction with Scott's Trail; target is Scott's Trail.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 02: FOLLOW Scott's Trail

- Physical role: end of Scott's Trail for this route
- Model frame: The packet says `02 0.13 mi (+1.05) FOLLOW Scott's Trail UNTIL end of Scott's Trail for this route.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#32 Scott's); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #32 Scott's; vehicle corridor or service/residential road context: East Sunset Peak Road; the branch to privilege is `Scott's Trail` until `end of Scott's Trail for this route`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: East Sunset Peak Road (unclassified) ~3m; #32 Scott's (path) ~43m; Scott's ~47m
- Decision as runner: Follow Scott's Trail until end of Scott's Trail for this route; target is return to car.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; wrong-direction choice has meaningful climb penalty
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 03: FOLLOW #32 Scott's / Scott's

- Physical role: parked car / trailhead
- Model frame: The packet says `03 1.18 mi (+0.62) EXIT FOLLOW #32 Scott's / Scott's UNTIL parked car / trailhead.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#31 Corrals); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #32 Scott's, #31 Corrals; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#32 Scott's / Scott's` until `parked car / trailhead`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #32 Scott's (path) ~1m; Scott's ~1m; #31 Corrals (path) ~40m; Corrals ~41m
- Decision as runner: Follow #32 Scott's / Scott's until parked car / trailhead; target is Upper Interpretive Trailhead.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Finish / return to car

- Physical role: Route endpoint / return-to-car point.
- Model frame: The packet endpoint closes the loop.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: Hull's Gulch Interpretive Trail; vehicle corridor or service/residential road context: East Sunset Peak Road, OSM track connector 112589; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: East Sunset Peak Road (unclassified) ~3m; Hull's Gulch Interpretive Trail ~7m; OSM track connector 112589 (track) ~18m
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

- Hull's Gulch Interpretive Trail (~3m, path, source=ridge_to_rivers_open_data)
- East Sunset Peak Road (~5m, vehicle, highway=unclassified, source=openstreetmap)
- OSM track connector 112589 (~23m, vehicle, highway=track, source=openstreetmap)
- #31 Corrals (~40m, path, highway=path, source=openstreetmap)
- Corrals (~41m, feature, source=ridge_to_rivers_open_data)
- #32 Scott's (~43m, path, highway=path, source=openstreetmap)
- Scott's (~47m, feature, source=ridge_to_rivers_open_data)

## Required Next Proof

- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.
- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.
- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.

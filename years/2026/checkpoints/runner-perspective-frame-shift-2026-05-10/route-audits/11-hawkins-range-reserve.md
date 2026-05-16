# Runner-Perspective Frame Shift: 11 - Hawkins Range Reserve

## Frame Contract

- Route card: `11` / outing `11-1`.
- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.
- Evidence used: `docs/field-packet/field-tool-data.json`, `docs/field-packet/gpx/official/11-hawkins-range-reserve-hawkins.gpx`, R2R open data, OSM connector overlay, official 2026 segment source.
- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.
- Frame decision: `needs-proof`. The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.
- Access status: known-or-mapped parking in packet data.
- Human-validity status for this audit: `needs_visual_proof`.

## Route Snapshot

- Trails: Hawkins.
- Official miles: 5.63; on-foot miles: 5.73.
- Door-to-door: p75 149 min; p90 167 min.
- Segment count: 3; wayfinding cue count: 3.

## Start-End-Junction Frame Shifts

### Start

- Physical role: Park here and start this outing.
- Model frame: The packet proves the route has a start coordinate and a first cue.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: OSM path connector 92028, Hawkins Reserve Loop, Harrow; vehicle corridor or service/residential road context: OSM service connector 37983; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Hawkins ~11m; OSM path connector 92028 (path) ~30m; OSM service connector 37983 (service) ~39m; Hawkins Reserve Loop (path) ~59m; Harrow (path) ~172m
- Decision as runner: Before moving, find the first signed trail or road-access line and confirm the car is parked where the return cue can actually resolve.
- Wrong-layer risk: start/finish access can fail even when route geometry passes
- Evidence boundary: cue GPX waypoint plus local R2R/OSM overlay; no live imagery or field photo in this audit

### Cue 01: FOLLOW Hawkins

- Physical role: signed Hawkins route / first official segment
- Model frame: The packet says `01 0.00 mi (+0.02) OFFICIAL START FOLLOW Hawkins UNTIL signed Hawkins route / first official segment.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (OSM path connector 92028, Hawkins Reserve Loop); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: OSM path connector 92028, Hawkins Reserve Loop; vehicle corridor or service/residential road context: OSM service connector 37983; the branch to privilege is `Hawkins` until `signed Hawkins route / first official segment`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Hawkins ~3m; OSM path connector 92028 (path) ~35m; OSM service connector 37983 (service) ~41m; Hawkins Reserve Loop (path) ~50m; Harrow ~162m
- Decision as runner: Follow Hawkins until signed Hawkins route / first official segment; target is Hawkins.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 02: FOLLOW Hawkins

- Physical role: end of Hawkins for this route
- Model frame: The packet says `02 0.02 mi (+5.63) FOLLOW Hawkins UNTIL end of Hawkins for this route.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (Hawkins Reserve Loop, OSM path connector 92028); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: Hawkins Reserve Loop, OSM path connector 92028; vehicle corridor or service/residential road context: OSM service connector 37983; the branch to privilege is `Hawkins` until `end of Hawkins for this route`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Hawkins ~4m; Hawkins Reserve Loop (path) ~27m; OSM path connector 92028 (path) ~63m; OSM service connector 37983 (service) ~68m; Harrow ~144m
- Decision as runner: Follow Hawkins until end of Hawkins for this route; target is return to car.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 03: FOLLOW #60 Hawkins / #61 Harrow

- Physical role: parked car / trailhead
- Model frame: The packet says `03 5.65 mi (+0.01) RETURN FOLLOW #60 Hawkins / #61 Harrow UNTIL parked car / trailhead.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (Hawkins Reserve Loop); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: Hawkins Reserve Loop; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#60 Hawkins / #61 Harrow` until `parked car / trailhead`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Hawkins ~0m; Hawkins Reserve Loop (path) ~1m; Harrow ~45m
- Decision as runner: Follow #60 Hawkins / #61 Harrow until parked car / trailhead; target is Hawkins Range Reserve Trailhead.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Finish / return to car

- Physical role: Route endpoint / return-to-car point.
- Model frame: The packet endpoint closes the loop.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: OSM path connector 92028, Hawkins Reserve Loop; vehicle corridor or service/residential road context: OSM service connector 37983; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Hawkins ~3m; OSM path connector 92028 (path) ~35m; OSM service connector 37983 (service) ~41m; Hawkins Reserve Loop (path) ~50m; Harrow ~162m
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

- Hawkins (~11m, feature, source=ridge_to_rivers_open_data)
- OSM path connector 92028 (~30m, path, highway=path, source=openstreetmap)
- OSM service connector 37983 (~39m, vehicle, highway=service, source=openstreetmap)
- Hawkins Reserve Loop (~59m, path, highway=path, source=openstreetmap)
- Harrow (~172m, path, highway=path, source=openstreetmap)

## Required Next Proof

- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.
- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.
- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.

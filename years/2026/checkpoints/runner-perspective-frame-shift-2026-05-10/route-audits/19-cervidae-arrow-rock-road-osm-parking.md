# Runner-Perspective Frame Shift: 19 - Cervidae / Arrow Rock Road OSM Parking

## Frame Contract

- Route card: `19` / outing `19-1`.
- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.
- Evidence used: `docs/field-packet/field-tool-data.json`, `docs/field-packet/gpx/official/19-cervidae-arrow-rock-road-osm-parking-cervidae-peak.gpx`, R2R open data, OSM connector overlay, official 2026 segment source.
- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.
- Frame decision: `needs-proof`. The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.
- Access status: parking/access proof-sensitive road or probe anchor.
- Human-validity status for this audit: `needs_visual_proof`.

## Route Snapshot

- Trails: Cervidae Peak.
- Official miles: 2.24; on-foot miles: 4.51.
- Door-to-door: p75 181 min; p90 203 min.
- Segment count: 1; wayfinding cue count: 3.

## Start-End-Junction Frame Shifts

### Start

- Physical role: Park here and start this outing.
- Model frame: The packet proves the route has a start coordinate and a first cue.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: OSM path connector 94802, OSM path connector 31106, OSM path connector 94803, OSM path connector 94801, OSM path connector 94804; vehicle corridor or service/residential road context: Arrow Rock Road, OSM service connector 94800; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Arrow Rock Road (unclassified) ~7m; OSM service connector 94800 (service) ~9m; OSM path connector 94802 (path) ~25m; OSM path connector 31106 (path) ~26m; OSM path connector 94803 (path) ~39m; OSM path connector 94801 (path) ~48m
- Decision as runner: Before moving, find the first signed trail or road-access line and confirm the car is parked where the return cue can actually resolve.
- Wrong-layer risk: start/finish access can fail even when route geometry passes; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: cue GPX waypoint plus local R2R/OSM overlay; no live imagery or field photo in this audit

### Cue 01: FOLLOW Cervidae Peak

- Physical role: signed Cervidae Peak route / first official segment
- Model frame: The packet says `01 0.00 mi (+0.00) OFFICIAL START FOLLOW Cervidae Peak UNTIL signed Cervidae Peak route / first official segment.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (OSM path connector 94802, OSM path connector 31106, OSM path connector 94803, OSM path connector 94801, OSM path connector 94804); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: OSM path connector 94802, OSM path connector 31106, OSM path connector 94803, OSM path connector 94801, OSM path connector 94804; vehicle corridor or service/residential road context: Arrow Rock Road, OSM service connector 94800; the branch to privilege is `Cervidae Peak` until `signed Cervidae Peak route / first official segment`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Arrow Rock Road (unclassified) ~7m; OSM service connector 94800 (service) ~9m; OSM path connector 94802 (path) ~25m; OSM path connector 31106 (path) ~26m; OSM path connector 94803 (path) ~39m; OSM path connector 94801 (path) ~48m
- Decision as runner: Follow Cervidae Peak until signed Cervidae Peak route / first official segment; target is Cervidae Peak.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 02: FOLLOW Cervidae Peak

- Physical role: end of Cervidae Peak for this route
- Model frame: The packet says `02 0.00 mi (+2.24) FOLLOW Cervidae Peak UNTIL end of Cervidae Peak for this route.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (OSM path connector 94802, OSM path connector 31106, OSM path connector 94803, OSM path connector 94801, OSM path connector 94804); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: OSM path connector 94802, OSM path connector 31106, OSM path connector 94803, OSM path connector 94801, OSM path connector 94804; vehicle corridor or service/residential road context: Arrow Rock Road, OSM service connector 94800; the branch to privilege is `Cervidae Peak` until `end of Cervidae Peak for this route`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Arrow Rock Road (unclassified) ~7m; OSM service connector 94800 (service) ~9m; OSM path connector 94802 (path) ~25m; OSM path connector 31106 (path) ~26m; OSM path connector 94803 (path) ~39m; OSM path connector 94801 (path) ~48m
- Decision as runner: Follow Cervidae Peak until end of Cervidae Peak for this route; target is return to car.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 03: FOLLOW Cervidae Peak

- Physical role: parked car / trailhead
- Model frame: The packet says `03 2.24 mi (+2.24) EXIT FOLLOW Cervidae Peak UNTIL parked car / trailhead.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (OSM path connector 31106, OSM path connector 93087, Cervadae West Side Trail); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: OSM path connector 31106, OSM path connector 93087, Cervadae West Side Trail; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Cervidae Peak` until `parked car / trailhead`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM path connector 31106 (path) ~12m; OSM path connector 93087 (path) ~12m; Cervadae West Side Trail (path) ~13m
- Decision as runner: Follow Cervidae Peak until parked car / trailhead; target is Cervidae / Arrow Rock Road OSM Parking.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Finish / return to car

- Physical role: Route endpoint / return-to-car point.
- Model frame: The packet endpoint closes the loop.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: OSM path connector 94802, OSM path connector 31106, OSM path connector 94803, OSM path connector 94801, OSM path connector 94804; vehicle corridor or service/residential road context: Arrow Rock Road, OSM service connector 94800; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Arrow Rock Road (unclassified) ~7m; OSM service connector 94800 (service) ~9m; OSM path connector 94802 (path) ~25m; OSM path connector 31106 (path) ~26m; OSM path connector 94803 (path) ~39m; OSM path connector 94801 (path) ~48m
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

- Arrow Rock Road (~7m, vehicle, highway=unclassified, source=openstreetmap)
- OSM service connector 94800 (~9m, vehicle, highway=service, source=openstreetmap)
- OSM path connector 93087 (~12m, path, highway=path, source=openstreetmap)
- Cervadae West Side Trail (~13m, path, highway=path, source=openstreetmap)
- OSM path connector 94802 (~25m, path, highway=path, source=openstreetmap)
- OSM path connector 31106 (~26m, path, highway=path, source=openstreetmap)
- OSM path connector 94803 (~39m, path, highway=path, source=openstreetmap)
- OSM path connector 94801 (~48m, path, highway=path, source=openstreetmap)
- OSM path connector 94804 (~56m, path, highway=path, source=openstreetmap)
- OSM path connector 94805 (~88m, path, highway=path, source=openstreetmap)

## Required Next Proof

- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.
- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.
- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.

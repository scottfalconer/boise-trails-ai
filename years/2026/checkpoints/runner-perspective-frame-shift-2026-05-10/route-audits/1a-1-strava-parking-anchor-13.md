# Runner-Perspective Frame Shift: 1A-1 - Strava parking anchor 13

## Frame Contract

- Route card: `1A-1` / outing `1-1`.
- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.
- Evidence used: `docs/field-packet/field-tool-data.json`, `docs/field-packet/gpx/official/1a-1-strava-parking-anchor-13-36th-street-chute.gpx`, R2R open data, OSM connector overlay, official 2026 segment source.
- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.
- Frame decision: `needs-proof`. The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.
- Access status: private-history parking anchor; usable as planning evidence but still public-proof limited.
- Human-validity status for this audit: `needs_visual_proof`.

## Route Snapshot

- Trails: 36th Street Chute.
- Official miles: 0.74; on-foot miles: 1.5.
- Door-to-door: p75 60 min; p90 68 min.
- Segment count: 1; wayfinding cue count: 3.

## Start-End-Junction Frame Shifts

### Start

- Physical role: Park here and start this outing.
- Model frame: The packet proves the route has a start coordinate and a first cue.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars or road-edge ambiguity are plausible because the start is a road or anchor-style access point; treat exact parking legality as a separate proof.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: OSM path connector 13997, OSM path connector 83300, OSM path connector 83301, OSM path connector 17103; vehicle corridor or service/residential road context: North 36th Street; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM path connector 13997 (path) ~8m; OSM path connector 83300 (path) ~13m; OSM path connector 83301 (path) ~17m; North 36th Street (tertiary) ~20m; OSM path connector 17103 (path) ~103m
- Decision as runner: Before moving, find the first signed trail or road-access line and confirm the car is parked where the return cue can actually resolve.
- Wrong-layer risk: start/finish access can fail even when route geometry passes; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: cue GPX waypoint plus local R2R/OSM overlay; no live imagery or field photo in this audit

### Cue 01: FOLLOW 36th Street Chute

- Physical role: signed 36th Street Chute route / first official segment
- Model frame: The packet says `01 0.00 mi (+0.02) OFFICIAL START FOLLOW 36th Street Chute UNTIL signed 36th Street Chute route / first official segment.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (OSM path connector 83300, OSM path connector 13997, OSM path connector 83301, OSM path connector 17103); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: OSM path connector 83300, OSM path connector 13997, OSM path connector 83301, OSM path connector 17103; vehicle corridor or service/residential road context: North 36th Street; the branch to privilege is `36th Street Chute` until `signed 36th Street Chute route / first official segment`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM path connector 83300 (path) ~3m; OSM path connector 13997 (path) ~4m; OSM path connector 83301 (path) ~6m; North 36th Street (tertiary) ~35m; OSM path connector 17103 (path) ~113m
- Decision as runner: Follow 36th Street Chute until signed 36th Street Chute route / first official segment; target is 36th Street Chute.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 02: FOLLOW 36th Street Chute

- Physical role: end of 36th Street Chute for this route
- Model frame: The packet says `02 0.02 mi (+0.74) FOLLOW 36th Street Chute UNTIL end of 36th Street Chute for this route.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (OSM path connector 13997, OSM path connector 83300, OSM path connector 83301, OSM path connector 17103); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: OSM path connector 13997, OSM path connector 83300, OSM path connector 83301, OSM path connector 17103; vehicle corridor or service/residential road context: North 36th Street; the branch to privilege is `36th Street Chute` until `end of 36th Street Chute for this route`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM path connector 13997 (path) ~0m; OSM path connector 83300 (path) ~24m; OSM path connector 83301 (path) ~24m; North 36th Street (tertiary) ~55m; OSM path connector 17103 (path) ~131m
- Decision as runner: Follow 36th Street Chute until end of 36th Street Chute for this route; target is return to car.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 03: FOLLOW OSM path connector 13997

- Physical role: parked car / trailhead
- Model frame: The packet says `03 0.76 mi (+0.72) EXIT FOLLOW OSM path connector 13997 UNTIL parked car / trailhead.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (OSM path connector 13996, OSM path connector 83301, OSM path connector 83300, OSM path connector 83299, OSM path connector 83298); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: OSM path connector 13997, OSM path connector 13996, OSM path connector 83301, OSM path connector 83300, OSM path connector 83299; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `OSM path connector 13997` until `parked car / trailhead`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM path connector 13997 (path) ~1m; OSM path connector 13996 (path) ~19m; OSM path connector 83301 (path) ~36m; OSM path connector 83300 (path) ~88m; OSM path connector 83299 (path) ~115m; OSM path connector 83298 (path) ~119m
- Decision as runner: Follow OSM path connector 13997 until parked car / trailhead; target is Strava parking anchor 13.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; generic OSM connector name may not exist on signs; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Finish / return to car

- Physical role: Route endpoint / return-to-car point.
- Model frame: The packet endpoint closes the loop.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars or road-edge ambiguity are plausible because the start is a road or anchor-style access point; treat exact parking legality as a separate proof.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: OSM path connector 83300, OSM path connector 13997, OSM path connector 83301, OSM path connector 17103; vehicle corridor or service/residential road context: North 36th Street; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM path connector 83300 (path) ~3m; OSM path connector 13997 (path) ~4m; OSM path connector 83301 (path) ~6m; North 36th Street (tertiary) ~35m; OSM path connector 17103 (path) ~113m
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

- OSM path connector 13997 (~8m, path, highway=path, source=openstreetmap)
- OSM path connector 83300 (~13m, path, highway=path, source=openstreetmap)
- OSM path connector 83301 (~17m, path, highway=path, source=openstreetmap)
- OSM path connector 13996 (~19m, path, highway=path, source=openstreetmap)
- North 36th Street (~20m, vehicle, highway=tertiary, source=openstreetmap)
- OSM path connector 17103 (~103m, path, highway=path, source=openstreetmap)
- OSM path connector 83299 (~115m, path, highway=path, source=openstreetmap)
- OSM path connector 83298 (~119m, path, highway=path, source=openstreetmap)

## Required Next Proof

- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.
- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.
- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.

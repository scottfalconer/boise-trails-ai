# Runner-Perspective Frame Shift: 5A - West Hidden Springs Drive road-parking anchor

## Frame Contract

- Route card: `5A` / outing `5-1`.
- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.
- Evidence used: `docs/field-packet/field-tool-data.json`, `docs/field-packet/gpx/official/5a-west-hidden-springs-drive-road-parking-anchor-barn-owl.gpx`, R2R open data, OSM connector overlay, official 2026 segment source.
- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.
- Frame decision: `needs-proof`. The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.
- Access status: parking/access proof-sensitive road or probe anchor.
- Human-validity status for this audit: `needs_visual_proof`.

## Route Snapshot

- Trails: Barn Owl.
- Official miles: 1.44; on-foot miles: 2.52.
- Door-to-door: p75 100 min; p90 112 min.
- Segment count: 2; wayfinding cue count: 3.

## Start-End-Junction Frame Shifts

### Start

- Physical role: Park here and start this outing.
- Model frame: The packet proves the route has a start coordinate and a first cue.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars or road-edge ambiguity are plausible because the start is a road or anchor-style access point; treat exact parking legality as a separate proof.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: OSM footway connector 35178, OSM footway connector 35278, OSM footway connector 90996, OSM footway connector 91001, OSM footway connector 90995; vehicle corridor or service/residential road context: West Hidden Springs Drive, North 17th Avenue; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: West Hidden Springs Drive (tertiary) ~0m; OSM footway connector 35178 (footway) ~8m; OSM footway connector 35278 (footway) ~10m; OSM footway connector 90996 (footway) ~17m; OSM footway connector 91001 (footway) ~17m; OSM footway connector 90995 (footway) ~21m
- Decision as runner: Before moving, find the first signed trail or road-access line and confirm the car is parked where the return cue can actually resolve.
- Wrong-layer risk: start/finish access can fail even when route geometry passes; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: cue GPX waypoint plus local R2R/OSM overlay; no live imagery or field photo in this audit

### Cue 01: FOLLOW Barn Owl / West Hidden Springs Drive

- Physical role: signed Barn Owl route / first official segment
- Model frame: The packet says `01 0.00 mi (+0.54) START/ACCESS FOLLOW Barn Owl / West Hidden Springs Drive UNTIL signed Barn Owl route / first official segment.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (OSM footway connector 35178, OSM footway connector 35278, OSM footway connector 90996, OSM footway connector 91001, OSM footway connector 90995); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: OSM footway connector 35178, OSM footway connector 35278, OSM footway connector 90996, OSM footway connector 91001, OSM footway connector 90995; vehicle corridor or service/residential road context: West Hidden Springs Drive, North 17th Avenue; the branch to privilege is `Barn Owl / West Hidden Springs Drive` until `signed Barn Owl route / first official segment`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: West Hidden Springs Drive (tertiary) ~3m; OSM footway connector 35178 (footway) ~5m; OSM footway connector 35278 (footway) ~14m; OSM footway connector 90996 (footway) ~18m; OSM footway connector 91001 (footway) ~18m; OSM footway connector 90995 (footway) ~24m
- Decision as runner: Follow Barn Owl / West Hidden Springs Drive until signed Barn Owl route / first official segment; target is Barn Owl.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 02: FOLLOW Barn Owl

- Physical role: end of Barn Owl for this route
- Model frame: The packet says `02 0.54 mi (+1.43) FOLLOW Barn Owl UNTIL end of Barn Owl for this route.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#85 Barn Owl, Barn Owl Connector, OSM footway connector 35278, OSM footway connector 95383, OSM footway connector 95384); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #85 Barn Owl, Barn Owl Connector, OSM footway connector 35278, OSM footway connector 95383, OSM footway connector 95384; vehicle corridor or service/residential road context: North 17th Way; the branch to privilege is `Barn Owl` until `end of Barn Owl for this route`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #85 Barn Owl (path) ~4m; Barn Owl ~5m; Barn Owl Connector (path) ~53m; OSM footway connector 35278 (footway) ~134m; North 17th Way (residential) ~139m; OSM footway connector 95383 (footway) ~144m
- Decision as runner: Follow Barn Owl until end of Barn Owl for this route; target is return to car.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch; wrong-direction choice has meaningful climb penalty
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 03: FOLLOW Barn Owl

- Physical role: parked car / trailhead
- Model frame: The packet says `03 1.97 mi (+0.64) EXIT FOLLOW Barn Owl UNTIL parked car / trailhead.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#85 Barn Owl, Barn Owl Connector, OSM footway connector 35278, OSM footway connector 95384); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #85 Barn Owl, Barn Owl Connector, OSM footway connector 35278, OSM footway connector 95384; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `Barn Owl` until `parked car / trailhead`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Barn Owl ~2m; #85 Barn Owl (path) ~3m; Barn Owl Connector (path) ~31m; OSM footway connector 35278 (footway) ~172m; OSM footway connector 95384 (footway) ~174m
- Decision as runner: Follow Barn Owl until parked car / trailhead; target is West Hidden Springs Drive road-parking anchor.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Finish / return to car

- Physical role: Route endpoint / return-to-car point.
- Model frame: The packet endpoint closes the loop.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars or road-edge ambiguity are plausible because the start is a road or anchor-style access point; treat exact parking legality as a separate proof.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: OSM footway connector 35178, OSM footway connector 35278, OSM footway connector 90996, OSM footway connector 91001, OSM footway connector 90995; vehicle corridor or service/residential road context: West Hidden Springs Drive, North 17th Avenue; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: West Hidden Springs Drive (tertiary) ~3m; OSM footway connector 35178 (footway) ~5m; OSM footway connector 35278 (footway) ~14m; OSM footway connector 90996 (footway) ~18m; OSM footway connector 91001 (footway) ~18m; OSM footway connector 90995 (footway) ~24m
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

- West Hidden Springs Drive (~0m, vehicle, highway=tertiary, source=openstreetmap)
- #85 Barn Owl (~4m, path, highway=path, source=openstreetmap)
- Barn Owl (~5m, feature, source=ridge_to_rivers_open_data)
- OSM footway connector 35178 (~8m, path, highway=footway, source=openstreetmap)
- OSM footway connector 35278 (~10m, path, highway=footway, source=openstreetmap)
- OSM footway connector 90996 (~17m, path, highway=footway, source=openstreetmap)
- OSM footway connector 91001 (~17m, path, highway=footway, source=openstreetmap)
- OSM footway connector 90995 (~21m, path, highway=footway, source=openstreetmap)
- OSM footway connector 90998 (~23m, path, highway=footway, source=openstreetmap)
- North 17th Avenue (~26m, vehicle, highway=residential, source=openstreetmap)
- Barn Owl Connector (~53m, path, highway=path, source=openstreetmap)
- North 17th Way (~139m, vehicle, highway=residential, source=openstreetmap)
- OSM footway connector 95383 (~144m, path, highway=footway, source=openstreetmap)
- OSM footway connector 95384 (~164m, path, highway=footway, source=openstreetmap)

## Required Next Proof

- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.
- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.
- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.

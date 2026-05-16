# Runner-Perspective Frame Shift: 15A-1 - Dry Creek / Sweet Connie roadside parking

## Frame Contract

- Route card: `15A-1` / outing `15-1`.
- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.
- Evidence used: `docs/field-packet/field-tool-data.json`, `docs/field-packet/gpx/official/15a-1-dry-creek-sweet-connie-roadside-parking-dry-creek-trail.gpx`, R2R open data, OSM connector overlay, official 2026 segment source.
- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.
- Frame decision: `needs-proof`. The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.
- Access status: parking/access proof-sensitive road or probe anchor.
- Human-validity status for this audit: `needs_visual_proof`.

## Route Snapshot

- Trails: Dry Creek Trail.
- Official miles: 6.97; on-foot miles: 11.89.
- Door-to-door: p75 229 min; p90 257 min.
- Segment count: 5; wayfinding cue count: 3.

## Start-End-Junction Frame Shifts

### Start

- Physical role: Park here and start this outing.
- Model frame: The packet proves the route has a start coordinate and a first cue.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars or road-edge ambiguity are plausible because the start is a road or anchor-style access point; treat exact parking legality as a separate proof.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: #77 Sweet Connie, #78 Dry Creek; vehicle corridor or service/residential road context: North Bogus Basin Road; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: North Bogus Basin Road (tertiary) ~9m; Sweet Connie ~10m; #77 Sweet Connie (path) ~12m; Dry Creek ~27m; #78 Dry Creek (path) ~31m
- Decision as runner: Before moving, find the first signed trail or road-access line and confirm the car is parked where the return cue can actually resolve.
- Wrong-layer risk: start/finish access can fail even when route geometry passes
- Evidence boundary: cue GPX waypoint plus local R2R/OSM overlay; no live imagery or field photo in this audit

### Cue 01: FOLLOW Dry Creek Trail

- Physical role: signed Dry Creek Trail route / first official segment
- Model frame: The packet says `01 0.00 mi (+0.39) START/ACCESS FOLLOW Dry Creek Trail UNTIL signed Dry Creek Trail route / first official segment.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#77 Sweet Connie, #78 Dry Creek); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #77 Sweet Connie, #78 Dry Creek; vehicle corridor or service/residential road context: North Bogus Basin Road; the branch to privilege is `Dry Creek Trail` until `signed Dry Creek Trail route / first official segment`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Sweet Connie ~1m; #77 Sweet Connie (path) ~4m; North Bogus Basin Road (tertiary) ~15m; Dry Creek ~24m; #78 Dry Creek (path) ~27m
- Decision as runner: Follow Dry Creek Trail until signed Dry Creek Trail route / first official segment; target is Dry Creek Trail.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 02: FOLLOW Dry Creek Trail

- Physical role: end of Dry Creek Trail for this route
- Model frame: The packet says `02 0.39 mi (+6.98) FOLLOW Dry Creek Trail UNTIL end of Dry Creek Trail for this route.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#78 Dry Creek, OSM path connector 106708); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #78 Dry Creek, OSM path connector 106708; vehicle corridor or service/residential road context: North Bogus Basin Road; the branch to privilege is `Dry Creek Trail` until `end of Dry Creek Trail for this route`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Dry Creek ~3m; #78 Dry Creek (path) ~23m; OSM path connector 106708 (path) ~30m; North Bogus Basin Road (tertiary) ~142m
- Decision as runner: Follow Dry Creek Trail until end of Dry Creek Trail for this route; target is return to car.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; wrong-direction choice has meaningful climb penalty
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 03: FOLLOW #78 Dry Creek / Dry Creek

- Physical role: parked car / trailhead
- Model frame: The packet says `03 7.37 mi (+5.61) EXIT FOLLOW #78 Dry Creek / Dry Creek UNTIL parked car / trailhead.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#79 Shingle Creek); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #79 Shingle Creek, #78 Dry Creek; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#78 Dry Creek / Dry Creek` until `parked car / trailhead`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Dry Creek ~1m; Shingle Creek ~34m; #79 Shingle Creek (path) ~35m; #78 Dry Creek (path) ~51m
- Decision as runner: Follow #78 Dry Creek / Dry Creek until parked car / trailhead; target is Dry Creek / Sweet Connie roadside parking.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Finish / return to car

- Physical role: Route endpoint / return-to-car point.
- Model frame: The packet endpoint closes the loop.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars or road-edge ambiguity are plausible because the start is a road or anchor-style access point; treat exact parking legality as a separate proof.
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
- OSM path connector 106708 (~30m, path, highway=path, source=openstreetmap)
- #78 Dry Creek (~31m, path, highway=path, source=openstreetmap)
- Shingle Creek (~34m, feature, source=ridge_to_rivers_open_data)
- #79 Shingle Creek (~35m, path, highway=path, source=openstreetmap)

## Required Next Proof

- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.
- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.
- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.

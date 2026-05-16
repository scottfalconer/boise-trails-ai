# Runner-Perspective Frame Shift: 17 - Simplot Lodge Parking Area

## Frame Contract

- Route card: `17` / outing `17-1`.
- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.
- Evidence used: `docs/field-packet/field-tool-data.json`, `docs/field-packet/gpx/official/17-simplot-lodge-parking-area-sunshine-xc-deer-point-trail-around-the-mountain-trail-the-f.gpx`, R2R open data, OSM connector overlay, official 2026 segment source.
- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.
- Frame decision: `needs-proof`. The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.
- Access status: known-or-mapped parking in packet data.
- Human-validity status for this audit: `needs_visual_proof`.

## Route Snapshot

- Trails: Sunshine XC, Deer Point Trail, Around the Mountain Trail, The Face Trail, Elk Meadows Trail.
- Official miles: 11.29; on-foot miles: 15.13.
- Door-to-door: p75 388 min; p90 435 min.
- Segment count: 12; wayfinding cue count: 11.

## Start-End-Junction Frame Shifts

### Start

- Physical role: Park here and start this outing.
- Model frame: The packet proves the route has a start coordinate and a first cue.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; no mapped side trail/path inside the local audit radius; vehicle corridor or service/residential road context: OSM service connector 103783, OSM service connector 95260, North Bogus Basin Road, North Bogus Shop Road; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 103783 (service) ~6m; OSM service connector 95260 (service) ~13m; North Bogus Basin Road (unclassified) ~35m; North Bogus Shop Road (service) ~38m; OSM track connector 106743 (track) ~86m; OSM service connector 59357 (service) ~90m
- Decision as runner: Before moving, find the first signed trail or road-access line and confirm the car is parked where the return cue can actually resolve.
- Wrong-layer risk: start/finish access can fail even when route geometry passes
- Evidence boundary: cue GPX waypoint plus local R2R/OSM overlay; no live imagery or field photo in this audit

### Cue 01: FOLLOW #91 Deer Point / North Bogus Shop Road / Bogus Creek Loop / North Bogus Creek Road

- Physical role: signed junction with Sunshine XC
- Model frame: The packet says `01 0.00 mi (+0.36) START/ACCESS FOLLOW #91 Deer Point / North Bogus Shop Road / Bogus Creek Loop / North Bogus Creek Road UNTIL signed junction with Sunshine XC.`.
- Runner frame: Runner frame: the immediate job is to keep the current trail until the named junction/landmark, with no extra branch proven by local data at this checkpoint.
- Likely visual field: no mapped side trail/path inside the local audit radius; vehicle corridor or service/residential road context: North Bogus Shop Road, OSM service connector 103783, OSM service connector 95260, North Bogus Basin Road; the branch to privilege is `#91 Deer Point / North Bogus Shop Road / Bogus Creek Loop / North Bogus Creek Road` until `signed junction with Sunshine XC`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: North Bogus Shop Road (service) ~2m; OSM service connector 103783 (service) ~29m; OSM service connector 95260 (service) ~48m; North Bogus Basin Road (unclassified) ~70m; OSM service connector 59357 (service) ~76m; OSM track connector 19702 (track) ~96m
- Decision as runner: Follow #91 Deer Point / North Bogus Shop Road / Bogus Creek Loop / North Bogus Creek Road until signed junction with Sunshine XC; target is Sunshine XC.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 02: FOLLOW Sunshine XC

- Physical role: signed junction with Deer Point Trail
- Model frame: The packet says `02 0.36 mi (+0.87) FOLLOW Sunshine XC UNTIL signed junction with Deer Point Trail.`.
- Runner frame: Runner frame: the immediate job is to keep the current trail until the named junction/landmark, with no extra branch proven by local data at this checkpoint.
- Likely visual field: mapped named route features near you: Bogus Creek Loop; vehicle corridor or service/residential road context: North Bogus Creek Road, OSM service connector 36782, North Bogus Shop Road, OSM track connector 8058; the branch to privilege is `Sunshine XC` until `signed junction with Deer Point Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: North Bogus Creek Road (service) ~3m; Bogus Creek Loop ~18m; OSM service connector 36782 (service) ~93m; North Bogus Shop Road (service) ~117m; OSM track connector 8058 (track) ~145m; Toll Road (track) ~160m
- Decision as runner: Follow Sunshine XC until signed junction with Deer Point Trail; target is Deer Point Trail.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 03: FOLLOW #92 Shindig / Bogus Creek Loop / Brewers Byway / Deer Point / Elk Meadows / Shindig

- Physical role: signed junction with Deer Point Trail
- Model frame: The packet says `03 1.23 mi (+0.59) CONNECTOR FOLLOW #92 Shindig / Bogus Creek Loop / Brewers Byway / Deer Point / Elk Meadows / Shindig UNTIL signed junction with Deer Point Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#142 Sunshine, #96 Brewers Byway Ext); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #142 Sunshine, #96 Brewers Byway Ext, Bogus Creek Loop; vehicle corridor or service/residential road context: #142 Sunshine; the branch to privilege is `#92 Shindig / Bogus Creek Loop / Brewers Byway / Deer Point / Elk Meadows / Shindig` until `signed junction with Deer Point Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Sunshine XC ~1m; #142 Sunshine (track) ~6m; Brewers Byway Ext ~33m; #96 Brewers Byway Ext (path) ~35m; Bogus Creek Loop (path) ~65m
- Decision as runner: Follow #92 Shindig / Bogus Creek Loop / Brewers Byway / Deer Point / Elk Meadows / Shindig until signed junction with Deer Point Trail; target is Deer Point Trail.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 04: TAKE Deer Point Trail

- Physical role: signed junction with #98 Around the Mountain Trail
- Model frame: The packet says `04 1.82 mi (+1.14) JCT TAKE Deer Point Trail UNTIL signed junction with #98 Around the Mountain Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#96 Brewer's Byway, #91 Deer Point, #144 Cabin Traverse, #94 Elk Meadows); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #96 Brewer's Byway, #91 Deer Point, #144 Cabin Traverse, #94 Elk Meadows, Deer Point Trail; vehicle corridor or service/residential road context: #144 Cabin Traverse; the branch to privilege is `Deer Point Trail` until `signed junction with #98 Around the Mountain Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #96 Brewer's Byway (path) ~2m; Brewers Byway ~2m; Deer Point ~29m; #91 Deer Point (path) ~30m; #144 Cabin Traverse (track) ~31m; Elk Meadows ~35m
- Decision as runner: Follow Deer Point Trail until signed junction with #98 Around the Mountain Trail; target is #98 Around the Mountain Trail.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 05: FOLLOW #98 Around the Mountain Trail

- Physical role: signed junction with #98 Around the Mountain Trail
- Model frame: The packet says `05 2.96 mi (+0.01) CONNECTOR FOLLOW #98 Around the Mountain Trail UNTIL signed junction with #98 Around the Mountain Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#91 Deer Point); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #91 Deer Point, #98 Around the Mountain; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#98 Around the Mountain Trail` until `signed junction with #98 Around the Mountain Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Deer Point ~2m; #91 Deer Point (path) ~22m; #98 Around the Mountain (path) ~41m; Around the Mountain ~41m
- Decision as runner: Follow #98 Around the Mountain Trail until signed junction with #98 Around the Mountain Trail; target is #98 Around the Mountain Trail.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 06: TAKE #98 Around the Mountain Trail

- Physical role: signed junction with The Face Trail
- Model frame: The packet says `06 2.97 mi (+6.64) JCT TAKE #98 Around the Mountain Trail UNTIL signed junction with The Face Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#91 Deer Point); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #91 Deer Point, #98 Around the Mountain; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `#98 Around the Mountain Trail` until `signed junction with The Face Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Deer Point ~2m; #91 Deer Point (path) ~22m; #98 Around the Mountain (path) ~41m; Around the Mountain ~41m
- Decision as runner: Follow #98 Around the Mountain Trail until signed junction with The Face Trail; target is The Face Trail.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 07: FOLLOW #95 Tempest / Lodge / Lodge Cat Track / OSM track connector 107366 / Tempest / The Face / War Eagle Road

- Physical role: signed junction with The Face Trail
- Model frame: The packet says `07 9.61 mi (+0.55) ROAD FOLLOW #95 Tempest / Lodge / Lodge Cat Track / OSM track connector 107366 / Tempest / The Face / War Eagle Road UNTIL signed junction with The Face Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#98 Around the Mountain); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #98 Around the Mountain; vehicle corridor or service/residential road context: OSM track connector 12041, Upper Nugget; the branch to privilege is `#95 Tempest / Lodge / Lodge Cat Track / OSM track connector 107366 / Tempest / The Face / War Eagle Road` until `signed junction with The Face Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #98 Around the Mountain (path) ~1m; Around the Mountain ~1m; OSM track connector 12041 (track) ~8m; Upper Nugget (track) ~49m
- Decision as runner: Follow #95 Tempest / Lodge / Lodge Cat Track / OSM track connector 107366 / Tempest / The Face / War Eagle Road until signed junction with The Face Trail; target is The Face Trail.
- Wrong-layer risk: generic OSM connector name may not exist on signs
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 08: TAKE The Face Trail

- Physical role: signed junction with Elk Meadows Trail
- Model frame: The packet says `08 10.16 mi (+1.15) JCT TAKE The Face Trail UNTIL signed junction with Elk Meadows Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#95 Tempest, #93 The Face); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #95 Tempest, #93 The Face; no vehicle corridor is proven near this checkpoint by the local overlay; the branch to privilege is `The Face Trail` until `signed junction with Elk Meadows Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #95 Tempest (path) ~1m; Tempest ~4m; The Face ~39m; #93 The Face (path) ~40m
- Decision as runner: Follow The Face Trail until signed junction with Elk Meadows Trail; target is Elk Meadows Trail.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 09: FOLLOW #144 Cabin Traverse / #93 The Face / Elk Meadows

- Physical role: signed junction with Elk Meadows Trail
- Model frame: The packet says `09 11.31 mi (+0.23) ROAD FOLLOW #144 Cabin Traverse / #93 The Face / Elk Meadows UNTIL signed junction with Elk Meadows Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#94 Elk Meadows); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #93 The Face, #144 Cabin Traverse, #94 Elk Meadows; vehicle corridor or service/residential road context: #144 Cabin Traverse; the branch to privilege is `#144 Cabin Traverse / #93 The Face / Elk Meadows` until `signed junction with Elk Meadows Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #93 The Face (path) ~1m; The Face ~1m; #144 Cabin Traverse (track) ~9m; Elk Meadows ~27m; #94 Elk Meadows (path) ~31m
- Decision as runner: Follow #144 Cabin Traverse / #93 The Face / Elk Meadows until signed junction with Elk Meadows Trail; target is Elk Meadows Trail.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 10: TAKE Elk Meadows Trail

- Physical role: end of Elk Meadows Trail for this route
- Model frame: The packet says `10 11.54 mi (+1.50) JCT TAKE Elk Meadows Trail UNTIL end of Elk Meadows Trail for this route.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#93 The Face, #144 Cabin Traverse, #94 Elk Meadows); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #93 The Face, #144 Cabin Traverse, #94 Elk Meadows; vehicle corridor or service/residential road context: #144 Cabin Traverse; the branch to privilege is `Elk Meadows Trail` until `end of Elk Meadows Trail for this route`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #93 The Face (path) ~1m; The Face ~1m; #144 Cabin Traverse (track) ~9m; Elk Meadows ~27m; #94 Elk Meadows (path) ~31m
- Decision as runner: Follow Elk Meadows Trail until end of Elk Meadows Trail for this route; target is return to car.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 11: FOLLOW #142 Sunshine / Bogus Creek Loop / Elk Meadows / Lodge / Lodge Cat Track / OSM track connector 12046 / OSM track connector 8307 / OSM track connector 91509 / Shafer Butte Road / Sunshine XC / Toll Road

- Physical role: parked car / trailhead
- Model frame: The packet says `11 13.04 mi (+2.26) EXIT FOLLOW #142 Sunshine / Bogus Creek Loop / Elk Meadows / Lodge / Lodge Cat Track / OSM track connector 12046 / OSM track connector 8307 / OSM track connector 91509 / Shafer Butte Road / Sunshine XC / Toll Road UNTIL parked car / trailhead.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (Packing Trail); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: Packing Trail; vehicle corridor or service/residential road context: Shafer Butte Road, National Forest Development Road 374, OSM track connector 111242, OSM track connector 12046; the branch to privilege is `#142 Sunshine / Bogus Creek Loop / Elk Meadows / Lodge / Lodge Cat Track / OSM track connector 12046 / OSM track connector 8307 / OSM track connector 91509 / Shafer Butte Road / Sunshine XC / Toll Road` until `parked car / trailhead`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Shafer Butte Road (track) ~0m; Elk Meadows ~2m; Lodge ~45m; National Forest Development Road 374 (track) ~71m; OSM track connector 111242 (track) ~108m; OSM track connector 12046 (track) ~117m
- Decision as runner: Follow #142 Sunshine / Bogus Creek Loop / Elk Meadows / Lodge / Lodge Cat Track / OSM track connector 12046 / OSM track connector 8307 / OSM track connector 91509 / Shafer Butte Road / Sunshine XC / Toll Road until parked car / trailhead; target is Simplot Lodge Parking Area.
- Wrong-layer risk: generic OSM connector name may not exist on signs
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Finish / return to car

- Physical role: Route endpoint / return-to-car point.
- Model frame: The packet endpoint closes the loop.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; no mapped side trail/path inside the local audit radius; vehicle corridor or service/residential road context: North Bogus Shop Road, OSM service connector 103783, OSM service connector 95260, North Bogus Basin Road; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: North Bogus Shop Road (service) ~2m; OSM service connector 103783 (service) ~29m; OSM service connector 95260 (service) ~48m; North Bogus Basin Road (unclassified) ~70m; OSM service connector 59357 (service) ~76m; OSM track connector 19702 (track) ~96m
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

- Shafer Butte Road (~0m, vehicle, highway=track, source=openstreetmap)
- #95 Tempest (~1m, path, highway=path, source=openstreetmap)
- Sunshine XC (~1m, feature, source=ridge_to_rivers_open_data)
- #96 Brewer's Byway (~2m, path, highway=path, source=openstreetmap)
- Brewers Byway (~2m, feature, source=ridge_to_rivers_open_data)
- North Bogus Creek Road (~3m, vehicle, highway=service, source=openstreetmap)
- Tempest (~4m, feature, source=ridge_to_rivers_open_data)
- #142 Sunshine (~6m, vehicle, highway=track, source=openstreetmap)
- OSM service connector 103783 (~6m, vehicle, highway=service, source=openstreetmap)
- OSM track connector 12041 (~8m, vehicle, highway=track, source=openstreetmap)
- OSM service connector 95260 (~13m, vehicle, highway=service, source=openstreetmap)
- Bogus Creek Loop (~18m, feature, source=ridge_to_rivers_open_data)
- Deer Point (~29m, feature, source=ridge_to_rivers_open_data)
- #91 Deer Point (~30m, path, highway=path, source=openstreetmap)
- #144 Cabin Traverse (~31m, vehicle, highway=track, source=openstreetmap)
- Brewers Byway Ext (~33m, feature, source=ridge_to_rivers_open_data)
- #96 Brewers Byway Ext (~35m, path, highway=path, source=openstreetmap)
- Elk Meadows (~35m, feature, source=ridge_to_rivers_open_data)
- North Bogus Basin Road (~35m, vehicle, highway=unclassified, source=openstreetmap)
- #94 Elk Meadows (~36m, path, highway=path, source=openstreetmap)

## Required Next Proof

- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.
- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.
- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.

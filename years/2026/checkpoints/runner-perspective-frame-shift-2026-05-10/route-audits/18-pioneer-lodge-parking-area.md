# Runner-Perspective Frame Shift: 18 - Pioneer Lodge Parking Area

## Frame Contract

- Route card: `18` / outing `18-1`.
- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.
- Evidence used: `docs/field-packet/field-tool-data.json`, `docs/field-packet/gpx/official/18-pioneer-lodge-parking-area-brewer-s-byway-extension-brewers-byway-shindig-tempest-trail.gpx`, R2R open data, OSM connector overlay, official 2026 segment source.
- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.
- Frame decision: `needs-proof`. The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.
- Access status: known-or-mapped parking in packet data.
- Human-validity status for this audit: `needs_visual_proof`.

## Route Snapshot

- Trails: Brewer's Byway Extension, Brewers Byway, Shindig, Tempest Trail, Lodge Trail, Mores Mtn Interpretive.
- Official miles: 5.08; on-foot miles: 11.25.
- Door-to-door: p75 320 min; p90 359 min.
- Segment count: 13; wayfinding cue count: 11.

## Start-End-Junction Frame Shifts

### Start

- Physical role: Park here and start this outing.
- Model frame: The packet proves the route has a start coordinate and a first cue.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: OSM footway connector 115829; vehicle corridor or service/residential road context: North Bogus Basin Road, Highway 40, OSM track connector 12036, OSM track connector 8307; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: North Bogus Basin Road (unclassified) ~4m; OSM footway connector 115829 (footway) ~14m; Highway 40 (track) ~17m; OSM track connector 12036 (track) ~51m; OSM track connector 8307 (track) ~59m; OSM track connector 91509 (track) ~59m
- Decision as runner: Before moving, find the first signed trail or road-access line and confirm the car is parked where the return cue can actually resolve.
- Wrong-layer risk: start/finish access can fail even when route geometry passes
- Evidence boundary: cue GPX waypoint plus local R2R/OSM overlay; no live imagery or field photo in this audit

### Cue 01: FOLLOW #98 Around the Mountain / North Bogus Basin Road / Toll Road / #142 Sunshine

- Physical role: signed junction with Brewer's Byway Extension
- Model frame: The packet says `01 0.00 mi (+1.07) START/ACCESS FOLLOW #98 Around the Mountain / North Bogus Basin Road / Toll Road / #142 Sunshine UNTIL signed junction with Brewer's Byway Extension.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (OSM footway connector 115829); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: OSM footway connector 115829; vehicle corridor or service/residential road context: North Bogus Basin Road, Highway 40, OSM track connector 12036, OSM track connector 8307; the branch to privilege is `#98 Around the Mountain / North Bogus Basin Road / Toll Road / #142 Sunshine` until `signed junction with Brewer's Byway Extension`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: North Bogus Basin Road (unclassified) ~3m; OSM footway connector 115829 (footway) ~17m; Highway 40 (track) ~25m; OSM track connector 12036 (track) ~43m; OSM track connector 8307 (track) ~55m; OSM track connector 91509 (track) ~55m
- Decision as runner: Follow #98 Around the Mountain / North Bogus Basin Road / Toll Road / #142 Sunshine until signed junction with Brewer's Byway Extension; target is Brewer's Byway Extension.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 02: FOLLOW Brewer's Byway Extension

- Physical role: signed junction with Brewers Byway
- Model frame: The packet says `02 1.07 mi (+0.60) FOLLOW Brewer's Byway Extension UNTIL signed junction with Brewers Byway.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#142 Sunshine, #96 Brewers Byway Ext, Bogus Creek Loop); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #142 Sunshine, #96 Brewers Byway Ext, Bogus Creek Loop; vehicle corridor or service/residential road context: #142 Sunshine, Connector; the branch to privilege is `Brewer's Byway Extension` until `signed junction with Brewers Byway`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #142 Sunshine (track) ~0m; Sunshine XC ~8m; Brewers Byway Ext ~37m; #96 Brewers Byway Ext (path) ~39m; Bogus Creek Loop (path) ~74m; Connector (track) ~174m
- Decision as runner: Follow Brewer's Byway Extension until signed junction with Brewers Byway; target is Brewers Byway.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 03: FOLLOW #96 Brewer's Byway / Brewers Byway

- Physical role: signed junction with Brewers Byway
- Model frame: The packet says `03 1.67 mi (+0.10) CONNECTOR FOLLOW #96 Brewer's Byway / Brewers Byway UNTIL signed junction with Brewers Byway.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#96 Brewers Byway Ext, #142 Sunshine, Brewers Cut-Off); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #96 Brewers Byway Ext, #96 Brewer's Byway, #142 Sunshine, Brewers Cut-Off; vehicle corridor or service/residential road context: #142 Sunshine; the branch to privilege is `#96 Brewer's Byway / Brewers Byway` until `signed junction with Brewers Byway`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #96 Brewers Byway Ext (path) ~1m; Brewers Byway Ext ~2m; #96 Brewer's Byway (path) ~4m; Brewers Byway ~5m; #142 Sunshine (track) ~51m; Brewers Cut-Off (path) ~147m
- Decision as runner: Follow #96 Brewer's Byway / Brewers Byway until signed junction with Brewers Byway; target is Brewers Byway.
- Wrong-layer risk: multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 04: TAKE Brewers Byway

- Physical role: signed junction with Shindig
- Model frame: The packet says `04 1.77 mi (+1.09) JCT TAKE Brewers Byway UNTIL signed junction with Shindig.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#96 Brewers Byway Ext, #96 Brewer's Byway, #142 Sunshine, Brewers Cut-Off); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #96 Brewers Byway Ext, #96 Brewer's Byway, #142 Sunshine, Brewers Cut-Off; vehicle corridor or service/residential road context: #142 Sunshine; the branch to privilege is `Brewers Byway` until `signed junction with Shindig`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #96 Brewers Byway Ext (path) ~0m; Brewers Byway Ext ~1m; #96 Brewer's Byway (path) ~8m; Brewers Byway ~8m; #142 Sunshine (track) ~53m; Brewers Cut-Off (path) ~129m
- Decision as runner: Follow Brewers Byway until signed junction with Shindig; target is Shindig.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 05: BEAR LEFT Shindig

- Physical role: signed junction with Tempest Trail
- Model frame: The packet says `05 2.86 mi (+0.12) JCT BEAR LEFT Shindig UNTIL signed junction with Tempest Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#96 Brewer's Byway, #92 Shindig, #144 Cabin Traverse, #94 Elk Meadows); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #96 Brewer's Byway, #92 Shindig, #144 Cabin Traverse, #94 Elk Meadows; vehicle corridor or service/residential road context: #144 Cabin Traverse; the branch to privilege is `Shindig` until `signed junction with Tempest Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #96 Brewer's Byway (path) ~1m; Brewers Byway ~4m; #92 Shindig (path) ~20m; Shindig ~21m; #144 Cabin Traverse (track) ~92m; Elk Meadows ~108m
- Decision as runner: Follow Shindig until signed junction with Tempest Trail; target is Tempest Trail.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 06: FOLLOW #144 Cabin Traverse / #95 Tempest / Elk Meadows / National Forest Development Road 374 / OSM track connector 8183 / Packing Trail / Shafer Butte Road / The Face

- Physical role: signed junction with Tempest Trail
- Model frame: The packet says `06 2.98 mi (+1.06) ROAD FOLLOW #144 Cabin Traverse / #95 Tempest / Elk Meadows / National Forest Development Road 374 / OSM track connector 8183 / Packing Trail / Shafer Butte Road / The Face UNTIL signed junction with Tempest Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#96 Brewer's Byway, #91 Deer Point, Deer Point Trail, #94 Elk Meadows); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #96 Brewer's Byway, #91 Deer Point, #144 Cabin Traverse, Deer Point Trail, #94 Elk Meadows; vehicle corridor or service/residential road context: #144 Cabin Traverse; the branch to privilege is `#144 Cabin Traverse / #95 Tempest / Elk Meadows / National Forest Development Road 374 / OSM track connector 8183 / Packing Trail / Shafer Butte Road / The Face` until `signed junction with Tempest Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #96 Brewer's Byway (path) ~1m; Brewers Byway ~1m; #91 Deer Point (path) ~24m; Deer Point ~24m; #144 Cabin Traverse (track) ~31m; Deer Point Trail (path) ~32m
- Decision as runner: Follow #144 Cabin Traverse / #95 Tempest / Elk Meadows / National Forest Development Road 374 / OSM track connector 8183 / Packing Trail / Shafer Butte Road / The Face until signed junction with Tempest Trail; target is Tempest Trail.
- Wrong-layer risk: generic OSM connector name may not exist on signs; multiple nearby trail lines can lure a tired runner onto a plausible wrong branch
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 07: TAKE Tempest Trail

- Physical role: signed junction with Lodge Trail
- Model frame: The packet says `07 4.04 mi (+0.81) JCT TAKE Tempest Trail UNTIL signed junction with Lodge Trail.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (Packing Trail, #95 Tempest); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: Packing Trail, #95 Tempest; vehicle corridor or service/residential road context: National Forest Development Road 374, OSM track connector 8183, Shafer Butte Road, OSM track connector 111337; the branch to privilege is `Tempest Trail` until `signed junction with Lodge Trail`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: National Forest Development Road 374 (track) ~2m; Packing Trail ~3m; OSM track connector 8183 (track) ~32m; #95 Tempest (path) ~39m; Tempest ~39m; Shafer Butte Road (track) ~49m
- Decision as runner: Follow Tempest Trail until signed junction with Lodge Trail; target is Lodge Trail.
- Wrong-layer risk: overlapping GPS line can make the correct direction ambiguous; wrong-direction choice has meaningful climb penalty
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 08: BEAR RIGHT Lodge Trail

- Physical role: signed junction with Mores Mtn Interpretive
- Model frame: The packet says `08 4.85 mi (+0.54) JCT BEAR RIGHT Lodge Trail UNTIL signed junction with Mores Mtn Interpretive.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (#95 Tempest); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: #95 Tempest; vehicle corridor or service/residential road context: Lodge Cat Track; the branch to privilege is `Lodge Trail` until `signed junction with Mores Mtn Interpretive`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: #95 Tempest (path) ~1m; Tempest ~1m; Lodge Cat Track (track) ~13m; Lodge ~15m
- Decision as runner: Follow Lodge Trail until signed junction with Mores Mtn Interpretive; target is Mores Mtn Interpretive.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 09: FOLLOW Elk Meadows / Mores Mountain Bike Trail / Mores Mountain Interpretive Trail / National Forest Development Road 374 / OSM service connector 8611 / OSM service connector 94685 / OSM service connector 94686 / OSM track connector 92827 / Shafer Butte Campground Road

- Physical role: signed junction with Mores Mtn Interpretive
- Model frame: The packet says `09 5.39 mi (+0.89) ROAD FOLLOW Elk Meadows / Mores Mountain Bike Trail / Mores Mountain Interpretive Trail / National Forest Development Road 374 / OSM service connector 8611 / OSM service connector 94685 / OSM service connector 94686 / OSM track connector 92827 / Shafer Butte Campground Road UNTIL signed junction with Mores Mtn Interpretive.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (Packing Trail); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: Packing Trail; vehicle corridor or service/residential road context: Shafer Butte Road, OSM track connector 12046, OSM track connector 12045, National Forest Development Road 374; the branch to privilege is `Elk Meadows / Mores Mountain Bike Trail / Mores Mountain Interpretive Trail / National Forest Development Road 374 / OSM service connector 8611 / OSM service connector 94685 / OSM service connector 94686 / OSM track connector 92827 / Shafer Butte Campground Road` until `signed junction with Mores Mtn Interpretive`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Lodge ~0m; Shafer Butte Road (track) ~1m; Elk Meadows ~3m; OSM track connector 12046 (track) ~70m; OSM track connector 12045 (track) ~110m; Packing Trail ~110m
- Decision as runner: Follow Elk Meadows / Mores Mountain Bike Trail / Mores Mountain Interpretive Trail / National Forest Development Road 374 / OSM service connector 8611 / OSM service connector 94685 / OSM service connector 94686 / OSM track connector 92827 / Shafer Butte Campground Road until signed junction with Mores Mtn Interpretive; target is Mores Mtn Interpretive.
- Wrong-layer risk: generic OSM connector name may not exist on signs
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 10: CONTINUE STRAIGHT Mores Mtn Interpretive

- Physical role: end of Mores Mtn Interpretive for this route
- Model frame: The packet says `10 6.28 mi (+1.93) JCT CONTINUE STRAIGHT Mores Mtn Interpretive UNTIL end of Mores Mtn Interpretive for this route.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (Mores Mountain Mountain Bike Trail, Mores Mountain Bike Trail, Mores Mountain Interpretive Trail); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: Mores Mountain Mountain Bike Trail, Mores Mountain Bike Trail, Mores Mountain Interpretive Trail; vehicle corridor or service/residential road context: OSM service connector 8611, Shafer Butte Campground Road, OSM service connector 13254, OSM service connector 94685; the branch to privilege is `Mores Mtn Interpretive` until `end of Mores Mtn Interpretive for this route`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: OSM service connector 8611 (service) ~4m; Mores Mountain Mountain Bike Trail (path) ~12m; Mores Mountain Bike Trail ~13m; Mores Mountain Interpretive Trail ~28m; Shafer Butte Campground Road (unclassified) ~75m; OSM service connector 13254 (service) ~100m
- Decision as runner: Follow Mores Mtn Interpretive until end of Mores Mtn Interpretive for this route; target is return to car.
- Wrong-layer risk: main risk is ordinary trail-sign confirmation, not a known route-data blocker
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Cue 11: FOLLOW #142 Sunshine / #96 Brewer's Byway / #98 Around the Mountain / Around the Mountain / Mores Mountain Bike Trail / Mores Mountain Interpretive Trail / National Forest Development Road 374 / OSM service connector 8611 / OSM service connector 94685 / OSM service connector 94686 / OSM track connector 107366 / OSM track connector 92827

- Physical role: parked car / trailhead
- Model frame: The packet says `11 8.21 mi (+3.40) EXIT FOLLOW #142 Sunshine / #96 Brewer's Byway / #98 Around the Mountain / Around the Mountain / Mores Mountain Bike Trail / Mores Mountain Interpretive Trail / National Forest Development Road 374 / OSM service connector 8611 / OSM service connector 94685 / OSM service connector 94686 / OSM track connector 107366 / OSM track connector 92827 UNTIL parked car / trailhead.`.
- Runner frame: Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data (Mores Mountain Mountain Bike Trail); use the signed cue and active leg before committing.
- Likely visual field: mapped trail/path choices near you: Mores Mountain Interpretive Trail, Mores Mountain Bike Trail, Mores Mountain Mountain Bike Trail; vehicle corridor or service/residential road context: Shafer Butte Campground Road, OSM service connector 8611; the branch to privilege is `#142 Sunshine / #96 Brewer's Byway / #98 Around the Mountain / Around the Mountain / Mores Mountain Bike Trail / Mores Mountain Interpretive Trail / National Forest Development Road 374 / OSM service connector 8611 / OSM service connector 94685 / OSM service connector 94686 / OSM track connector 107366 / OSM track connector 92827` until `parked car / trailhead`; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: Mores Mountain Interpretive Trail ~1m; Shafer Butte Campground Road (unclassified) ~89m; Mores Mountain Bike Trail ~132m; Mores Mountain Mountain Bike Trail (path) ~133m; OSM service connector 8611 (service) ~163m
- Decision as runner: Follow #142 Sunshine / #96 Brewer's Byway / #98 Around the Mountain / Around the Mountain / Mores Mountain Bike Trail / Mores Mountain Interpretive Trail / National Forest Development Road 374 / OSM service connector 8611 / OSM service connector 94685 / OSM service connector 94686 / OSM track connector 107366 / OSM track connector 92827 until parked car / trailhead; target is Pioneer Lodge Parking Area.
- Wrong-layer risk: generic OSM connector name may not exist on signs
- Evidence boundary: field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed

### Finish / return to car

- Physical role: Route endpoint / return-to-car point.
- Model frame: The packet endpoint closes the loop.
- Runner frame: Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem.
- Likely visual field: car/parking orientation first; mapped trail/path choices near you: OSM footway connector 115829; vehicle corridor or service/residential road context: North Bogus Basin Road, Highway 40, OSM track connector 12036, OSM track connector 8307; the branch to privilege is the first or final packet cue, not a nearby plausible line; actual visibility, signs, and car movement remain field/imagery proof gaps.
- Nearby trails/roads from local overlays: North Bogus Basin Road (unclassified) ~3m; OSM footway connector 115829 (footway) ~17m; Highway 40 (track) ~25m; OSM track connector 12036 (track) ~43m; OSM track connector 8307 (track) ~55m; OSM track connector 91509 (track) ~55m
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

- #142 Sunshine (~0m, vehicle, highway=track, source=openstreetmap)
- National Forest Development Road 374 (~2m, vehicle, highway=track, source=openstreetmap)
- Packing Trail (~3m, path, source=ridge_to_rivers_open_data)
- #96 Brewer's Byway (~4m, path, highway=path, source=openstreetmap)
- North Bogus Basin Road (~4m, vehicle, highway=unclassified, source=openstreetmap)
- OSM service connector 8611 (~4m, vehicle, highway=service, source=openstreetmap)
- Brewers Byway (~5m, feature, source=ridge_to_rivers_open_data)
- Sunshine XC (~8m, feature, source=ridge_to_rivers_open_data)
- Mores Mountain Mountain Bike Trail (~12m, path, highway=path, source=openstreetmap)
- Lodge Cat Track (~13m, vehicle, highway=track, source=openstreetmap)
- Mores Mountain Bike Trail (~13m, path, source=ridge_to_rivers_open_data)
- OSM footway connector 115829 (~14m, path, highway=footway, source=openstreetmap)
- Lodge (~15m, feature, source=ridge_to_rivers_open_data)
- Highway 40 (~17m, vehicle, highway=track, source=openstreetmap)
- #92 Shindig (~20m, path, highway=path, source=openstreetmap)
- Shindig (~21m, feature, source=ridge_to_rivers_open_data)
- #91 Deer Point (~24m, path, highway=path, source=openstreetmap)
- Deer Point (~24m, feature, source=ridge_to_rivers_open_data)
- Mores Mountain Interpretive Trail (~28m, path, source=ridge_to_rivers_open_data)
- Deer Point Trail (~32m, path, highway=path, source=openstreetmap)

## Required Next Proof

- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.
- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.
- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.

# 2026 Outing Menu

Status: written companion to the canonical outing map.

Use this like the map: pick the door-to-door time you actually have, choose one parked-start outing, then check current trail conditions and signage before leaving.

## Summary

- Open runnable outings: 23
- Manual design holds: 1
- Remaining official segments represented: 251
- Full-plan official miles: 164.42
- Full-plan on-foot miles: 280.23
- Full-plan on-foot/official ratio: 1.7x
- Map: `2026-outing-menu-map.example.html`

## How To Use

- Each row is one executable parked-start outing, not a calendar day and not a multi-start package.
- Door-to-door time uses the planner's configured origin estimate, including drive, parking/prep, access, route/return movement, and drive home.
- Completed outings are omitted when all official segment IDs in that outing are already in `completed_segment_ids`.
- If a row says it belongs to a package with multiple starts, pair it with the related start only when today's time allows.
- Manual design holds are not runnable menu items yet. They record coverage placeholders that need a better human route before scheduling.

## Manual Design Areas

### 16A Sweet Connie / Shingle / Sheep Camp manual design area

Decision: Do not schedule the Hawkins-start Sweet Connie + Shingle/Sheep placeholder as a normal outing. Keep Stack Rock Connector as a clean 16B outing, and redesign the remaining official segments from a lower Sweet/Dry access or a validated mixed-mode route.

| Current placeholder | Door-to-door | Official mi | On-foot mi | Why held |
|---|---:|---:|---:|---|
| 16A from Hawkins Range Reserve | 9h 52m | 11.62 | 36.48 | Coverage placeholder with excessive access/return mileage from Hawkins; not recommended as a runnable outing. |

Current best split probe:
- Alternatives: 16A-1, 16A-2
- Official miles: 11.62
- On-foot miles: 27.16
- Door-to-door if run separately: 7h 16m
- Improvement vs current 16A placeholder: 9.32 on-foot miles
- Probe acceptance passed: True

| Alternative | Status | Target official | Target on-foot | Required segments | Notes |
|---|---|---:|---:|---|---|
| 16A-1: Sweet Connie ascent from lower Sweet/Dry access | gpx_generated_parking_manual | 6.09 | 10.5-14.5 | 1665, 1666, 1667 | Sweet Connie should be treated as the climb, not a Hawkins errand.; If the GPX cannot preserve uphill credit cleanly, split or redesign before scheduling. Probe: graph_validated, 12.2 on-foot mi, ascent=True. |
| 16A-2: Shingle Creek + Sheep Camp lower loop | gpx_generated_parking_manual | 5.53 | 13.0-17.0 | 1656, 1653 | Use a lower Dry Creek/Shingle route shape before considering Hawkins access.; This should beat the current 19.62-mile placeholder or stay in manual design. Probe: graph_validated, 14.96 on-foot mi, ascent=True. |
| 16A-3: All-section Sweet Connie + Stack + Shingle + Sheep experiment | experimental_only | 15.12 | 24.0-32.0 | 1665, 1666, 1667, 1656, 1653, 1663, 1664 | Generate only to compare coverage-per-drive.; Reject as default if over 30 miles on foot or over 7 hours moving; 30-32 miles remains experimental only. Probe: draft, 30.59 on-foot mi, ascent=True. |
| 16A-S: Optional shuttle/drop-off comparison | comparison_only | n/a | n/a | 1665, 1666, 1667, 1656, 1653 | Allowed only as a separately-accounted variant.; Bike, car, or rideshare travel cannot count as on-foot challenge progress. |

Acceptance gates:
- Sweet Connie official segments 1665, 1666, and 1667 traverse uphill.
- Shingle Creek official segment 1656 traverses uphill.
- No GPX point gap over 0.05 miles; prefer 0.03 miles or less.
- No private/no-foot connector edges.
- No unexplained road connector.
- Replacement beats the current 36.48-mile Hawkins placeholder by at least 8 on-foot miles.
- Parking/access stays manual_required until verified.
- All-section route is rejected as default if over 30 on-foot miles or over 7 hours moving.

## 2 hours or less

| Outing | Door-to-door | Park/start | Official mi | On-foot mi | Remaining segs | Route package | Trails |
|---|---:|---|---:|---:|---:|---|---|
| 4B | 1h 7m | Upper Interpretive | 1.05 | 2.01 | 1 / 1 | Package 4 (3 starts): Table Rock / Castle / Rock Island / Shoshone-Paiute | Scott's Trail |
| 4A | 1h 17m | Bob's | 2.84 | 4.07 | 4 / 4 | Package 4 (3 starts): Table Rock / Castle / Rock Island / Shoshone-Paiute | Bob's Trail, Urban Connector |
| 1B | 1h 36m | Harrison Hollow | 4.72 | 5.69 | 12 / 12 | Package 1 (2 starts): Hillside / Harrison / West Climb frontside | Who Now Loop Trail, Harrison Ridge, Harrison Hollow, Kemper's Ridge Trail, Hippie Shake Trail |
| 7 | 1h 41m | Seamans Gulch | 2.25 | 3.77 | 6 / 6 | Package 7: Seaman / Veterans westside pod | Seaman Gulch Trail, Wild Phlox Trail |
| 8 | 1h 48m | Homestead | 2.26 | 4.05 | 3 / 3 | Package 8: Oregon Trail / Harris Ridge / Peace Valley | Harris Ridge Trail, Peace Valley Overlook |
| 16B | 1h 49m | Freddy's Stack Rock | 3.5 | 4.39 | 2 / 2 | Package 16 (2 starts): Sweet Connie / Shingle / Sheep Camp / Stack Rock | Stack Rock Connector |
| 11 | 1h 51m | Hawkins Range Reserve | 5.63 | 5.73 | 3 / 3 | Package 11: Hawkins | Hawkins |

## 2-3 hours

| Outing | Door-to-door | Park/start | Official mi | On-foot mi | Remaining segs | Route package | Trails |
|---|---:|---|---:|---:|---:|---|---|
| 15B | 2h 1m | Dry Creek Parking Area/Trailhead | 4.02 | 4.87 | 8 / 8 | Package 15 (2 starts): Dry Creek lower cluster | Red Tail Trail, Landslide |
| 1A | 2h 8m | West Climb | 3.86 | 7.39 | 10 / 10 | Package 1 (2 starts): Hillside / Harrison / West Climb frontside | Full Sail Trail, Bob Smylie, Buena Vista Trail, 36th Street Chute |
| 9 | 2h 19m | Veterans | 4.68 | 5.78 | 13 / 13 | Package 9: Eagle Bike Park / Red Tail / Rabbit pod | Veterans, Big Springs, Rabbit Run, D's Chaos, REI Connection |
| 19 | 2h 29m | Cervidae / Arrow Rock Road OSM Parking | 2.24 | 4.51 | 1 / 1 | Package 19: Cervidae Peak | Cervidae Peak |
| 14 | 2h 58m | Orchard Gulch | 8.45 | 10.74 | 6 / 6 | Package 14: Watchman / Five Mile / Orchard / Rocky Canyon | Orchard Gulch Trail, Five Mile Gulch Trail, Watchman Trail |

## 3-4 hours

| Outing | Door-to-door | Park/start | Official mi | On-foot mi | Remaining segs | Route package | Trails |
|---|---:|---|---:|---:|---:|---|---|
| 3 | 3h 1m | Freestone Creek | 8.31 | 12.13 | 28 / 28 | Package 3: Military / Cottonwood / Eagle Ridge core | Military Reserve Connection, Mountain Cove, Central Ridge Trail, Central Ridge Spur, Ridge Crest, Cottonwood Creek Trail, Connection (Eagle Ridge), Eagle Ridge Trail, Elephant Rock Loop, Heroes Trail |
| 4C | 3h 4m | Warm Springs Golf Course | 6.6 | 10.12 | 29 / 29 | Package 4 (3 starts): Table Rock / Castle / Rock Island / Shoshone-Paiute | Tram Trail, Rock Island, Rock Garden, Table Rock Trail, Quarry Trail - Castle Rock, Shoshone-Paiute, Table Rock Quarry Trail |
| 12 | 3h 27m | 8th Street ATV Parking Area | 7.81 | 12.86 | 10 / 10 | Package 12: Upper 8th / Corrals / Sidewinder | 8th Street Motorcycle Trail, Sidewinder Trail, Corrals Trail |
| 5 | 3h 41m | Cartwright | 7.99 | 13.56 | 11 / 11 | Package 5: Polecat core | Polecat Loop, Doe Ridge, Quick Draw, Barn Owl |
| 2 | 4h | Hulls Gulch | 13.11 | 17.26 | 25 / 25 | Package 2: Camel's Back / Kestrel / Crestline / Lower Hulls even-day | Lower Hull's Gulch Trail, Hull's Gulch Interpretive, Crestline Trail, Red Cliffs, Kestral Trail, Owl's Roost, Chickadee Ridge Trail, Gold Finch, 15th St. Trail |

## 4+ hours

| Outing | Door-to-door | Park/start | Official mi | On-foot mi | Remaining segs | Route package | Trails |
|---|---:|---|---:|---:|---:|---|---|
| 18 | 4h 14m | Pioneer Lodge Parking Area | 5.08 | 11.25 | 13 / 13 | Package 18: Bogus day 2: Mores / Brewers / Tempest / Lodge / Shindig | Brewer's Byway Extension, Brewers Byway, Shindig, Tempest Trail, Lodge Trail, Mores Mtn Interpretive |
| 15A | 4h 35m | MillerGulch Parking Area/Trailhead | 9.33 | 18.65 | 8 / 8 | Package 15 (2 starts): Dry Creek lower cluster | Connector, Highlands Trail, Dry Creek Trail |
| 17 | 5h 8m | Simplot Lodge Parking Area | 11.29 | 15.13 | 12 / 12 | Package 17: Bogus day 1: ATM / Deer / Elk / Sunshine | Sunshine XC, Deer Point Trail, Around the Mountain Trail, The Face Trail, Elk Meadows Trail |
| 6 | 5h 48m | Cartwright | 13.67 | 21.53 | 8 / 8 | Package 6: Cartwright / Peggy's / lower Dry Creek interface | Peggy's Trail, Chukar Butte Trail, Cartwright Connector, Cartwright Ridge, CHBH Connector |
| 13 | 6h 9m | Freestone Creek | 14.35 | 25.12 | 16 / 16 | Package 13: Freestone / Three Bears / Shane's / Curlew connector block | Three Bears Trail, Femrite's Patrol, Freestone Ridge, Two Point, Shane's Trail, Shane's Connector, Fat Tire Traverse, Curlew Connection |
| 10 | 6h 48m | Dry Creek Parking Area/Trailhead | 9.76 | 23.14 | 17 / 17 | Package 10: Harlow's / Spring Creek / north pod | Bitterbrush Trail, Currant Creek, Harlow's Hollows, Harlow's Hollows Connector, Ricochet, Shooting Range, Whistling Pig, Twisted Spring, Spring Creek |

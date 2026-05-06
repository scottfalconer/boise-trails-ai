# Strict Profile Max-Coverage Plan

Objective: extract strict-current-profile max-coverage fallback field days

## Verdict

- Accepted as completion plan: False
- Status: `partial_strict_profile_fallback_not_completion`
- Scenario: `strict_current_p90_bounds`
- P90 bounds: 260 weekday / 180 weekend

## Summary

- Field days: 31 (22 weekday / 9 weekend)
- Covered segments: 219
- Missing segments: 32
- Covered official miles: 122.45
- Missing official miles: 41.98
- Total p75: 6002 min
- Max p90 stress: 1.0

## Missing Segments

| Segment | Trail | Official mi |
|---:|---|---:|
| 1489 | Around the Mountain Trail 2 | 0.53 |
| 1490 | Around the Mountain Trail 3 | 2.0 |
| 1491 | Around the Mountain Trail 4 | 1.7 |
| 1492 | Around the Mountain Trail 5 | 0.44 |
| 1493 | Around the Mountain Trail 6 | 0.87 |
| 1540 | Deer Point Trail 1 | 1.14 |
| 1542 | Dry Creek Trail 1 | 0.58 |
| 1543 | Dry Creek Trail 2 | 0.74 |
| 1544 | Dry Creek Trail 3 | 0.99 |
| 1545 | Dry Creek Trail 4 | 3.02 |
| 1546 | Dry Creek Trail 5 | 1.65 |
| 1597 | Peggy's Trail 1 | 4.56 |
| 1653 | Sheep Camp Trail 1 | 0.77 |
| 1655 | Shindig 2 | 0.12 |
| 1656 | Shingle Creek Trail 1 | 4.76 |
| 1657 | Shooting Range 1 | 0.28 |
| 1660 | Sidewinder Trail 1 | 1.34 |
| 1661 | Spring Creek 1 | 0.08 |
| 1662 | Spring Creek 2 | 2.34 |
| 1665 | Sweet Connie Trail 1 | 0.84 |
| 1666 | Sweet Connie Trail 2 | 0.71 |
| 1667 | Sweet Connie Trail 3 | 4.53 |
| 1680 | The Face Trail 1 | 1.15 |
| 1689 | Twisted Spring 3 | 0.07 |
| 1704 | Harlow's Hollows 4 | 0.2 |
| 1705 | Harlow's Hollows 3 | 0.49 |
| 1707 | Harlow's Hollows 2 | 0.39 |
| 1708 | Harlow's Hollows Connector 1 | 0.88 |
| 1709 | Cartwright Connector 1 | 1.7 |
| 1721 | Lodge Trail 1 | 0.54 |
| 1731 | Cervidae Peak 1 | 2.24 |
| 1750 | Around the Mountain Trail 7 | 0.34 |

## Field Days

| Date | Type | P75 | P90 | Bound | Official segments | On foot | Field day |
|---|---|---:|---:|---:|---:|---:|---|
| 2026-06-18 | weekday | 207 | 232 | 260 | 14 | 9.91 | weekday-hybrid_candidate_index::combo-lower-hulls-gulch-trail-red-cliffs-kestral-trail-owls-roost-chickadee-ridge-trail-15th-st-trail-gold-finch::Hulls Gulch Trailhead |
| 2026-06-19 | weekday | 180 | 202 | 260 | 3 | 6.43 | weekday-personal_route_menu::chukar-butte-trail::Dry Creek Parking Area/Trailhead |
| 2026-06-20 | weekend | 128 | 143 | 180 | 10 | 7.39 | weekend-canonical_field_menu::combo-full-sail-trail-buena-vista-trail-bob-smylie-36th-street-chute::West Climb Trailhead |
| 2026-06-21 | weekend | 139 | 156 | 180 | 3 | 4.05 | weekend-canonical_field_menu::block-oregon_trail_harris_peace_valley::Homestead Trail Access Point |
| 2026-06-22 | weekday | 180 | 202 | 260 | 3 | 7.06 | weekday-personal_route_menu::five-mile-gulch-trail::Orchard Gulch Trail Access Point |
| 2026-06-23 | weekday | 186 | 209 | 260 | 5 | 8.47 | weekday-personal_route_menu::three-bears-trail::Five Mile Creek Trail Access Point |
| 2026-06-24 | weekday | 206 | 231 | 260 | 5 | 6.24 | weekday-personal_route_menu::mores-mtn-interpretive::Pioneer Lodge Parking Area |
| 2026-06-25 | weekday | 207 | 240 | 260 | 3 | 5.23 | weekday-personal_route_menu::sunshine-xc::Simplot Lodge Parking Area--personal_route_menu::tempest-trail::Pioneer Lodge Parking Area |
| 2026-06-26 | weekday | 213 | 242 | 260 | 6 | 10.02 | weekday-personal_route_menu::connector::MillerGulch Parking Area/Trailhead--personal_route_menu::corrals-trail::MillerGulch Parking Area/Trailhead |
| 2026-06-27 | weekend | 141 | 158 | 180 | 12 | 5.69 | weekend-canonical_field_menu::combo-who-now-loop-trail-harrison-ridge-harrison-hollow-kempers-ridge-trail-hippie-shake-trail::Harrison Hollow Trailhead |
| 2026-06-28 | weekend | 143 | 161 | 180 | 17 | 4.93 | weekend-hybrid_candidate_index::combo-table-rock-trail-quarry-trail-castle-rock-rock-garden-shoshone-paiute::Old Pen Trailhead |
| 2026-06-29 | weekday | 213 | 243 | 260 | 12 | 7.05 | weekday-personal_route_menu::seaman-gulch-trail::Seamans Gulch Trailhead--canonical_field_menu::combo-landslide-red-tail-trail::Dry Creek Parking Area/Trailhead |
| 2026-06-30 | weekday | 217 | 245 | 260 | 21 | 9.86 | weekday-personal_route_menu::central-ridge-trail::Cottonwood Creek Trailhead--hybrid_candidate_index::combo-mountain-cove-heroes-trail-cottonwood-creek-trail-eagle-ridge-trail-connection-eagle-ridge-elephant-rock-loop::Cottonwood Creek Trailhead |
| 2026-07-01 | weekday | 217 | 245 | 260 | 11 | 9.73 | weekday-personal_route_menu::hulls-gulch-interpretive::8th Street ATV Parking Area--canonical_field_menu::bobs-trail-urban-connector::Bob's Trailhead |
| 2026-07-02 | weekday | 212 | 246 | 260 | 4 | 5.77 | weekday-personal_route_menu::brewers-byway::Pioneer Lodge Parking Area--personal_route_menu::brewers-byway-extension::Pioneer Lodge Parking Area |
| 2026-07-03 | weekday | 220 | 247 | 260 | 3 | 8.59 | weekday-personal_route_menu::fat-tire-traverse-curlew-connection::Lower Interpretive Trailhead |
| 2026-07-04 | weekend | 149 | 167 | 180 | 3 | 5.73 | weekend-canonical_field_menu::block-hawkins::Hawkins Range Reserve Trailhead |
| 2026-07-05 | weekend | 152 | 171 | 180 | 12 | 6.19 | weekend-hybrid_candidate_index::combo-rock-island-table-rock-quarry-trail-tram-trail::Warm Springs Golf Course Parking/Trailhead |
| 2026-07-06 | weekday | 222 | 253 | 260 | 3 | 6.91 | weekday-personal_route_menu::elk-meadows-trail::Simplot Lodge Parking Area--canonical_field_menu::scotts-trail::Upper Interpretive Trailhead |
| 2026-07-07 | weekday | 222 | 253 | 260 | 3 | 8.68 | weekday-personal_route_menu::orchard-gulch-trail::Orchard Gulch Trail Access Point--personal_route_menu::femrites-patrol-shanes-connector::Five Mile Creek Trail Access Point |
| 2026-07-08 | weekday | 224 | 253 | 260 | 8 | 9.14 | weekday-personal_route_menu::watchman-trail::Five Mile Creek Trail Access Point--hybrid_candidate_index::combo-ridge-crest-central-ridge-spur::Freestone Creek Trailhead |
| 2026-07-09 | weekday | 222 | 255 | 260 | 3 | 7.6 | weekday-canonical_field_menu::stack-rock-connector::Freddy's Stack Rock Trailhead--single_segment_split_probe::single-segment-1488-around-the-mountain-trail::Simplot Lodge Parking Area |
| 2026-07-10 | weekday | 224 | 255 | 260 | 5 | 8.98 | weekday-personal_route_menu::military-reserve-connection::Freestone Creek Trailhead--personal_route_menu::shanes-trail::Freestone Creek Trailhead--personal_route_menu::two-point::Freestone Creek Trailhead |
| 2026-07-11 | weekend | 151 | 172 | 180 | 6 | 5.34 | weekend-personal_route_menu::highlands-trail::Bob's Trailhead--personal_route_menu::8th-street-motorcycle-trail::8th Street ATV Parking Area |
| 2026-07-12 | weekend | 149 | 175 | 180 | 2 | 3.26 | weekend-forced_anchor_probe::single-segment-1626-ricochet::Avimor Spring Valley Creek parking::Avimor Spring Valley Creek parking--forced_anchor_probe::single-segment-1687-twisted-spring::Avimor Spring Valley Creek parking::Avimor Spring Valley Creek parking |
| 2026-07-13 | weekday | 227 | 256 | 260 | 6 | 10.14 | weekday-personal_route_menu::crestline-trail::Hulls Gulch Trailhead--personal_route_menu::freestone-ridge::Freestone Creek Trailhead |
| 2026-07-14 | weekday | 215 | 257 | 260 | 3 | 5.36 | weekday-forced_anchor_probe::single-segment-1688-twisted-spring::Avimor Spring Valley Creek parking::Avimor Spring Valley Creek parking--forced_anchor_probe::single-segment-1696-whistling-pig::Avimor Spring Valley Creek parking::Avimor Spring Valley Creek parking--forced_anchor_probe::single-segment-1706-harlows-hollows::Avimor Spring Valley Creek parking::Avimor Spring Valley Creek parking |
| 2026-07-15 | weekday | 227 | 259 | 260 | 15 | 7.25 | weekday-personal_route_menu::wild-phlox-trail::Seamans Gulch Trailhead--canonical_field_menu::block-eagle_bike_park_red_tail::Veterans Trailhead |
| 2026-07-16 | weekday | 227 | 260 | 260 | 6 | 8.41 | weekday-personal_route_menu::barn-owl::Dry Creek Parking Area/Trailhead--hybrid_candidate_index::combo-currant-creek-bitterbrush-trail::Dry Creek Parking Area/Trailhead |
| 2026-07-17 | weekday | 229 | 260 | 260 | 9 | 10.19 | weekday-personal_route_menu::polecat-loop::Cartwright Trailhead--personal_route_menu::cartwright-ridge::Cartwright Trailhead |
| 2026-07-18 | weekend | 153 | 175 | 180 | 3 | 4.98 | weekend-personal_route_menu::chbh-connector::Cartwright Trailhead--hybrid_candidate_index::combo-quick-draw-doe-ridge::Cartwright Trailhead |

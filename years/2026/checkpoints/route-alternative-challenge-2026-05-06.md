# Route Alternative Challenge

Objective: challenge selected high-overhead field-menu outings against the generated candidate universe

## Summary

- Selected field-menu components: 26
- Generated candidate universe: 390
- High-overhead targets challenged: 5
- Better exact candidates found: 0
- Better superset candidates found: 0
- Current selected route is best exact generated candidate: 5
- No exact generated alternative: 0
- Manual map review still required: 5
- Targets with DEM elevation metrics: 5 / 5
- Targets with p75 door-to-door time: 5 / 5

## Target Results

| Label | Selected | Status | Best exact | Best superset | Recommendation |
|---|---|---|---|---|---|
| 13 | 14.35 official / 25.12 on-foot (1.751x, 4893 ft, 490 min p75) from Freestone Creek Trailhead | current_best_exact_candidate_in_existing_universe | block-freestone_three_bears_curlew - 14.35 official / 25.12 on-foot | none | No generated exact alternative beats the selected route; manual/local map review is still needed for absolute efficiency. |
| 15 | 9.33 official / 18.65 on-foot (1.999x, 4056 ft, 363 min p75) from MillerGulch Parking Area/Trailhead | current_best_exact_candidate_in_existing_universe | connector-highlands-trail-dry-creek-trail - 9.33 official / 18.65 on-foot | block-dry_creek_lower - 13.35 official / 32.9 on-foot; +8 segments | No generated exact alternative beats the selected route; manual/local map review is still needed for absolute efficiency. |
| 6 | 13.67 official / 21.53 on-foot (1.575x, 3495 ft, 448 min p75) from Cartwright Trailhead | current_best_exact_candidate_in_existing_universe | block-cartwright_peggy_interface - 13.67 official / 21.53 on-foot | none | No generated exact alternative beats the selected route; manual/local map review is still needed for absolute efficiency. |
| 18 | 5.08 official / 11.25 on-foot (2.215x, 1777 ft, 320 min p75) from Pioneer Lodge Parking Area | current_best_exact_candidate_in_existing_universe | block-bogus_mores_lodge_tempest - 5.08 official / 11.25 on-foot | combo-brewers-byway-extension-tempest-trail-brewers-byway-the-face-trail-mores-mtn-interpretive-lodge-trail-shindig - 6.24 official / 14.43 on-foot; +1 segments | No generated exact alternative beats the selected route; manual/local map review is still needed for absolute efficiency. |
| 19 | 2.24 official / 4.51 on-foot (2.013x, 2047 ft, 181 min p75) from Cervidae / Arrow Rock Road OSM Parking | current_best_exact_candidate_in_existing_universe | block-cervidae_peak - 2.24 official / 4.51 on-foot | none | No generated exact alternative beats the selected route; manual/local map review is still needed for absolute efficiency. |

## Details

### 13 - Freestone / Three Bears / Shane's / Curlew connector block

- Candidate id: `block-freestone_three_bears_curlew`
- Trails: Three Bears Trail, Femrite's Patrol, Freestone Ridge, Two Point, Shane's Trail, Shane's Connector, Fat Tire Traverse, Curlew Connection
- Exact alternatives checked: 2
- Superset alternatives checked: 0
- High-overlap alternatives checked: 3
- Top high-overlap generated candidates:
  - `combo-three-bears-trail-freestone-ridge-central-ridge-trail-shanes-trail-two-point-femrites-patrol-shanes-connector`: overlaps 13 segments; 12.13 official / 18.66 on-foot from Freestone Creek Trailhead
  - `combo-three-bears-trail-freestone-ridge-central-ridge-trail-shanes-trail-femrites-patrol-shanes-connector`: overlaps 12 segments; 10.92 official / 18.0 on-foot from Freestone Creek Trailhead
  - `combo-three-bears-trail-freestone-ridge-central-ridge-trail-shanes-trail-two-point`: overlaps 11 segments; 11.63 official / 17.89 on-foot from Freestone Creek Trailhead

### 15 - Dry Creek lower cluster

- Candidate id: `connector-highlands-trail-dry-creek-trail`
- Trails: Connector, Highlands Trail, Dry Creek Trail
- Exact alternatives checked: 2
- Superset alternatives checked: 2
- High-overlap alternatives checked: 0

### 6 - Cartwright / Peggy's / lower Dry Creek interface

- Candidate id: `block-cartwright_peggy_interface`
- Trails: Peggy's Trail, Chukar Butte Trail, Cartwright Connector, Cartwright Ridge, CHBH Connector
- Exact alternatives checked: 2
- Superset alternatives checked: 0
- High-overlap alternatives checked: 1
- Top high-overlap generated candidates:
  - `combo-polecat-loop-peggys-trail-cartwright-connector-chbh-connector-chukar-butte-trail`: overlaps 6 segments; 17.53 official / 27.37 on-foot from Cartwright Trailhead

### 18 - Bogus day 2: Mores / Brewers / Tempest / Lodge / Shindig

- Candidate id: `block-bogus_mores_lodge_tempest`
- Trails: Brewer's Byway Extension, Brewers Byway, Shindig, Tempest Trail, Lodge Trail, Mores Mtn Interpretive
- Exact alternatives checked: 2
- Superset alternatives checked: 1
- High-overlap alternatives checked: 3
- Top high-overlap generated candidates:
  - `combo-brewers-byway-extension-tempest-trail-brewers-byway-the-face-trail-mores-mtn-interpretive-lodge-trail`: overlaps 12 segments; 6.12 official / 12.92 on-foot from Pioneer Lodge Parking Area
  - `combo-brewers-byway-extension-tempest-trail-brewers-byway-the-face-trail-mores-mtn-interpretive-shindig`: overlaps 12 segments; 5.69 official / 13.36 on-foot from Pioneer Lodge Parking Area
  - `brewers-byway-extension-tempest-trail-brewers-byway-the-face-trail-mores-mtn-interpretive`: overlaps 11 segments; 5.58 official / 14.12 on-foot from Pioneer Lodge Parking Area

### 19 - Cervidae Peak

- Candidate id: `block-cervidae_peak`
- Trails: Cervidae Peak
- Exact alternatives checked: 3
- Superset alternatives checked: 0
- High-overlap alternatives checked: 0

## Caveats

- This report only compares already-generated candidates. It does not prove that no better hand-designed GPX exists.
- Superset alternatives are not automatic replacements because they may duplicate or move official work from another outing.
- A route remains unproven for absolute efficiency until the relevant local-map/GPX design area has also been reviewed.

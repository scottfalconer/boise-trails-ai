# Human-Executable Field-Day Layer

Generated: 2026-05-24T03:38:11Z

Objective: group certified route cards into day-level execution bundles while keeping promotion gaps visible.

## Execution Model

- Primary execution artifact: `field_day_layer`.
- Certification unit: `certified_route_card`.
- Phone default view: `field-days`.
- Route-card promotion and audit gaps stay visible until the underlying cards are audit-clean.
- Single-loop field days use the certified route-card door-to-door timing unless an explicit override explains the calendar value.
- Multi-start field days keep the calendar assignment timing as the day-level authority.
- P90 bounds are represented as dated availability values; weekday/weekend labels are context only.

## Summary

- Calendar days: 31
- Active execution days: 29
- Reserve days: 2
- Field days: 31
- Loops: 46
- Multi-start days: 13
- Coverage: 251/251 official segments
- Total p75: 6647 min
- Max p90: 359 min
- Single-loop timing repairs: 6
- Unrepaired single-loop timing mismatches: 0
- Schedule p90 violations: 0
- Total between-start drive: 76 min
- Day GPX validation passed: True
- Certified route-card loops: 46
- Needs route-card audit fix: 0
- Needs route-card promotion: 0
- Publication status: `field_day_certified`

## Field Days

| Date | Weekday | Type | P75 | P90 / bound | Loops | Transfer min | Official mi | On-foot mi | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 2026-06-18 | Thursday | weekday | 204 | 229 / 292 | 1 | 0 | 1.70 | 9.55 | executable_route_card |
| 2026-06-19 | Friday | weekday | 170 | 191 / 292 | 1 | 0 | 4.26 | 6.97 | executable_route_card |
| 2026-06-20 | Saturday | weekend | 182 | 204 / 360 | 1 | 0 | 5.10 | 8.76 | executable_route_card |
| 2026-06-21 | Sunday | weekend | 0 | 0 / 360 | 0 | 0 | 0.00 | 0.00 | reusable_empty_field_day |
| 2026-06-22 | Monday | weekday | 180 | 202 / 292 | 1 | 0 | 4.68 | 5.78 | executable_route_card |
| 2026-06-23 | Tuesday | weekday | 155 | 174 / 292 | 1 | 0 | 4.83 | 5.34 | executable_route_card |
| 2026-06-24 | Wednesday | weekday | 252 | 286 / 292 | 3 | 4 | 6.04 | 11.23 | executable_field_day |
| 2026-06-25 | Thursday | weekday | 217 | 245 / 292 | 2 | 5 | 7.91 | 9.73 | executable_field_day |
| 2026-06-26 | Friday | weekday | 220 | 247 / 292 | 1 | 0 | 4.09 | 8.59 | executable_route_card |
| 2026-06-27 | Saturday | weekend | 271 | 312 / 360 | 2 | 2 | 2.64 | 9.20 | executable_field_day |
| 2026-06-28 | Sunday | weekend | 279 | 313 / 360 | 1 | 0 | 6.64 | 10.17 | executable_route_card |
| 2026-06-29 | Monday | weekday | 214 | 248 / 292 | 2 | 0 | 2.01 | 6.12 | executable_field_day |
| 2026-06-30 | Tuesday | weekday | 228 | 259 / 292 | 2 | 0 | 3.46 | 9.04 | executable_field_day |
| 2026-07-01 | Wednesday | weekday | 227 | 260 / 292 | 2 | 0 | 3.89 | 7.95 | executable_field_day |
| 2026-07-02 | Thursday | weekday | 236 | 268 / 292 | 2 | 27 | 3.29 | 6.52 | executable_field_day |
| 2026-07-03 | Friday | weekday | 242 | 272 / 292 | 1 | 0 | 8.45 | 10.74 | executable_route_card |
| 2026-07-04 | Saturday | weekend | 289 | 324 / 360 | 1 | 0 | 7.30 | 9.64 | executable_route_card |
| 2026-07-05 | Sunday | weekend | 301 | 341 / 360 | 2 | 0 | 12.40 | 13.15 | executable_field_day |
| 2026-07-06 | Monday | weekday | 96 | 108 / 292 | 1 | 0 | 4.72 | 6.36 | executable_route_card |
| 2026-07-07 | Tuesday | weekday | 249 | 279 / 292 | 1 | 0 | 6.09 | 12.20 | executable_route_card |
| 2026-07-08 | Wednesday | weekday | 241 | 279 / 292 | 4 | 6 | 2.97 | 7.37 | executable_field_day |
| 2026-07-09 | Thursday | weekday | 250 | 280 / 292 | 1 | 0 | 8.31 | 12.13 | executable_route_card |
| 2026-07-10 | Friday | weekday | 247 | 281 / 292 | 2 | 8 | 6.27 | 8.64 | executable_field_day |
| 2026-07-11 | Saturday | weekend | 106 | 119 / 360 | 1 | 0 | 0.77 | 3.31 | executable_route_card |
| 2026-07-12 | Sunday | weekend | 320 | 359 / 360 | 1 | 0 | 5.08 | 11.25 | executable_route_card |
| 2026-07-13 | Monday | weekday | 249 | 283 / 292 | 2 | 9 | 9.13 | 10.12 | executable_field_day |
| 2026-07-14 | Tuesday | weekday | 254 | 286 / 292 | 1 | 0 | 10.19 | 13.32 | executable_route_card |
| 2026-07-15 | Wednesday | weekday | 255 | 287 / 292 | 1 | 0 | 6.72 | 13.10 | executable_route_card |
| 2026-07-16 | Thursday | weekday | 256 | 290 / 292 | 2 | 9 | 4.60 | 9.01 | executable_field_day |
| 2026-07-17 | Friday | weekday | 257 | 291 / 292 | 3 | 6 | 5.53 | 10.73 | executable_field_day |
| 2026-07-18 | Saturday | weekend | 0 | 0 / 360 | 0 | 0 | 0.00 | 0.00 | reusable_empty_field_day |

## Loop Certification Detail

### 2026-06-18 Thursday

- `FD04A` from `Freestone Creek` - `certified_route_card`

### 2026-06-19 Friday

- `FD01A` from `Warm Springs Golf Course` - `certified_route_card`

### 2026-06-20 Saturday

- `FD23C` from `8th Street ATV Parking Area` - `certified_route_card`

### 2026-06-21 Sunday

- Reserve / buffer day - no route planned.

### 2026-06-22 Monday

- `FD02A` from `Veterans` - `certified_route_card`

### 2026-06-23 Tuesday

- `FD03A` from `Chukar Butte prior parking anchor` - `certified_route_card`

### 2026-06-24 Wednesday

- `FD19A` from `Hulls Gulch` - `certified_route_card`
- `FD19B` from `Hulls Gulch` - `certified_route_card`
- `FD19C` from `Freestone Creek` - `certified_route_card`

### 2026-06-25 Thursday

- `FD05A` from `8th Street ATV Parking Area` - `certified_route_card`
- `4A` from `Bob's` - `certified_route_card`

### 2026-06-26 Friday

- `FD06A` from `Lower Interpretive` - `certified_route_card`

### 2026-06-27 Saturday

- `FD25A` from `Simplot Lodge Parking Area` - `certified_route_card`
- `FD25B` from `Pioneer Lodge Parking Area` - `certified_route_card`

### 2026-06-28 Sunday

- `FD26A` from `Simplot Lodge Parking Area` - `certified_route_card`

### 2026-06-29 Monday

- `FD07A` from `Simplot Lodge Parking Area` - `certified_route_card`
- `FD07B` from `Simplot Lodge Parking Area` - `certified_route_card`

### 2026-06-30 Tuesday

- `FD08A` from `Cartwright` - `certified_route_card`
- `FD08B` from `Cartwright` - `certified_route_card`

### 2026-07-01 Wednesday

- `FD09A` from `West Hidden Springs Drive road-parking anchor` - `certified_route_card`
- `10B` from `Dry Creek Parking Area/Trailhead` - `certified_route_card`

### 2026-07-02 Thursday

- `19` from `Cervidae / Arrow Rock Road OSM Parking` - `certified_route_card`
- `4B` from `Upper Interpretive` - `certified_route_card`

### 2026-07-03 Friday

- `14` from `Orchard Gulch` - `certified_route_card`

### 2026-07-04 Saturday

- `H1` from `Avimor Spring Valley Creek parking` - `certified_route_card`

### 2026-07-05 Sunday

- `FD28A` from `MillerGulch Parking Area/Trailhead` - `certified_route_card`
- `15A-1` from `Dry Creek / Sweet Connie roadside parking` - `certified_route_card`

### 2026-07-06 Monday

- `FD12B` from `Harrison Hollow` - `certified_route_card`

### 2026-07-07 Tuesday

- `16A-1` from `Dry Creek / Sweet Connie roadside parking` - `certified_route_card`

### 2026-07-08 Wednesday

- `FD14A` from `Cartwright` - `certified_route_card`
- `FD14B` from `Cartwright` - `certified_route_card`
- `FD14C` from `Cartwright` - `certified_route_card`
- `FD14D` from `Full Sail Trailhead, N 36th St Parking` - `certified_route_card`

### 2026-07-09 Thursday

- `FD15A` from `Freestone Creek` - `certified_route_card`

### 2026-07-10 Friday

- `7` from `Seamans Gulch` - `certified_route_card`
- `15B` from `Dry Creek Parking Area/Trailhead` - `certified_route_card`

### 2026-07-11 Saturday

- `16A-2` from `Dry Creek / Sweet Connie roadside parking` - `certified_route_card`

### 2026-07-12 Sunday

- `18` from `Pioneer Lodge Parking Area` - `certified_route_card`

### 2026-07-13 Monday

- `16B` from `Freddy's Stack Rock` - `certified_route_card`
- `11` from `Hawkins Range Reserve` - `certified_route_card`

### 2026-07-14 Tuesday

- `FD18A` from `Cartwright` - `certified_route_card`

### 2026-07-15 Wednesday

- `FD20A` from `Freestone Creek` - `certified_route_card`

### 2026-07-16 Thursday

- `FD21A` from `Homestead` - `certified_route_card`
- `FD21C` from `Old Pen` - `certified_route_card`

### 2026-07-17 Friday

- `FD22A` from `Bob's` - `certified_route_card`
- `FD22B` from `Hulls Gulch` - `certified_route_card`
- `FD22C` from `The Grove` - `certified_route_card`

### 2026-07-18 Saturday

- Reserve / buffer day - no route planned.

## Known Gaps

- Day-of Ridge to Rivers conditions, closures, heat, water, and parking checks still apply.

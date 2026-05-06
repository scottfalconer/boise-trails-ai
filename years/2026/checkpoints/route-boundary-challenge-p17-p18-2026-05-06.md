# Route Boundary Challenge

Objective: challenge package boundary routing as a combined coverage problem with distance, elevation, and p75 time

Packages: 17, 18
Areas: Bogus day 1: ATM / Deer / Elk / Sunshine, Bogus day 2: Mores / Brewers / Tempest / Lodge / Shindig

## Summary

- Target segments: 25
- Target official miles: 16.37
- Current: 26.38 mi, 708 min p75, 4379 ft ascent, 2 routes: block-bogus_atm_deer_elk_sunshine, block-bogus_mores_lodge_tempest
- Candidate pool: 7
- Candidate pool requires DEM ascent + p75 time: True
- Candidate pool excludes draft routes: True
- Covering combos returned: 20
- Better generated metric count: 2
- Dominant generated combo count: 0
- Generated combo beats current: False
- All returned combos include DEM ascent: True
- All returned combos include p75 time: True

## Best Generated Covers

- By on-foot miles: 26.38 mi, 708 min p75, 4379 ft ascent, 2 routes: block-bogus_mores_lodge_tempest, block-bogus_atm_deer_elk_sunshine
- By p75 door-to-door time: 26.38 mi, 708 min p75, 4379 ft ascent, 2 routes: block-bogus_mores_lodge_tempest, block-bogus_atm_deer_elk_sunshine
- By ascent: 29.14 mi, 748 min p75, 4049 ft ascent, 2 routes: combo-brewers-byway-extension-tempest-trail-brewers-byway-the-face-trail-mores-mtn-interpretive-lodge-trail-shindig, combo-around-the-mountain-trail-elk-meadows-trail-deer-point-trail-sunshine-xc
- By grade-adjusted miles: 29.14 mi, 748 min p75, 4049 ft ascent, 2 routes: combo-brewers-byway-extension-tempest-trail-brewers-byway-the-face-trail-mores-mtn-interpretive-lodge-trail-shindig, combo-around-the-mountain-trail-elk-meadows-trail-deer-point-trail-sunshine-xc
- Best dominant combo: none

## Metric Comparisons

| Metric | Status | Current | Best | Delta | Best candidate ids |
|---|---|---:|---:|---:|---|
| on_foot_miles | current_not_meaningfully_beaten | 26.38 | 26.38 | 0.0 | block-bogus_mores_lodge_tempest, block-bogus_atm_deer_elk_sunshine |
| door_to_door_p75_minutes | current_not_meaningfully_beaten | 708 | 708 | 0.0 | block-bogus_mores_lodge_tempest, block-bogus_atm_deer_elk_sunshine |
| ascent_ft | better_generated_combo_found | 4379 | 4049 | 330.0 | combo-brewers-byway-extension-tempest-trail-brewers-byway-the-face-trail-mores-mtn-interpretive-lodge-trail-shindig, combo-around-the-mountain-trail-elk-meadows-trail-deer-point-trail-sunshine-xc |
| grade_adjusted_miles | better_generated_combo_found | 20.79 | 20.42 | 0.37 | combo-brewers-byway-extension-tempest-trail-brewers-byway-the-face-trail-mores-mtn-interpretive-lodge-trail-shindig, combo-around-the-mountain-trail-elk-meadows-trail-deer-point-trail-sunshine-xc |

## Top Covers By On-Foot Miles

| On-foot | P75 min | Ascent | Grade-adjusted | Routes | Extra segs | Candidate ids |
|---:|---:|---:|---:|---:|---:|---|
| 26.38 | 708 | 4379 | 20.79 | 2 | 0 | block-bogus_mores_lodge_tempest, block-bogus_atm_deer_elk_sunshine |
| 29.14 | 748 | 4049 | 20.42 | 2 | 0 | combo-brewers-byway-extension-tempest-trail-brewers-byway-the-face-trail-mores-mtn-interpretive-lodge-trail-shindig, combo-around-the-mountain-trail-elk-meadows-trail-deer-point-trail-sunshine-xc |
| 29.56 | 760 | 4125 | 21.67 | 2 | 0 | combo-brewers-byway-extension-tempest-trail-brewers-byway-the-face-trail-mores-mtn-interpretive-lodge-trail-shindig, block-bogus_atm_deer_elk_sunshine |
| 31.73 | 884 | 4550 | 21.62 | 3 | 0 | block-bogus_mores_lodge_tempest, block-bogus_atm_deer_elk_sunshine, combo-lodge-trail-shindig |
| 32.98 | 898 | 4155 | 21.07 | 3 | 0 | combo-brewers-byway-extension-tempest-trail-brewers-byway-the-face-trail-mores-mtn-interpretive-lodge-trail, combo-around-the-mountain-trail-elk-meadows-trail-deer-point-trail-sunshine-xc, combo-lodge-trail-shindig |
| 33.4 | 910 | 4231 | 22.32 | 3 | 0 | combo-brewers-byway-extension-tempest-trail-brewers-byway-the-face-trail-mores-mtn-interpretive-lodge-trail, block-bogus_atm_deer_elk_sunshine, combo-lodge-trail-shindig |
| 33.42 | 904 | 4081 | 20.57 | 3 | 0 | combo-brewers-byway-extension-tempest-trail-brewers-byway-the-face-trail-mores-mtn-interpretive-shindig, combo-around-the-mountain-trail-elk-meadows-trail-deer-point-trail-sunshine-xc, combo-lodge-trail-shindig |
| 33.84 | 916 | 4157 | 21.82 | 3 | 0 | combo-brewers-byway-extension-tempest-trail-brewers-byway-the-face-trail-mores-mtn-interpretive-shindig, block-bogus_atm_deer_elk_sunshine, combo-lodge-trail-shindig |
| 34.49 | 924 | 4220 | 21.25 | 3 | 0 | combo-brewers-byway-extension-tempest-trail-brewers-byway-the-face-trail-mores-mtn-interpretive-lodge-trail-shindig, combo-around-the-mountain-trail-elk-meadows-trail-deer-point-trail-sunshine-xc, combo-lodge-trail-shindig |
| 34.91 | 936 | 4296 | 22.5 | 3 | 0 | combo-brewers-byway-extension-tempest-trail-brewers-byway-the-face-trail-mores-mtn-interpretive-lodge-trail-shindig, block-bogus_atm_deer_elk_sunshine, combo-lodge-trail-shindig |

## Caveats

- This is a generated-candidate boundary challenge. It can disprove obvious generated alternatives, but it does not replace manual GPX/local-map design.
- Door-to-door p75 times are summed per route, so splitting a boundary across more starts is penalized with each route's drive and prep time.
- Extra official segments are reported because a superset route may only be useful if it replaces work from another package.

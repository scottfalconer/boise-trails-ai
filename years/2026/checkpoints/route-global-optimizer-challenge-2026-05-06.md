# Route Global Optimizer Challenge

Objective: challenge the complete field menu against a global executable set-cover optimizer

## Summary

- Target segments: 251
- Candidate pool: 107
- Current menu: 268.2 mi, 6220 min p75, 54924 ft ascent, 26 routes
- Successful optimizer solutions: 4
- Dominant solutions: 0
- Global optimizer beats current: False
- Best dominant solution: none

## Solutions

| Objective | Success | On-foot | P75 min | Ascent | Grade-adjusted | Routes | Current routes | Generated routes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| on_foot_miles | True | 268.2 | 6220 | 54924 | 219.28 | 26 | 26 | 0 |
| door_to_door_p75_minutes | True | 268.2 | 6220 | 54924 | 219.28 | 26 | 26 | 0 |
| ascent_ft | True | 268.2 | 6220 | 54924 | 219.28 | 26 | 26 | 0 |
| balanced | True | 268.2 | 6220 | 54924 | 219.28 | 26 | 26 | 0 |

## Dominance Checks

| Candidate | Dominates | Better metrics | Worse metrics | Deltas |
|---|---|---|---|---|
| block-westside_seaman_veterans, block-hawkins, block-cervidae_peak, combo-landslide-red-tail-trail, combo-full-sail-trai | False |  |  | {'on_foot_miles': 0.0, 'door_to_door_p75_minutes': 0.0, 'ascent_ft': 0.0, 'grade_adjusted_miles': 0.0} |
| block-westside_seaman_veterans, block-hawkins, block-cervidae_peak, combo-landslide-red-tail-trail, combo-full-sail-trai | False |  |  | {'on_foot_miles': 0.0, 'door_to_door_p75_minutes': 0.0, 'ascent_ft': 0.0, 'grade_adjusted_miles': 0.0} |
| block-westside_seaman_veterans, block-hawkins, block-cervidae_peak, combo-landslide-red-tail-trail, combo-full-sail-trai | False |  |  | {'on_foot_miles': 0.0, 'door_to_door_p75_minutes': 0.0, 'ascent_ft': 0.0, 'grade_adjusted_miles': 0.0} |
| block-westside_seaman_veterans, block-hawkins, block-cervidae_peak, combo-landslide-red-tail-trail, combo-full-sail-trai | False |  |  | {'on_foot_miles': 0.0, 'door_to_door_p75_minutes': 0.0, 'ascent_ft': 0.0, 'grade_adjusted_miles': 0.0} |

## Caveats

- This optimizer only uses executable generated candidates with DEM ascent and p75 time, plus the current field-menu components.
- Draft generated routes are excluded. A draft route can seed manual GPX design but cannot beat the current executable menu.
- Set-cover results are proof candidates; any route selected outside the current menu still needs field-facing GPX and cue promotion before use.

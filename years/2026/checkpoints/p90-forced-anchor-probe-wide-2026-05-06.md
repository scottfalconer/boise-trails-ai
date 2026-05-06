# P90 Forced Anchor Probe

Objective: test p90-missing official segments against nearby forced parking anchors

## Summary

- Target segments: 15
- Probe rows: 1050
- P90 bound: 260 min
- Rows under p90 bound: 16
- Strict field-ready segments resolved: 2
- Conditional-only segments resolved: 12
- Strict missing segment ids: 1626, 1656, 1657, 1661, 1662, 1687, 1688, 1689, 1696, 1705, 1706, 1707, 1708
- Conditional missing segment ids: 1656

## Strict Best Rows

| Segment | Trail | Anchor | P90 | P75 | On foot | Parking |
|---:|---|---|---:|---:|---:|---|
| 1545 | Dry Creek Trail | Strava parking anchor 23 | 228 | 203 | 9.21 | strava_seen_prior_challenge_window |
| 1667 | Sweet Connie Trail | Strava parking anchor 23 | 250 | 223 | 10.04 | strava_seen_prior_challenge_window |

## Conditional Best Rows

| Segment | Trail | Anchor | P90 | P75 | On foot | Field ready | Parking |
|---:|---|---|---:|---:|---:|---|---|
| 1545 | Dry Creek Trail | Dry Creek / Sweet Connie roadside parking | 226 | 201 | 9.15 | False | source_verified_roadside |
| 1626 | Ricochet | Harlow's / Hidden Springs west access probe | 104 | 92 | 1.1 | False | manual_required |
| 1657 | Shooting Range | Harlow's / Hidden Springs west access probe | 108 | 96 | 1.29 | False | manual_required |
| 1661 | Spring Creek | Harlow's / Hidden Springs west access probe | 79 | 70 | 0.13 | False | manual_required |
| 1662 | Spring Creek | Harlow's / Hidden Springs west access probe | 161 | 143 | 3.61 | False | manual_required |
| 1667 | Sweet Connie Trail | Dry Creek / Sweet Connie roadside parking | 248 | 221 | 9.96 | False | source_verified_roadside |
| 1687 | Twisted Spring | Harlow's / Hidden Springs west access probe | 108 | 96 | 1.38 | False | manual_required |
| 1688 | Twisted Spring | Harlow's / Hidden Springs west access probe | 90 | 80 | 0.75 | False | manual_required |
| 1689 | Twisted Spring | Harlow's / Hidden Springs west access probe | 81 | 72 | 0.22 | False | manual_required |
| 1696 | Whistling Pig | Harlow's / Hidden Springs west access probe | 136 | 121 | 2.29 | False | manual_required |
| 1705 | Harlow's Hollows | Harlow's / Hidden Springs west access probe | 154 | 137 | 3.27 | False | manual_required |
| 1706 | Harlow's Hollows | Harlow's / Hidden Springs west access probe | 90 | 80 | 0.56 | False | manual_required |
| 1707 | Harlow's Hollows | Harlow's / Hidden Springs west access probe | 135 | 120 | 2.45 | False | manual_required |
| 1708 | Harlow's Hollows Connector | Harlow's / Hidden Springs west access probe | 129 | 115 | 1.98 | False | manual_required |

## Caveats

- This is still a probe, not a promoted field-menu route set.
- Strict rows require field-ready parking. Conditional rows may depend on manual parking/access verification.
- Private Strava-derived anchor coordinates stay in the ignored private source file; this report records names and metrics only.

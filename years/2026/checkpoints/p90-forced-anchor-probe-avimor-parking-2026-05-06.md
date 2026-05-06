# P90 Forced Anchor Probe

Objective: test p90-missing official segments against nearby forced parking anchors

## Summary

- Target segments: 15
- Probe rows: 180
- P90 bound: 260 min
- Rows under p90 bound: 40
- Strict field-ready segments resolved: 14
- Conditional-only segments resolved: 0
- Strict missing segment ids: 1656
- Conditional missing segment ids: 1656

## Strict Best Rows

| Segment | Trail | Anchor | P90 | P75 | On foot | Parking |
|---:|---|---|---:|---:|---:|---|
| 1545 | Dry Creek Trail | Strava parking anchor 23 | 228 | 203 | 9.21 | strava_seen_prior_challenge_window |
| 1626 | Ricochet | Avimor Spring Valley Creek parking | 138 | 123 | 2.36 | osm_amenity_parking_fee_no_capacity_36_source_checked |
| 1657 | Shooting Range | Avimor Spring Valley Creek parking | 123 | 109 | 1.73 | osm_amenity_parking_fee_no_capacity_36_source_checked |
| 1661 | Spring Creek | Avimor Spring Valley Creek parking | 117 | 104 | 1.49 | osm_amenity_parking_fee_no_capacity_36_source_checked |
| 1662 | Spring Creek | Avimor Spring Valley Creek parking | 201 | 179 | 5.07 | osm_amenity_parking_fee_no_capacity_36_source_checked |
| 1667 | Sweet Connie Trail | Strava parking anchor 23 | 250 | 223 | 10.04 | strava_seen_prior_challenge_window |
| 1687 | Twisted Spring | Avimor Spring Valley Creek parking | 97 | 86 | 0.9 | osm_amenity_parking_fee_no_capacity_36_source_checked |
| 1688 | Twisted Spring | Avimor Spring Valley Creek parking | 114 | 101 | 1.33 | osm_amenity_parking_fee_no_capacity_36_source_checked |
| 1689 | Twisted Spring | Avimor Spring Valley Creek parking | 115 | 102 | 1.38 | osm_amenity_parking_fee_no_capacity_36_source_checked |
| 1696 | Whistling Pig | Avimor Spring Valley Creek parking | 133 | 118 | 2.01 | osm_amenity_parking_fee_no_capacity_36_source_checked |
| 1705 | Harlow's Hollows | Avimor Twisted Spring parking | 175 | 156 | 4.35 | osm_amenity_parking_capacity_16_source_checked |
| 1706 | Harlow's Hollows | Avimor Spring Valley Creek parking | 130 | 116 | 2.02 | osm_amenity_parking_fee_no_capacity_36_source_checked |
| 1707 | Harlow's Hollows | Avimor Twisted Spring parking | 162 | 144 | 3.57 | osm_amenity_parking_capacity_16_source_checked |
| 1708 | Harlow's Hollows Connector | Avimor Twisted Spring parking | 157 | 140 | 3.26 | osm_amenity_parking_capacity_16_source_checked |

## Conditional Best Rows

| Segment | Trail | Anchor | P90 | P75 | On foot | Field ready | Parking |
|---:|---|---|---:|---:|---:|---|---|
| 1545 | Dry Creek Trail | Dry Creek / Sweet Connie roadside parking | 226 | 201 | 9.15 | False | source_verified_roadside |
| 1626 | Ricochet | Harlow's / Hidden Springs west access probe | 104 | 92 | 1.1 | False | manual_required |
| 1657 | Shooting Range | Harlow's / Hidden Springs west access probe | 108 | 96 | 1.29 | False | manual_required |
| 1661 | Spring Creek | Harlow's / Hidden Springs west access probe | 79 | 70 | 0.13 | False | manual_required |
| 1662 | Spring Creek | Harlow's / Hidden Springs west access probe | 161 | 143 | 3.61 | False | manual_required |
| 1667 | Sweet Connie Trail | Dry Creek / Sweet Connie roadside parking | 248 | 221 | 9.96 | False | source_verified_roadside |
| 1687 | Twisted Spring | Avimor Spring Valley Creek parking | 97 | 86 | 0.9 | True | osm_amenity_parking_fee_no_capacity_36_source_checked |
| 1688 | Twisted Spring | Harlow's / Hidden Springs west access probe | 90 | 80 | 0.75 | False | manual_required |
| 1689 | Twisted Spring | Harlow's / Hidden Springs west access probe | 81 | 72 | 0.22 | False | manual_required |
| 1696 | Whistling Pig | Avimor Spring Valley Creek parking | 133 | 118 | 2.01 | True | osm_amenity_parking_fee_no_capacity_36_source_checked |
| 1705 | Harlow's Hollows | Harlow's / Hidden Springs west access probe | 154 | 137 | 3.27 | False | manual_required |
| 1706 | Harlow's Hollows | Harlow's / Hidden Springs west access probe | 90 | 80 | 0.56 | False | manual_required |
| 1707 | Harlow's Hollows | Harlow's / Hidden Springs west access probe | 135 | 120 | 2.45 | False | manual_required |
| 1708 | Harlow's Hollows Connector | Harlow's / Hidden Springs west access probe | 129 | 115 | 1.98 | False | manual_required |

## Caveats

- This is still a probe, not a promoted field-menu route set.
- Strict rows require field-ready parking. Conditional rows may depend on manual parking/access verification.
- Private Strava-derived anchor coordinates stay in the ignored private source file; this report records names and metrics only.

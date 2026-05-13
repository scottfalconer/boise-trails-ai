# Harlow / Avimor H1 Gate Repair Audit

Generated: 2026-05-13T00:45:52Z

Decision: `keep_gated_repaired_candidate`

## Summary

- Current cluster: 34.0 mi / 991 p75 / 1117 p90
- Repaired H1: 9.64 mi / 289 p75 / 324 p90
- Delta: -24.36 mi / -702 p75 / -793 p90
- Direct gap fallback: 0.43 mi -> 0.0 mi
- Explicit official repeat: 0.61 mi across `1626, 1661, 1687, 1688, 1689, 1704`
- Hypothetical replacement coverage: 251/251 official segments

## Gate Status

- Repaired/explained: direct_gap_fallback, hidden_self_repeat_accounting, candidate_dem_p75_p90_reprice, candidate_field_cue_sheet, set_coverage_after_replacement
- Remaining blockers: needs_public_safe_cueable_access_review, needs_field_packet_route_card_promotion, needs_field_packet_recertification

## Direct Gap Repairs

| Target | Old gap | Repaired path | Repeat priced | Connector names |
|---|---:|---:|---:|---|
| `1687` Twisted Spring 1 | 0.06 | 0.1 | 0.01 | OSM footway connector 22098, OSM footway connector 22306, OSM footway connector 22310, OSM footway connector 22443, Twisted Spring Trail - #8 |
| `1704` Harlow's Hollows 4 | 0.37 | 0.47 | 0.05 | Burnt Car Draw - #10, Cartwright Road - #20, OSM path connector 105691, OSM path connector 13946, OSM path connector 13947, The Wall - #29 |

## Field Cue Sheet

1. **Start at Avimor Spring Valley Creek parking** - Begin from the public parking anchor and follow the mapped Twisted Spring access connector toward Twisted Spring Trail #8. Start the BTC recording before leaving the car. The first official credit leg is Twisted Spring 1; the access snap includes a short priced repeat, not hidden new credit.
2. **Twisted Spring sequence** - Run Twisted Spring 1, Twisted Spring 2, and Twisted Spring 3 in the generated direction, then continue through the short Ricochet connector. Credit targets: 1687, 1688, 1689, then 1626.
3. **Ricochet to Shooting Range and Whistling Pig** - Use the mapped North Smokeys Draw Place / Ricochet #2 / Shooting Range #5 connector, then continue onto Whistling Pig. Credit targets: 1657 and 1696.
4. **Connector to Spring Creek** - Use Twisted Spring Trail - #8 / Whistling Pig - #3 to reach Spring Creek 1. Includes 0.21 mi repeat official (1688, 1689); no new credit. This connector explicitly prices the Twisted Spring repeat before Spring Creek credit.
5. **Spring Creek** - Run Spring Creek 1 and Spring Creek 2 in sequence. Credit targets: 1661 and 1662.
6. **Connector to Harlow's Hollows** - Use Burnt Car Draw - #10 / Cartwright Road - #20 / The Wall - #29 to reach Harlow's Hollows 4. Includes 0.05 mi repeat official (1704); no new credit. The old straight-line gap is replaced by a mapped graph path using Burnt Car Draw, Cartwright Road, The Wall, and OSM path connectors.
7. **Harlow's Hollows chain** - Run Harlow's Hollows 4, Harlow's Hollows 3, Harlow's Hollows 2, Harlow's Hollows Connector, and Harlow's Hollows 1. Credit targets: 1704, 1705, 1707, 1708, and 1706.
8. **Return to the car** - Return to Avimor Spring Valley Creek parking on Spring Creek - #9 / Twisted Spring Trail - #8. Includes 0.34 mi repeat official (1626, 1661, 1687, 1688); no new credit. Return leg is not new official credit and includes explicit repeat mileage.

## Repeat Audit

- Status: `passed`
- Hidden self-repeat ids: []
- Unpriced repeat ids: []
- Latent credit ids: []

## Parking Source Sync

- Status: `already_synced_in_field_packet_source`
- Candidate confidence: `osm_amenity_parking_fee_no_capacity_36_source_checked`
- Source: `osm_overpass_amenity_parking_2026_05_06_plus_alltrails_spring_valley_creek`

## Not Promoted

This audit does not remove FD27A, FD27B, FD27C, FD24A, or FD30A from the active packet. Promotion still requires public-safe access review, route-card source replacement, packet regeneration, and the normal certification chain.

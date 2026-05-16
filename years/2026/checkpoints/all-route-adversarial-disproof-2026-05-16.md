# All-Route Adversarial Disproof Pass

Date: 2026-05-16

Purpose: attack every current 2026 route one by one and decide whether it holds, stays gated, or needs repair. This is deliberately stronger than the prior group-level proof: the route set does not get credit merely because the packet-level audits are green.

Frame decision: hold with explicit gates.

This is not a magic global optimum claim. It proves that, against the current accepted anchors, route-review pack, generated candidate universe, p75/DEM evidence, repeat-accounting checks, boundary/global challenge artifacts, and field-packet consistency checks, I did not find a known current replacement that is both materially better and field-ready. A future legal anchor, current closure, or certified hand-built GPX can still disprove an individual route.

## Summary

- Routes attacked: 43
- Route proof records: 43
- Deterministic same-credit dominance failures: 0
- Repeat optimization warnings now open: 0
- Repeat optimization warnings closed: 39
- Efficiency verdict: proven with public-access caveat (achieved=True)

Decision counts:
- `HOLD_CONDITION_GATED`: 6
- `HOLD_PROVEN_BUNDLE_PRESSURE`: 12
- `HOLD_PROVEN_CURRENT`: 12
- `HOLD_PROVEN_HIGH_COST`: 9
- `HOLD_PUBLIC_ACCESS_RECHECK`: 1
- `HOLD_PROVISIONAL_FIELD_WALKTHROUGH`: 3

## Frames Used

- Exact-credit/start attack: could the same official segment set be earned from an accepted anchor with at least 0.25 miles or 10 p75 minutes saved?
- Bundle/partition attack: do same-trailhead or boundary routes create a better human route if combined or split differently?
- Runnable-cost attack: are high ratio, high non-credit, or declared-repeat miles actually priced with p75 and DEM effort, or just hidden fatigue?
- Human-validity/artifact attack: do route card, map data, GPX/cues, route review, and audits point at the same car-to-car outing?
- Displacement attack: does the current global/generated candidate universe produce a dominant replacement, even if it is not the literal route under review?

## Route-by-Route Disproof Ledger

### FD28A - Miller Gulch singleton

- Decision: `HOLD_PROVEN_CURRENT`
- Candidate: `connector`
- Start: MillerGulch Parking Area/Trailhead
- Official segments: 1523
- Trails: Connector
- Cost: 0.67 official mi / 1.26 on-foot mi / 1.88x / 0.59 non-credit mi / 0.01 declared-repeat mi / p75 51 min / p90 63 min
- Closed warning codes: none
- Same-credit/start attack: Same-credit alternatives are non-dominant in the current pack; best listed savings 0.00 mi / 0 min.
- Bundle/boundary/global attack: Attacked as a tiny-credit out-and-back. Same-credit review found no material accepted-anchor improvement beyond the current Miller Gulch start, and no repeat/bundle audit warning remains open.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held as current route; no material disproof pressure remains in current audits.
- What would disprove later: A certified, legal, same-credit or same-day replacement with current parking/access evidence, p75/DEM effort, complete GPX/cues, and at least 0.25 miles or 10 p75 minutes saved.

### FD19A - 8th Street / Hulls / Crestline

- Decision: `HOLD_PROVEN_BUNDLE_PRESSURE`
- Candidate: `kestral-trail`
- Start: Hulls Gulch
- Official segments: 1583
- Trails: Kestral Trail
- Cost: 0.75 official mi / 1.55 on-foot mi / 2.07x / 0.80 non-credit mi / 0.01 declared-repeat mi / p75 55 min / p90 62 min
- Closed warning codes: `same_trailhead_bundle_candidate`
- Same-credit/start attack: Same-credit alternatives are non-dominant in the current pack; best listed savings 0.10 mi / 6 min.
- Bundle/boundary/global attack: Attacked by merging same-trailhead Hulls cards and by substituting the old 8th/Crestline superset. The superset is worse, and same-trailhead rows are scheduling opportunities, not certified lower-footmile replacements.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held despite same-trailhead bundle pressure because bundling is not proven to reduce runnable cost.
- What would disprove later: A certified same-trailhead Hulls or 8th Street bundle that proves lower total p75/on-foot burden without turning a short outing into a harder field day.

### FD14A - Cartwright / Polecat / Peggy

- Decision: `HOLD_PROVEN_BUNDLE_PRESSURE`
- Candidate: `doe-ridge`
- Start: Cartwright
- Official segments: 1541
- Trails: Doe Ridge
- Cost: 0.46 official mi / 1.08 on-foot mi / 2.35x / 0.62 non-credit mi / 0.10 declared-repeat mi / p75 58 min / p90 65 min
- Closed warning codes: `same_trailhead_bundle_candidate`
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked by bundling Cartwright cards and recombining Polecat/Peggy boundaries. The boundary challenge and repeat audit found no dominant generated combo with p75, DEM effort, and field-ready continuity.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held despite same-trailhead bundle pressure because bundling is not proven to reduce runnable cost.
- What would disprove later: A certified Cartwright/Polecat/Peggy bundle that reduces car-to-car p75 or on-foot burden and keeps the cues usable at overlapping junctions.

### FD14D - Lower 36th / Full Sail repaired singleton

- Decision: `HOLD_PROVISIONAL_FIELD_WALKTHROUGH`
- Candidate: `accepted-replacement-fd14d-36th-street-chute-lower-36th`
- Start: Full Sail Trailhead, N 36th St Parking
- Official segments: 1482
- Trails: 36th Street Chute
- Cost: 0.74 official mi / 1.50 on-foot mi / 2.03x / 0.76 non-credit mi / 0.74 declared-repeat mi / p75 60 min / p90 68 min
- Closed warning codes: none
- Same-credit/start attack: Same-credit alternatives are non-dominant in the current pack; best listed savings 0.14 mi / 6 min.
- Bundle/boundary/global attack: Attacked with the original FD14D failure: stale Full Sail versus lower N 36th. The current route is already the repaired lower-anchor version; remaining risk is field-walkthrough promotion, not known dominance.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held against known dominance, but not promoted beyond its separate field-walkthrough gate.
- What would disprove later: A certified, legal, same-credit or same-day replacement with current parking/access evidence, p75/DEM effort, complete GPX/cues, and at least 0.25 miles or 10 p75 minutes saved.

### 4B - Upper Interpretive singleton

- Decision: `HOLD_PROVEN_CURRENT`
- Candidate: `scotts-trail`
- Start: Upper Interpretive
- Official segments: 1643
- Trails: Scott's Trail
- Cost: 1.05 official mi / 2.01 on-foot mi / 1.91x / 0.96 non-credit mi / 0.02 declared-repeat mi / p75 79 min / p90 89 min
- Closed warning codes: none
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked as a short-credit access route. No same-credit accepted replacement or repeat/bundle pressure is present in the current packet.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held as current route; no material disproof pressure remains in current audits.
- What would disprove later: A certified, legal, same-credit or same-day replacement with current parking/access evidence, p75/DEM effort, complete GPX/cues, and at least 0.25 miles or 10 p75 minutes saved.

### FD22C - The Grove / Gold Finch connector group

- Decision: `HOLD_PROVEN_CURRENT`
- Candidate: `combo-owls-roost-chickadee-ridge-trail-15th-st-trail-gold-finch`
- Start: The Grove
- Official segments: 1596, 1518, 1517, 1481, 1567, 1568
- Trails: Owl's Roost, Chickadee Ridge Trail, 15th St. Trail, Gold Finch
- Cost: 2.03 official mi / 3.51 on-foot mi / 1.73x / 1.48 non-credit mi / 0.68 declared-repeat mi / p75 91 min / p90 102 min
- Closed warning codes: none
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked as a possible Hulls/15th Street bundle. It carries several exact official segments from The Grove with no current same-credit or repeat-warning displacement.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held as current route; no material disproof pressure remains in current audits.
- What would disprove later: A certified, legal, same-credit or same-day replacement with current parking/access evidence, p75/DEM effort, complete GPX/cues, and at least 0.25 miles or 10 p75 minutes saved.

### 4A - Bob's / Highlands singleton

- Decision: `HOLD_PROVEN_CURRENT`
- Candidate: `bobs-trail-urban-connector`
- Start: Bob's
- Official segments: 1498, 1499, 1500, 1690
- Trails: Bob's Trail, Urban Connector
- Cost: 2.84 official mi / 4.07 on-foot mi / 1.43x / 1.23 non-credit mi / 0.04 declared-repeat mi / p75 97 min / p90 109 min
- Closed warning codes: none
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked as a possible Table Rock/Highlands boundary merge. Existing challenge artifacts did not produce a better exact or boundary replacement, and the current route is below high-ratio pressure.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held as current route; no material disproof pressure remains in current audits.
- What would disprove later: A certified, legal, same-credit or same-day replacement with current parking/access evidence, p75/DEM effort, complete GPX/cues, and at least 0.25 miles or 10 p75 minutes saved.

### FD09A - Hidden Springs repaired singleton

- Decision: `HOLD_PROVISIONAL_FIELD_WALKTHROUGH`
- Candidate: `accepted-replacement-fd09a-barn-owl-west-hidden-springs`
- Start: West Hidden Springs Drive road-parking anchor
- Official segments: 1494, 1495
- Trails: Barn Owl
- Cost: 1.44 official mi / 2.52 on-foot mi / 1.75x / 1.08 non-credit mi / 0.61 declared-repeat mi / p75 100 min / p90 112 min
- Closed warning codes: none
- Same-credit/start attack: Same-credit alternatives are non-dominant in the current pack; best listed savings 0.20 mi / 3 min.
- Bundle/boundary/global attack: Attacked with the FD09A stale Dry Creek start regression. The current route is the accepted West Hidden Springs repair; remaining risk is field-walkthrough promotion, not a known accepted-anchor dominance.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held against known dominance, but not promoted beyond its separate field-walkthrough gate.
- What would disprove later: A certified, legal, same-credit or same-day replacement with current parking/access evidence, p75/DEM effort, complete GPX/cues, and at least 0.25 miles or 10 p75 minutes saved.

### FD14B - Cartwright / Polecat / Peggy

- Decision: `HOLD_PROVEN_BUNDLE_PRESSURE`
- Candidate: `chbh-connector`
- Start: Cartwright
- Official segments: 1516, 1610
- Trails: CHBH Connector, Quick Draw
- Cost: 1.29 official mi / 3.16 on-foot mi / 2.45x / 1.87 non-credit mi / 0.67 declared-repeat mi / p75 103 min / p90 119 min
- Closed warning codes: `same_trailhead_bundle_candidate`
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked by bundling Cartwright cards and recombining Polecat/Peggy boundaries. The boundary challenge and repeat audit found no dominant generated combo with p75, DEM effort, and field-ready continuity.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held despite same-trailhead bundle pressure because bundling is not proven to reduce runnable cost.
- What would disprove later: A certified Cartwright/Polecat/Peggy bundle that reduces car-to-car p75 or on-foot burden and keeps the cues usable at overlapping junctions.

### FD19B - 8th Street / Hulls / Crestline

- Decision: `HOLD_PROVEN_BUNDLE_PRESSURE`
- Candidate: `lower-hulls-gulch-trail-red-cliffs`
- Start: Hulls Gulch
- Official segments: 1585, 1586, 1587, 1588, 1589, 1615, 1616
- Trails: Lower Hull's Gulch Trail, Red Cliffs
- Cost: 3.45 official mi / 4.92 on-foot mi / 1.43x / 1.47 non-credit mi / 0.18 declared-repeat mi / p75 104 min / p90 117 min
- Closed warning codes: `same_trailhead_bundle_candidate`
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked by merging same-trailhead Hulls cards and by substituting the old 8th/Crestline superset. The superset is worse, and same-trailhead rows are scheduling opportunities, not certified lower-footmile replacements.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held despite same-trailhead bundle pressure because bundling is not proven to reduce runnable cost.
- What would disprove later: A certified same-trailhead Hulls or 8th Street bundle that proves lower total p75/on-foot burden without turning a short outing into a harder field day.

### FD22B - 8th Street / Hulls / Crestline

- Decision: `HOLD_PROVEN_BUNDLE_PRESSURE`
- Candidate: `crestline-trail`
- Start: Hulls Gulch
- Official segments: 1535, 1534, 1533, 1532
- Trails: Crestline Trail
- Cost: 1.82 official mi / 4.46 on-foot mi / 2.45x / 2.64 non-credit mi / 0.06 declared-repeat mi / p75 104 min / p90 118 min
- Closed warning codes: `same_trailhead_bundle_candidate`
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked by merging same-trailhead Hulls cards and by substituting the old 8th/Crestline superset. The superset is worse, and same-trailhead rows are scheduling opportunities, not certified lower-footmile replacements.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held despite same-trailhead bundle pressure because bundling is not proven to reduce runnable cost.
- What would disprove later: A certified same-trailhead Hulls or 8th Street bundle that proves lower total p75/on-foot burden without turning a short outing into a harder field day.

### 16A-2 - Dry Creek / Sweet Connie / Package 16

- Decision: `HOLD_PROVEN_HIGH_COST`
- Candidate: `manual-16a-2`
- Start: Dry Creek / Sweet Connie roadside parking
- Official segments: 1653
- Trails: Sheep Camp Trail
- Cost: 0.77 official mi / 3.31 on-foot mi / 4.30x / 2.54 non-credit mi / 0.66 declared-repeat mi / p75 106 min / p90 119 min
- Closed warning codes: `high_on_foot_to_official_ratio`, `same_trailhead_bundle_candidate`
- Same-credit/start attack: Same-credit alternatives are non-dominant in the current pack; best listed savings 0.01 mi / 0 min.
- Bundle/boundary/global attack: Attacked by reviving the old Hawkins/Sweet placeholder, merging same-start Dry/Sweet cards, and searching for generated exact replacements. The accepted split removes the 36.48-mile placeholder and no lower-footmile field-ready replacement is proven.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held despite high human-footmile pressure because current proofs do not show a safer lower-cost replacement.
- What would disprove later: A field-certified Dry/Sweet/Sweet Connie split or lower access anchor that preserves the exact official work and beats the accepted split by at least 0.25 miles or 10 p75 minutes without parking or cue risk.

### FD21B - Table Rock / Old Pen / Castle Rock

- Decision: `HOLD_PROVEN_CURRENT`
- Candidate: `combo-table-rock-trail-quarry-trail-castle-rock-shoshone-paiute`
- Start: Old Pen
- Official segments: 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1605, 1606, 1607, 1608, 1609, 1659, 1658
- Trails: Table Rock Trail, Quarry Trail - Castle Rock, Shoshone-Paiute
- Cost: 2.34 official mi / 3.80 on-foot mi / 1.62x / 1.46 non-credit mi / 0.85 declared-repeat mi / p75 115 min / p90 129 min
- Closed warning codes: none
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked by restoring the old aggregate, trying shorter high-overlap candidates, and checking supersets. Shorter candidates miss official work and supersets/aggregates are not dominant.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held as current route; no material disproof pressure remains in current audits.
- What would disprove later: A certified, legal, same-credit or same-day replacement with current parking/access evidence, p75/DEM effort, complete GPX/cues, and at least 0.25 miles or 10 p75 minutes saved.

### FD07A - Bogus Basin

- Decision: `HOLD_CONDITION_GATED`
- Candidate: `sunshine-xc`
- Start: Simplot Lodge Parking Area
- Official segments: 1713
- Trails: Sunshine XC
- Cost: 0.87 official mi / 2.15 on-foot mi / 2.47x / 1.28 non-credit mi / 0.02 declared-repeat mi / p75 119 min / p90 134 min
- Closed warning codes: `same_trailhead_bundle_candidate`
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked by merging Simplot/Pioneer cards, replacing small-credit out-and-backs, and recombining Bogus day work. High ratios are explained by lodge parking and official segments away from starts; no dominant replacement exists, but day-of road/trail/closure checks remain mandatory.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held against known route dominance, with live condition/access checks required near field use.
- What would disprove later: A certified Bogus-specific route with legal parking, current access, p75/DEM effort, complete GPX/cues, and same official credit that materially reduces on-foot or p75 cost; or a current closure/access finding that invalidates the lodge start.

### FD08A - Cartwright / Polecat / Peggy

- Decision: `HOLD_PROVEN_BUNDLE_PRESSURE`
- Candidate: `cartwright-ridge`
- Start: Cartwright
- Official segments: 1509, 1508
- Trails: Cartwright Ridge
- Cost: 1.76 official mi / 4.39 on-foot mi / 2.49x / 2.63 non-credit mi / 0.01 declared-repeat mi / p75 125 min / p90 142 min
- Closed warning codes: `same_trailhead_bundle_candidate`
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked by bundling Cartwright cards and recombining Polecat/Peggy boundaries. The boundary challenge and repeat audit found no dominant generated combo with p75, DEM effort, and field-ready continuity.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held despite same-trailhead bundle pressure because bundling is not proven to reduce runnable cost.
- What would disprove later: A certified Cartwright/Polecat/Peggy bundle that reduces car-to-car p75 or on-foot burden and keeps the cues usable at overlapping junctions.

### 7 - Westside / Seamans / Veterans

- Decision: `HOLD_PROVEN_CURRENT`
- Candidate: `block-westside_seaman_veterans`
- Start: Seamans Gulch
- Official segments: 1644, 1645, 1646, 1647, 1701, 1702
- Trails: Seaman Gulch Trail, Wild Phlox Trail
- Cost: 2.25 official mi / 3.77 on-foot mi / 1.68x / 1.52 non-credit mi / 0.27 declared-repeat mi / p75 127 min / p90 143 min
- Closed warning codes: none
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked as a same-corridor westside merge. The current routes are below hard efficiency pressure and no generated replacement beats the current packet under p75/DEM/coverage gates.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held as current route; no material disproof pressure remains in current audits.
- What would disprove later: A certified, legal, same-credit or same-day replacement with current parking/access evidence, p75/DEM effort, complete GPX/cues, and at least 0.25 miles or 10 p75 minutes saved.

### FD08B - Cartwright / Polecat / Peggy

- Decision: `HOLD_PROVEN_BUNDLE_PRESSURE`
- Candidate: `cartwright-connector`
- Start: Cartwright
- Official segments: 1709
- Trails: Cartwright Connector
- Cost: 1.70 official mi / 4.65 on-foot mi / 2.74x / 2.95 non-credit mi / 0.01 declared-repeat mi / p75 127 min / p90 144 min
- Closed warning codes: `same_trailhead_bundle_candidate`
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked by bundling Cartwright cards and recombining Polecat/Peggy boundaries. The boundary challenge and repeat audit found no dominant generated combo with p75, DEM effort, and field-ready continuity.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held despite same-trailhead bundle pressure because bundling is not proven to reduce runnable cost.
- What would disprove later: A certified Cartwright/Polecat/Peggy bundle that reduces car-to-car p75 or on-foot burden and keeps the cues usable at overlapping junctions.

### 16B - Stack Rock separated clean route

- Decision: `HOLD_PROVEN_CURRENT`
- Candidate: `stack-rock-connector`
- Start: Freddy's Stack Rock
- Official segments: 1664, 1663
- Trails: Stack Rock Connector
- Cost: 3.50 official mi / 4.39 on-foot mi / 1.25x / 0.89 non-credit mi / 0.21 declared-repeat mi / p75 131 min / p90 147 min
- Closed warning codes: none
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked as a possible Package 16 merge. It is a clean Stack Rock Connector outing with no repeat-warning pressure and keeps Package 16 from absorbing unnecessary mountain access burden.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held as current route; no material disproof pressure remains in current audits.
- What would disprove later: A certified, legal, same-credit or same-day replacement with current parking/access evidence, p75/DEM effort, complete GPX/cues, and at least 0.25 miles or 10 p75 minutes saved.

### FD05A - 8th Street / Hulls / Crestline

- Decision: `HOLD_PROVEN_BUNDLE_PRESSURE`
- Candidate: `hulls-gulch-interpretive`
- Start: 8th Street ATV Parking Area
- Official segments: 1751, 1729, 1727, 1725, 1726, 1728, 1730
- Trails: Hull's Gulch Interpretive
- Cost: 5.07 official mi / 5.66 on-foot mi / 1.12x / 0.59 non-credit mi / 0.01 declared-repeat mi / p75 133 min / p90 154 min
- Closed warning codes: `same_trailhead_bundle_candidate`
- Same-credit/start attack: Same-credit alternatives are non-dominant in the current pack; best listed savings 0.00 mi / 0 min.
- Bundle/boundary/global attack: Attacked by merging same-trailhead Hulls cards and by substituting the old 8th/Crestline superset. The superset is worse, and same-trailhead rows are scheduling opportunities, not certified lower-footmile replacements.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held despite same-trailhead bundle pressure because bundling is not proven to reduce runnable cost.
- What would disprove later: A certified same-trailhead Hulls or 8th Street bundle that proves lower total p75/on-foot burden without turning a short outing into a harder field day.

### 15B - Dry Creek / Sweet Connie / Package 16

- Decision: `HOLD_PROVEN_BUNDLE_PRESSURE`
- Candidate: `combo-landslide-red-tail-trail`
- Start: Dry Creek Parking Area/Trailhead
- Official segments: 1624, 1623, 1622, 1621, 1620, 1619, 1618, 1584
- Trails: Red Tail Trail, Landslide
- Cost: 4.02 official mi / 4.87 on-foot mi / 1.21x / 0.85 non-credit mi / 0.01 declared-repeat mi / p75 148 min / p90 166 min
- Closed warning codes: `same_trailhead_bundle_candidate`
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked by reviving the old Hawkins/Sweet placeholder, merging same-start Dry/Sweet cards, and searching for generated exact replacements. The accepted split removes the 36.48-mile placeholder and no lower-footmile field-ready replacement is proven.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held despite same-trailhead bundle pressure because bundling is not proven to reduce runnable cost.
- What would disprove later: A field-certified Dry/Sweet/Sweet Connie split or lower access anchor that preserves the exact official work and beats the accepted split by at least 0.25 miles or 10 p75 minutes without parking or cue risk.

### 11 - Dry Creek / Sweet Connie / Package 16

- Decision: `HOLD_PROVEN_CURRENT`
- Candidate: `block-hawkins`
- Start: Hawkins Range Reserve
- Official segments: 1571, 1572, 1573
- Trails: Hawkins
- Cost: 5.63 official mi / 5.73 on-foot mi / 1.02x / 0.10 non-credit mi / 0.00 declared-repeat mi / p75 149 min / p90 167 min
- Closed warning codes: none
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked by reviving the old Hawkins/Sweet placeholder, merging same-start Dry/Sweet cards, and searching for generated exact replacements. The accepted split removes the 36.48-mile placeholder and no lower-footmile field-ready replacement is proven.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held as current route; no material disproof pressure remains in current audits.
- What would disprove later: A field-certified Dry/Sweet/Sweet Connie split or lower access anchor that preserves the exact official work and beats the accepted split by at least 0.25 miles or 10 p75 minutes without parking or cue risk.

### 10B - Dry Creek / Sweet Connie / Package 16

- Decision: `HOLD_PROVEN_BUNDLE_PRESSURE`
- Candidate: `manual-10b`
- Start: Dry Creek Parking Area/Trailhead
- Official segments: 1497, 1538, 1537, 1536
- Trails: Bitterbrush Trail, Currant Creek
- Cost: 2.45 official mi / 5.43 on-foot mi / 2.22x / 2.98 non-credit mi / 0.32 declared-repeat mi / p75 152 min / p90 171 min
- Closed warning codes: `same_trailhead_bundle_candidate`
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked by reviving the old Hawkins/Sweet placeholder, merging same-start Dry/Sweet cards, and searching for generated exact replacements. The accepted split removes the 36.48-mile placeholder and no lower-footmile field-ready replacement is proven.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held despite same-trailhead bundle pressure because bundling is not proven to reduce runnable cost.
- What would disprove later: A field-certified Dry/Sweet/Sweet Connie split or lower access anchor that preserves the exact official work and beats the accepted split by at least 0.25 miles or 10 p75 minutes without parking or cue risk.

### FD03A - Chukar Butte repaired singleton

- Decision: `HOLD_PROVISIONAL_FIELD_WALKTHROUGH`
- Candidate: `accepted-replacement-fd03a-chukar-butte-strava-anchor-19`
- Start: Chukar Butte private-derived parking anchor
- Official segments: 1521, 1520, 1519
- Trails: Chukar Butte Trail
- Cost: 4.83 official mi / 5.34 on-foot mi / 1.11x / 0.51 non-credit mi / 0.02 declared-repeat mi / p75 155 min / p90 174 min
- Closed warning codes: none
- Same-credit/start attack: Same-credit alternatives are non-dominant in the current pack; best listed savings 0.03 mi / 4 min.
- Bundle/boundary/global attack: Attacked with the stale public-start card and a private-derived closer anchor. The current route already uses the accepted Chukar Butte repair; remaining risk is field-walkthrough promotion, not known dominance.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held against known dominance, but not promoted beyond its separate field-walkthrough gate.
- What would disprove later: A certified, legal, same-credit or same-day replacement with current parking/access evidence, p75/DEM effort, complete GPX/cues, and at least 0.25 miles or 10 p75 minutes saved.

### FD07B - Bogus Basin

- Decision: `HOLD_CONDITION_GATED`
- Candidate: `deer-point-trail`
- Start: Simplot Lodge Parking Area
- Official segments: 1540
- Trails: Deer Point Trail
- Cost: 1.14 official mi / 3.97 on-foot mi / 3.48x / 2.83 non-credit mi / 0.01 declared-repeat mi / p75 155 min / p90 174 min
- Closed warning codes: `high_on_foot_to_official_ratio`, `same_trailhead_bundle_candidate`
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked by merging Simplot/Pioneer cards, replacing small-credit out-and-backs, and recombining Bogus day work. High ratios are explained by lodge parking and official segments away from starts; no dominant replacement exists, but day-of road/trail/closure checks remain mandatory.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held against known route dominance, with live condition/access checks required near field use.
- What would disprove later: A certified Bogus-specific route with legal parking, current access, p75/DEM effort, complete GPX/cues, and same official credit that materially reduces on-foot or p75 cost; or a current closure/access finding that invalidates the lodge start.

### FD21A - Harris Ranch / Peace Valley singleton

- Decision: `HOLD_PROVEN_CURRENT`
- Candidate: `peace-valley-overlook-harris-ridge-trail`
- Start: Homestead
- Official segments: 1722, 1723, 1724
- Trails: Peace Valley Overlook, Harris Ridge Trail
- Cost: 2.26 official mi / 5.21 on-foot mi / 2.31x / 2.95 non-credit mi / 0.26 declared-repeat mi / p75 159 min / p90 179 min
- Closed warning codes: none
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked as an isolated high-access route. The exact-credit gate and current candidate universe show no accepted lower-footmile replacement with field-ready p75 and effort.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held as current route; no material disproof pressure remains in current audits.
- What would disprove later: A certified, legal, same-credit or same-day replacement with current parking/access evidence, p75/DEM effort, complete GPX/cues, and at least 0.25 miles or 10 p75 minutes saved.

### FD25B - Bogus Basin

- Decision: `HOLD_CONDITION_GATED`
- Candidate: `the-face-trail`
- Start: Pioneer Lodge Parking Area
- Official segments: 1680
- Trails: The Face Trail
- Cost: 1.15 official mi / 4.30 on-foot mi / 3.74x / 3.15 non-credit mi / 0.02 declared-repeat mi / p75 160 min / p90 180 min
- Closed warning codes: `high_on_foot_to_official_ratio`, `same_trailhead_bundle_candidate`
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked by merging Simplot/Pioneer cards, replacing small-credit out-and-backs, and recombining Bogus day work. High ratios are explained by lodge parking and official segments away from starts; no dominant replacement exists, but day-of road/trail/closure checks remain mandatory.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held against known route dominance, with live condition/access checks required near field use.
- What would disprove later: A certified Bogus-specific route with legal parking, current access, p75/DEM effort, complete GPX/cues, and same official credit that materially reduces on-foot or p75 cost; or a current closure/access finding that invalidates the lodge start.

### FD01A - Table Rock / Old Pen / Castle Rock

- Decision: `HOLD_PROVEN_HIGH_COST`
- Candidate: `combo-rock-island-table-rock-quarry-trail-rock-garden-tram-trail`
- Start: Warm Springs Golf Course
- Official segments: 1635, 1636, 1637, 1638, 1639, 1640, 1641, 1642, 1634, 1633, 1632, 1686, 1668, 1669, 1670
- Trails: Rock Island, Rock Garden, Tram Trail, Table Rock Quarry Trail
- Cost: 4.26 official mi / 6.97 on-foot mi / 1.64x / 2.71 non-credit mi / 2.66 declared-repeat mi / p75 170 min / p90 191 min
- Closed warning codes: `high_declared_repeat_miles`
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked by restoring the old aggregate, trying shorter high-overlap candidates, and checking supersets. Shorter candidates miss official work and supersets/aggregates are not dominant.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held despite high human-footmile pressure because current proofs do not show a safer lower-cost replacement.
- What would disprove later: A certified, legal, same-credit or same-day replacement with current parking/access evidence, p75/DEM effort, complete GPX/cues, and at least 0.25 miles or 10 p75 minutes saved.

### FD25A - Bogus Basin

- Decision: `HOLD_CONDITION_GATED`
- Candidate: `elk-meadows-trail`
- Start: Simplot Lodge Parking Area
- Official segments: 1553, 1554
- Trails: Elk Meadows Trail
- Cost: 1.49 official mi / 4.90 on-foot mi / 3.29x / 3.41 non-credit mi / 0.03 declared-repeat mi / p75 171 min / p90 193 min
- Closed warning codes: `high_on_foot_to_official_ratio`, `same_trailhead_bundle_candidate`
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked by merging Simplot/Pioneer cards, replacing small-credit out-and-backs, and recombining Bogus day work. High ratios are explained by lodge parking and official segments away from starts; no dominant replacement exists, but day-of road/trail/closure checks remain mandatory.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held against known route dominance, with live condition/access checks required near field use.
- What would disprove later: A certified Bogus-specific route with legal parking, current access, p75/DEM effort, complete GPX/cues, and same official credit that materially reduces on-foot or p75 cost; or a current closure/access finding that invalidates the lodge start.

### 9 - Westside / Seamans / Veterans

- Decision: `HOLD_PROVEN_CURRENT`
- Candidate: `block-eagle_bike_park_red_tail`
- Start: Veterans
- Official segments: 1691, 1692, 1693, 1496, 1614, 1613, 1612, 1611, 1539, 1752, 1753, 1754, 1625
- Trails: Veterans, Big Springs, Rabbit Run, D's Chaos, REI Connection
- Cost: 4.68 official mi / 5.78 on-foot mi / 1.24x / 1.10 non-credit mi / 1.49 declared-repeat mi / p75 180 min / p90 202 min
- Closed warning codes: none
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked as a same-corridor westside merge. The current routes are below hard efficiency pressure and no generated replacement beats the current packet under p75/DEM/coverage gates.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held as current route; no material disproof pressure remains in current audits.
- What would disprove later: A certified, legal, same-credit or same-day replacement with current parking/access evidence, p75/DEM effort, complete GPX/cues, and at least 0.25 miles or 10 p75 minutes saved.

### 19 - Cervidae isolated singleton

- Decision: `HOLD_PROVEN_HIGH_COST`
- Candidate: `block-cervidae_peak`
- Start: Cervidae / Arrow Rock Road OSM Parking
- Official segments: 1731
- Trails: Cervidae Peak
- Cost: 2.24 official mi / 4.51 on-foot mi / 2.01x / 2.27 non-credit mi / 2.24 declared-repeat mi / p75 181 min / p90 203 min
- Closed warning codes: `high_declared_repeat_miles`
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked as a high-repeat isolated peak card. Cervidae has no proven better legal start or same-credit replacement in the current candidate universe, and the repeat burden is declared/priced.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held despite high human-footmile pressure because current proofs do not show a safer lower-cost replacement.
- What would disprove later: A certified, legal, same-credit or same-day replacement with current parking/access evidence, p75/DEM effort, complete GPX/cues, and at least 0.25 miles or 10 p75 minutes saved.

### FD04A - Freestone / Military / Three Bears

- Decision: `HOLD_PROVEN_HIGH_COST`
- Candidate: `combo-two-point-femrites-patrol-shanes-connector`
- Start: Freestone Creek
- Official segments: 1748, 1652, 1558, 1649, 1650, 1651
- Trails: Two Point, Shane's Connector, Femrite's Patrol, Shane's Trail
- Cost: 3.54 official mi / 9.55 on-foot mi / 2.70x / 6.01 non-credit mi / 1.64 declared-repeat mi / p75 204 min / p90 229 min
- Closed warning codes: `high_non_credit_miles`, `same_trailhead_bundle_candidate`
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked by restoring the old Freestone/Three Bears/Curlew aggregate, shorter high-overlap variants, and boundary recombination. The old aggregate is worse; shorter candidates miss official segments; no dominant boundary/global replacement is proven.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held despite high human-footmile pressure because current proofs do not show a safer lower-cost replacement.
- What would disprove later: A continuous certified Freestone/Military/Three Bears recombination that covers the same official set, keeps p75/DEM evidence, and reduces human footmiles without introducing hidden repeats or latent credit.

### FD06A - Freestone / Military / Three Bears

- Decision: `HOLD_PROVEN_CURRENT`
- Candidate: `fat-tire-traverse-curlew-connection`
- Start: Lower Interpretive
- Official segments: 1555, 1711, 1710
- Trails: Fat Tire Traverse, Curlew Connection
- Cost: 4.09 official mi / 8.59 on-foot mi / 2.10x / 4.50 non-credit mi / 0.26 declared-repeat mi / p75 220 min / p90 247 min
- Closed warning codes: none
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked by restoring the old Freestone/Three Bears/Curlew aggregate, shorter high-overlap variants, and boundary recombination. The old aggregate is worse; shorter candidates miss official segments; no dominant boundary/global replacement is proven.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held as current route; no material disproof pressure remains in current audits.
- What would disprove later: A continuous certified Freestone/Military/Three Bears recombination that covers the same official set, keeps p75/DEM evidence, and reduces human footmiles without introducing hidden repeats or latent credit.

### 15A-1 - Dry Creek / Sweet Connie / Package 16

- Decision: `HOLD_PROVEN_BUNDLE_PRESSURE`
- Candidate: `multi-start-15a-15a-ms-03-1-dry-creek-trail`
- Start: Dry Creek / Sweet Connie roadside parking
- Official segments: 1542, 1543, 1544, 1545, 1546, 1656
- Trails: Dry Creek Trail, Shingle Creek Trail
- Cost: 11.73 official mi / 11.89 on-foot mi / 1.01x / 0.16 non-credit mi / 0.02 declared-repeat mi / p75 229 min / p90 257 min
- Closed warning codes: `same_trailhead_bundle_candidate`
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked by reviving the old Hawkins/Sweet placeholder, merging same-start Dry/Sweet cards, and searching for generated exact replacements. The accepted split removes the 36.48-mile placeholder and no lower-footmile field-ready replacement is proven.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held despite same-trailhead bundle pressure because bundling is not proven to reduce runnable cost.
- What would disprove later: A field-certified Dry/Sweet/Sweet Connie split or lower access anchor that preserves the exact official work and beats the accepted split by at least 0.25 miles or 10 p75 minutes without parking or cue risk.

### 14 - Watchman / Five Mile / Rocky Canyon

- Decision: `HOLD_PROVEN_CURRENT`
- Candidate: `block-watchman_five_mile_rocky`
- Start: Orchard Gulch
- Official segments: 1595, 1560, 1561, 1562, 1695, 1694
- Trails: Orchard Gulch Trail, Five Mile Gulch Trail, Watchman Trail
- Cost: 8.45 official mi / 10.74 on-foot mi / 1.27x / 2.29 non-credit mi / 0.22 declared-repeat mi / p75 242 min / p90 272 min
- Closed warning codes: none
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked as a possible boundary merge with adjacent north/east work. The route is below high-ratio pressure and current boundary/global challenges do not displace it.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held as current route; no material disproof pressure remains in current audits.
- What would disprove later: A certified, legal, same-credit or same-day replacement with current parking/access evidence, p75/DEM effort, complete GPX/cues, and at least 0.25 miles or 10 p75 minutes saved.

### FD12A - West Climb / Harrison / Full Sail

- Decision: `HOLD_PROVEN_HIGH_COST`
- Candidate: `combo-who-now-loop-trail-harrison-ridge-harrison-hollow-kempers-ridge-trail-full-sail-trail-buena-vista-trail-bob-smylie-hippie-shake-trail`
- Start: West Climb
- Official segments: 1697, 1698, 1699, 1700, 1716, 1717, 1714, 1715, 1579, 1581, 1582, 1565, 1566, 1504, 1505, 1506, 1507, 1755, 1718, 1719, 1578
- Trails: Who Now Loop Trail, Harrison Ridge, Harrison Hollow, Kemper's Ridge Trail, Full Sail Trail, Buena Vista Trail, Bob Smylie, Hippie Shake Trail
- Cost: 7.85 official mi / 10.61 on-foot mi / 1.35x / 2.76 non-credit mi / 4.69 declared-repeat mi / p75 242 min / p90 272 min
- Closed warning codes: `high_declared_repeat_miles`
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked because declared repeat looks suspicious. The route is not high-ratio, the official repeat is declared and priced, and no boundary/global replacement beats it.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held despite high human-footmile pressure because current proofs do not show a safer lower-cost replacement.
- What would disprove later: A certified, legal, same-credit or same-day replacement with current parking/access evidence, p75/DEM effort, complete GPX/cues, and at least 0.25 miles or 10 p75 minutes saved.

### 16A-1 - Dry Creek / Sweet Connie / Package 16

- Decision: `HOLD_PROVEN_HIGH_COST`
- Candidate: `manual-16a-1`
- Start: Dry Creek / Sweet Connie roadside parking
- Official segments: 1665, 1666, 1667
- Trails: Sweet Connie Trail
- Cost: 6.09 official mi / 12.20 on-foot mi / 2.00x / 6.11 non-credit mi / 6.09 declared-repeat mi / p75 249 min / p90 279 min
- Closed warning codes: `high_declared_repeat_miles`, `high_non_credit_miles`, `same_trailhead_bundle_candidate`
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked by reviving the old Hawkins/Sweet placeholder, merging same-start Dry/Sweet cards, and searching for generated exact replacements. The accepted split removes the 36.48-mile placeholder and no lower-footmile field-ready replacement is proven.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held despite high human-footmile pressure because current proofs do not show a safer lower-cost replacement.
- What would disprove later: A field-certified Dry/Sweet/Sweet Connie split or lower access anchor that preserves the exact official work and beats the accepted split by at least 0.25 miles or 10 p75 minutes without parking or cue risk.

### 3 - Freestone / Military / Three Bears

- Decision: `HOLD_PROVEN_HIGH_COST`
- Candidate: `block-military_core`
- Start: Freestone Creek
- Official segments: 1590, 1594, 1593, 1592, 1591, 1515, 1514, 1513, 1512, 1511, 1510, 1720, 1629, 1631, 1630, 1627, 1628, 1529, 1530, 1531, 1522, 1551, 1550, 1549, 1548, 1552, 1574, 1575
- Trails: Military Reserve Connection, Mountain Cove, Central Ridge Trail, Central Ridge Spur, Ridge Crest, Cottonwood Creek Trail, Connection (Eagle Ridge), Eagle Ridge Trail, Elephant Rock Loop, Heroes Trail
- Cost: 8.31 official mi / 12.13 on-foot mi / 1.46x / 3.82 non-credit mi / 6.51 declared-repeat mi / p75 250 min / p90 280 min
- Closed warning codes: `high_declared_repeat_miles`, `same_trailhead_bundle_candidate`
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked by restoring the old Freestone/Three Bears/Curlew aggregate, shorter high-overlap variants, and boundary recombination. The old aggregate is worse; shorter candidates miss official segments; no dominant boundary/global replacement is proven.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held despite high human-footmile pressure because current proofs do not show a safer lower-cost replacement.
- What would disprove later: A continuous certified Freestone/Military/Three Bears recombination that covers the same official set, keeps p75/DEM evidence, and reduces human footmiles without introducing hidden repeats or latent credit.

### FD18A - Cartwright / Polecat / Peggy

- Decision: `HOLD_PROVEN_HIGH_COST`
- Candidate: `polecat-loop-peggys-trail`
- Start: Cartwright
- Official segments: 1599, 1602, 1600, 1598, 1601, 1603, 1604, 1597
- Trails: Polecat Loop, Peggy's Trail
- Cost: 10.19 official mi / 13.32 on-foot mi / 1.31x / 3.13 non-credit mi / 2.34 declared-repeat mi / p75 254 min / p90 286 min
- Closed warning codes: `high_declared_repeat_miles`, `same_trailhead_bundle_candidate`
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked by bundling Cartwright cards and recombining Polecat/Peggy boundaries. The boundary challenge and repeat audit found no dominant generated combo with p75, DEM effort, and field-ready continuity.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held despite high human-footmile pressure because current proofs do not show a safer lower-cost replacement.
- What would disprove later: A certified Cartwright/Polecat/Peggy bundle that reduces car-to-car p75 or on-foot burden and keeps the cues usable at overlapping junctions.

### FD20A - Freestone / Military / Three Bears

- Decision: `HOLD_PROVEN_HIGH_COST`
- Candidate: `three-bears-trail-freestone-ridge`
- Start: Freestone Creek
- Official segments: 1681, 1682, 1683, 1684, 1685, 1563, 1564
- Trails: Three Bears Trail, Freestone Ridge
- Cost: 6.72 official mi / 13.10 on-foot mi / 1.95x / 6.38 non-credit mi / 1.44 declared-repeat mi / p75 255 min / p90 287 min
- Closed warning codes: `high_non_credit_miles`, `same_trailhead_bundle_candidate`
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked by restoring the old Freestone/Three Bears/Curlew aggregate, shorter high-overlap variants, and boundary recombination. The old aggregate is worse; shorter candidates miss official segments; no dominant boundary/global replacement is proven.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held despite high human-footmile pressure because current proofs do not show a safer lower-cost replacement.
- What would disprove later: A continuous certified Freestone/Military/Three Bears recombination that covers the same official set, keeps p75/DEM evidence, and reduces human footmiles without introducing hidden repeats or latent credit.

### 12 - 8th Street / Hulls / Crestline

- Decision: `HOLD_PROVEN_BUNDLE_PRESSURE`
- Candidate: `block-upper_8th_corrals_sidewinder`
- Start: 8th Street ATV Parking Area
- Official segments: 1483, 1484, 1485, 1486, 1660, 1524, 1525, 1526, 1527, 1528, 1576, 1577
- Trails: 8th Street Motorcycle Trail, Sidewinder Trail, Corrals Trail, Highlands Trail
- Cost: 9.49 official mi / 12.86 on-foot mi / 1.36x / 3.37 non-credit mi / 1.85 declared-repeat mi / p75 262 min / p90 294 min
- Closed warning codes: `same_trailhead_bundle_candidate`
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked by merging same-trailhead Hulls cards and by substituting the old 8th/Crestline superset. The superset is worse, and same-trailhead rows are scheduling opportunities, not certified lower-footmile replacements.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held despite same-trailhead bundle pressure because bundling is not proven to reduce runnable cost.
- What would disprove later: A certified same-trailhead Hulls or 8th Street bundle that proves lower total p75/on-foot burden without turning a short outing into a harder field day.

### FD26A - Bogus Basin

- Decision: `HOLD_CONDITION_GATED`
- Candidate: `around-the-mountain-trail`
- Start: Simplot Lodge Parking Area
- Official segments: 1488, 1489, 1490, 1491, 1492, 1493, 1750
- Trails: Around the Mountain Trail
- Cost: 6.64 official mi / 10.17 on-foot mi / 1.53x / 3.53 non-credit mi / 2.15 declared-repeat mi / p75 279 min / p90 313 min
- Closed warning codes: `high_declared_repeat_miles`, `same_trailhead_bundle_candidate`
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked by merging Simplot/Pioneer cards, replacing small-credit out-and-backs, and recombining Bogus day work. High ratios are explained by lodge parking and official segments away from starts; no dominant replacement exists, but day-of road/trail/closure checks remain mandatory.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held against known route dominance, with live condition/access checks required near field use.
- What would disprove later: A certified Bogus-specific route with legal parking, current access, p75/DEM effort, complete GPX/cues, and same official credit that materially reduces on-foot or p75 cost; or a current closure/access finding that invalidates the lodge start.

### H1 - Avimor / Harlow Spring

- Decision: `HOLD_PUBLIC_ACCESS_RECHECK`
- Candidate: `H1-avimor-native-harlow-spring-loop`
- Start: Avimor Spring Valley Creek parking
- Official segments: 1626, 1657, 1661, 1662, 1687, 1688, 1689, 1696, 1704, 1705, 1706, 1707, 1708
- Trails: Twisted Spring, Ricochet, Shooting Range, Whistling Pig, Spring Creek, Harlow's Hollows, Harlow's Hollows Connector
- Cost: 7.30 official mi / 9.64 on-foot mi / 1.32x / 2.34 non-credit mi / 0.68 declared-repeat mi / p75 289 min / p90 324 min
- Closed warning codes: none
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked as a late-added H1 correction, artifact-drift risk, and public-access proof gap. Current field packet/map data agree on H1 and no route-review dominance remains, but the Avimor owner page says trails are open for Avimor residents while the current start proof rests on OSM plus AllTrails. That is enough to keep the route active as the best known candidate, but not enough to call the access layer accepted.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort still prove the generated outing shape. They do not prove public participant access at Avimor Spring Valley Creek parking.
- Result: Held against known dominance and optimizer attacks, but downgraded from accepted-current to public-access recheck. Do not treat H1 as public-ready until BTC organizer, Avimor, or current field signage confirms public challenge access from this start.
- What would disprove later: If Avimor or BTC organizer confirmation does not permit public participant use of the Spring Valley Creek start, redesign H1 from a certifiable public anchor or keep it waived/gated. If confirmation or signage proves public access, restore accepted-current status after rerunning the route proof.

### 18 - Bogus Basin

- Decision: `HOLD_CONDITION_GATED`
- Candidate: `block-bogus_mores_lodge_tempest`
- Start: Pioneer Lodge Parking Area
- Official segments: 1703, 1501, 1503, 1502, 1655, 1678, 1679, 1721, 1732, 1733, 1734, 1735, 1736
- Trails: Brewer's Byway Extension, Brewers Byway, Shindig, Tempest Trail, Lodge Trail, Mores Mtn Interpretive
- Cost: 5.08 official mi / 11.25 on-foot mi / 2.21x / 6.17 non-credit mi / 0.86 declared-repeat mi / p75 320 min / p90 359 min
- Closed warning codes: `high_non_credit_miles`, `same_trailhead_bundle_candidate`
- Same-credit/start attack: No same-credit accepted alternative is recorded.
- Bundle/boundary/global attack: Attacked by merging Simplot/Pioneer cards, replacing small-credit out-and-backs, and recombining Bogus day work. High ratios are explained by lodge parking and official segments away from starts; no dominant replacement exists, but day-of road/trail/closure checks remain mandatory.
- Human-validity attack: Current field packet, route review pack, repeat audit, efficiency audit, p75 timing, and DEM effort are the proof layers; this does not replace day-of field judgment.
- Result: Held against known route dominance, with live condition/access checks required near field use.
- What would disprove later: A certified Bogus-specific route with legal parking, current access, p75/DEM effort, complete GPX/cues, and same official credit that materially reduces on-foot or p75 cost; or a current closure/access finding that invalidates the lodge start.

## Limits
- Does not prove a brand-new legal anchor or hand-drawn GPX cannot beat the current packet later.
- Does not replace current closure, mud, heat, water, snow, road, or parking checks before field execution.
- Does not remove provisional field-walkthrough status from accepted replacement routes.
- Does not include raw private coordinates, raw activity ids, or private GPS traces.

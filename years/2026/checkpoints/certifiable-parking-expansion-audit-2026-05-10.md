# Certifiable Parking Expansion Audit - 2026-05-10

## Question

When a route optimization depends on a questionable nearest-road start, should
the planner stop at that road pin, or search outward for the nearest
certifiable parking anchor and price the connector?

## Result

Search outward. The route-mapping heuristic is valid and should become part of
candidate generation:

1. Identify the nearest graph-valid road or trail tie-in.
2. Independently search for the nearest certifiable parking surface within a
   reasonable connector budget.
3. Recompute connector mileage, on-foot mileage, p75, p90, and cue complexity
   from the certifiable anchor before rejecting or promoting the route.

This matters because the best human-executable route may start at a public
park, official lot, amenity lot, or published event/start location, even if
that adds a neighborhood connector to the official trail tie-in.

## 10A Case Study

The route-pain index identified `10A` as the strongest remaining current
route-card optimization target. The paper `10A-MS-08` split saved 3.38 on-foot
miles and 43 p75 minutes, but the access verification blocked both of its
informal start assumptions:

- North Burnt Car Place remained a physically plausible but uncertified
  residential road-parking probe.
- Harlow's / Hidden Springs west access probe was not a certified car start.

The contrarian reframing was to stop trying to prove those nearest road pins
and instead test Foothills Heritage Park / Avimor Spring Valley Creek parking
as the certifiable dual anchor, absorbing the neighborhood connector to the
Harlow/Burnt Car tie-in.

## Computed Reframe

Inputs:

- Current phone-packet route card: `docs/field-packet/field-tool-data.json`.
- Multi-start paper alternative:
  `years/2026/checkpoints/multi-start-alternative-audit-2026-05-08.json`.
- Access blocker record:
  `years/2026/checkpoints/10a-ms-08-access-verification-2026-05-10.md`.
- Source-verified Avimor anchors:
  `years/2026/inputs/personal/2026-harlow-spring-manual-route-design-v1.json`
  and `years/2026/checkpoints/parking-access-verification-2026-05-06.md`.

| Variant | Anchor | Status | On-foot mi | P75 | P90 | Finding |
|---|---|---:|---:|---:|---:|---|
| Current active `10A` | Harlow's / Hidden Springs west access probe | active but parked-start proof weak | 13.62 | 360 | 404 | Baseline to beat; start itself still needs certification. |
| Paper `10A-MS-08` | North Burnt + Harlow west road/probe starts | blocked | 10.24 | 317 | n/a | Best paper saving, but not legally executable today. |
| All `10A` from Spring Valley Creek / FHP | Avimor Spring Valley Creek parking | draft | 16.45 | 416 | 466 | Existing generator shape is worse than baseline. |
| All `10A` from Twisted Spring | Avimor Twisted Spring parking | draft | 16.91 | 421 | 472 | Worse than baseline. |
| All `10A` from Harlow west probe | Harlow's / Hidden Springs west access probe | draft | 15.15 | 387 | 434 | Worse than baseline and not certified. |
| `10A-MS-08` partition reanchored to Spring Valley Creek / FHP | Avimor Spring Valley Creek parking | mixed draft/graph-valid | 15.14 | 418 rough same-day | 469 rough same-day | Worse than baseline in the current planner. |

## Decision

Do not promote the simple Foothills Heritage Park / Spring Valley Creek collapse
as a `10A` replacement yet. The current route generator prices that shape worse
than the active `10A` card.

This does not disprove the heuristic. It proves the current generator is still
too blunt for this class of optimization. The high-value approach is a
certifiable-anchor expansion pass that can also accept an explicit tie-in
waypoint or connector corridor. For `10A`, the remaining salvage path is not
"replace the start with FHP" mechanically; it is "start at FHP, route to a
specific Harlow/Burnt Car tie-in, then hand-shape or generate the official loop
from that tie-in, and compare the resulting GPX/cues/p75/p90."

## Planner Implication

Add a reusable candidate-generation phase:

- For each high-pain route or blocked nearest-road alternative, generate a
  nearest-road candidate and a nearest-certifiable-parking candidate set.
- Keep parks, official lots, amenity parking, event meeting points, public
  community lots, and source-described route starts as candidate anchors.
- Allow a candidate to declare a connector tie-in waypoint when the anchor is
  not the natural official-segment endpoint.
- Compare legal executable cost, not only graph shortest distance.
- Keep the route parked-start gated until the same source artifact produces
  continuous GPX, cue text, p75/p90, segment coverage, ascent-direction
  evidence, and access certification.

## Next Work

1. Add certifiable-anchor expansion to the multi-start/access audit so the
   planner does not stop at residential road probes.
2. Add a waypoint-constrained connector corridor mode for manual route designs.
3. Generate an explicit FHP/Spring Valley Creek to Harlow/Burnt Car tie-in GPX
   and compare that hand-shaped route against current `10A`.
4. Only promote if the resulting route beats baseline after connector mileage,
   p75/p90, cue complexity, and parked-start certification.

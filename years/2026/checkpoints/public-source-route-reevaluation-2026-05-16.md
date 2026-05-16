# Public-Source Route Reevaluation

Date: 2026-05-16

Frame decision: `hold`.

Purpose: record current public-source findings that change route confidence after the all-route adversarial disproof pass. This checkpoint does not mutate the active field packet by itself. It tells the route proofs what can and cannot be treated as accepted.

## Summary

- Routes rechecked: 43
- Route-set mutation required now: no
- Routes downgraded: 0
- Routes downgraded by public source only before user confirmation: 1
- Existing condition gates reaffirmed: 6
- User-confirmed access: `H1`

The strongest public-source finding was not a same-credit dominance issue. It was an access-proof issue: H1 remains the best known route shape in the current generated packet, and the public source layer alone does not prove the Avimor Spring Valley Creek start. The user has now confirmed Avimor access, so the private field-use gate is resolved while the public-source ambiguity remains documented.

## Sources Checked

- [Avimor Trails and Outdoors](https://www.avimor.com/trails-and-outdoors): Avimor frames trail access as resident access and describes resident-only permits for some trail uses.
- [Bogus Basin Deer Point Stewardship Project 2026](https://bogusbasin.org/about-bogus/culture/deer-point-stewardship-project-2026/): confirms weekday road/trail closure windows through Friday, June 19, 2026.
- [Ridge to Rivers special management strategies](https://www.ridgetorivers.org/trail-news/ridge-to-rivers-adopts-management-strategies-from-pilot-trail-program/): confirms direction/separation rules for Lower Hulls, Polecat, Around the Mountain, and Bucktail-related pedestrian access.
- [BoiseTrails current trail-status page](https://boisetrails.com/): shows current May 2026 mud, snow, ice, and closure signals. These gate immediate field tests, but do not by themselves mutate the June/July route set.
- User-provided access confirmation: user confirmed Avimor access on 2026-05-16.

## Route Impacts

### H1 - Avimor / Harlow Spring

- Previous public-only proof status: `needs_public_access_confirmation`
- New proof status: `accepted_current_user_confirmed`
- New route decision: `HOLD_PROVEN_CURRENT`
- Candidate: `H1-avimor-native-harlow-spring-loop`
- Current start: Avimor Spring Valley Creek parking

Reason: the route proof relied on OSM plus AllTrails parking/start evidence. Avimor's current public owner page says trails are open for Avimor residents, which does not by itself prove public Boise Trails Challenge participant access. The user confirmed Avimor access, so the private field-use gate is now accepted as user-reviewed access.

What did not change: no known accepted same-credit anchor or optimizer replacement currently dominates the H1 route shape.

Required action: keep normal day-of signage and condition checks. If Avimor/BTC access rules change later, re-open this gate or redesign from a certifiable public anchor.

### Bogus Routes

- Route labels: `FD07A`, `FD07B`, `FD25B`, `FD25A`, `FD26A`, `18`
- Decision remains: `HOLD_CONDITION_GATED`

The Bogus source does not create a new route-shape replacement. It reinforces that June 18 and June 19, 2026 are not clean default field days for Bogus/Deer Point/Pat's/Bogus Basin Road access because the closure notice runs through Friday, June 19, 2026 with weekday closure windows.

Required action: schedule after the closure window when possible and check Bogus/Ridge to Rivers day-of conditions before field use.

### Direction And Use-Managed Trails

- Route labels: `FD19B`, `FD18A`, `FD26A`, `FD04A`
- Decision: no route mutation from public sources alone

Ridge to Rivers management rules reinforce existing date, direction, and use-separation checks for Lower Hulls, Polecat, Around the Mountain, and Bucktail-related pedestrian access. This is a field-execution gate, not a new optimizer result.

## Decision

The all-route proof remains useful and H1 now counts again as an accepted active route proof for the private field packet. The public-only source layer was ambiguous, but the user-reviewed access confirmation resolves the field-use gate. The correct ongoing action is normal day-of signage and condition checking, not rerouting.

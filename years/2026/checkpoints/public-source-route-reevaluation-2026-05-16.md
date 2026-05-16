# Public-Source Route Reevaluation

Date: 2026-05-16

Frame decision: `needs-proof`.

Purpose: record current public-source findings that change route confidence after the all-route adversarial disproof pass. This checkpoint does not mutate the active field packet by itself. It tells the route proofs what can and cannot be treated as accepted.

## Summary

- Routes rechecked: 43
- Route-set mutation required now: no
- Routes downgraded: 1
- Existing condition gates reaffirmed: 6
- New access-proof gate: `H1`

The strongest new finding is not a same-credit dominance issue. It is an access-proof issue: H1 remains the best known route shape in the current generated packet, but the public source layer no longer supports calling the Avimor Spring Valley Creek start accepted without confirmation.

## Sources Checked

- [Avimor Trails and Outdoors](https://www.avimor.com/trails-and-outdoors): Avimor frames trail access as resident access and describes resident-only permits for some trail uses.
- [Bogus Basin Deer Point Stewardship Project 2026](https://bogusbasin.org/about-bogus/culture/deer-point-stewardship-project-2026/): confirms weekday road/trail closure windows through Friday, June 19, 2026.
- [Ridge to Rivers special management strategies](https://www.ridgetorivers.org/trail-news/ridge-to-rivers-adopts-management-strategies-from-pilot-trail-program/): confirms direction/separation rules for Lower Hulls, Polecat, Around the Mountain, and Bucktail-related pedestrian access.
- [BoiseTrails current trail-status page](https://boisetrails.com/): shows current May 2026 mud, snow, ice, and closure signals. These gate immediate field tests, but do not by themselves mutate the June/July route set.

## Route Impacts

### H1 - Avimor / Harlow Spring

- Previous proof status: `accepted_current`
- New proof status: `needs_public_access_confirmation`
- New route decision: `HOLD_PUBLIC_ACCESS_RECHECK`
- Candidate: `H1-avimor-native-harlow-spring-loop`
- Current start: Avimor Spring Valley Creek parking

Reason: the route proof relied on OSM plus AllTrails parking/start evidence. Avimor's current public owner page says trails are open for Avimor residents. That does not prove that a public Boise Trails Challenge participant can use the Spring Valley Creek start.

What did not change: no known accepted same-credit anchor or optimizer replacement currently dominates the H1 route shape.

Required action: get BTC organizer, Avimor, or current field-signage confirmation for public participant use from the Spring Valley Creek start. If that cannot be proven, redesign H1 from a certifiable public anchor or keep it explicitly gated.

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

The all-route proof remains useful, but it now has one explicit public-access caveat. H1 should not count as an accepted active route proof until access is confirmed above the OSM/AllTrails layer. The correct next action is proof or redesign, not silent promotion.

# Unexpected Optimization Shortlist

## Purpose

This is the compressed result of the runner-perspective frame shift. The long route-by-route audits are evidence. This file names the route changes worth proving next.

## 1. Rebuild The Dry Creek / Shingle / Sweet Connie Cluster Around Real Loop Behavior

Routes affected: `15A-1`, `16A-1`, `16A-2`.

Why this surfaced:

- `16A-2` has a 2.71x on-foot/official ratio and 9.43 non-new-credit miles.
- The runner-perspective audit flags a 4.47-mile overlap repeat and a 7.76-mile exit leg.
- Public Strava route behavior shows a known Shingle-up / Dry-down loop from a Bogus Basin Road pullout, not a pure out-and-back.

Hypothesis:

The current cluster is probably paying for access and return movement in the wrong unit. Rebuild it as a small family of loop/split candidates: Shingle-up/Dry-down, Sweet Connie climb, Sheep Camp add-on, and same-day re-park variants.

Proof path:

- Verify Bogus Basin Road pullout legality and current access.
- Build candidate GPX for Shingle-up/Dry-down with official segment coverage.
- Test whether Sheep Camp belongs in that same card or a separate nearby re-park card.
- Preserve ascent-only rules and future-day coverage.

## 2. Split The Freestone Mega-Route Into Human Loop Units

Routes affected: `13`, secondarily `3`.

Why this surfaced:

- `13` is the top-scoring optimization target: 25.12 on-foot miles, 10.77 non-new-credit miles, seven high-priority overlap/repeat leads.
- Public Strava route behavior shows a compact Shane's / Three Bears loop from Freestone Creek using Mountain Cove as a warm-up and Central Ridge as the return.

Hypothesis:

`13` is likely doing too much in one card. Split it into at least two human-recognizable loops: a Shane's/Three Bears/Freestone start loop and a separate Freestone Ridge / Curlew / Two Point / Fat Tire Traverse solution.

Proof path:

- Extract Shane's/Three Bears/Central Ridge coverage from `13` and validate remaining segment set.
- Build the leftover Freestone Ridge/Curlew candidate separately.
- Compare p75, route-finding penalty, overlap count, and future-day preservation against the current `13`.

## 3. Pull A Compact Hulls / Kestrel / Crestline Loop Out Of The Frontside Bundle

Routes affected: `2`, `12`.

Why this surfaced:

- `2` has 20 medium leads and three high leads, mostly repeat/connector warnings.
- Public route behavior treats Lower Hulls up, Motorcycle connector, Crestline, and Kestrel down as a compact human loop.
- Current local reality makes Lower Hulls day legality a hard constraint, not a note.

Hypothesis:

The frontside bundle is mixing a stable public loop with too many other official edges. A compact Hulls/Kestrel/Crestline card may reduce cognitive load and isolate the odd/even legality rule.

Proof path:

- Build a compact Hulls/Kestrel/Crestline card using even-day foot legality.
- Reassign remaining `2` and `12` official edges into separate cards.
- Compare route count increase against total p75, overlap warnings, and day-of legality simplicity.

## 4. Replace Generic Bogus Return Connectors With Named Runner Return Logic

Routes affected: `17`, `18`.

Why this surfaced:

- Public Strava behavior says Around the Mountain does not close its own loop and runners can return from the Pioneer side via Yellow Brick Road / Bogus Creek.
- Current Bogus cards include OSM service/track connector names and several connector/repeat warnings.

Hypothesis:

The Bogus problem is a named-return and day-level handoff problem more than a pure segment grouping problem. A runner-known Pioneer-to-Simplot return path, or an explicit Simplot/Pioneer same-day transfer, may outperform generic connector routing.

Proof path:

- Verify current 2026 closure window, ATM direction, and foot legality for the return path.
- Compare Simplot-only, Pioneer-only, and same-day transfer variants.
- Replace generic OSM track labels where a named legal runner return is available.

## 5. Re-anchor Table Rock / Quarry / Rock Island Around Real Public Starts

Routes affected: `4C-1`, `4C-2`, secondarily `4A`, `4B`.

Why this surfaced:

- `4C-2` uses a private-history parking anchor and has 10 medium leads.
- Public route behavior shows Old Penitentiary / Eagle Rock / Quarry as a compact public start pattern.
- Rock Island behavior is technical and bike-oriented, which may make a route graph look cleaner than the runner experience.

Hypothesis:

The best Table Rock/Castle/Rock Island route set may be a set of compact public-start loops instead of a private-anchor bundle.

Proof path:

- Compare Old Pen, Warm Springs Golf Course, and Castle-side starts.
- Verify pedestrian legality / yield rules for Rock Island.
- Keep Shoshone-Paiute, Quarry, Table Rock, and Rock Island official coverage intact while reducing anchor dependence.

## 6. Keep Polecat Direction And Start Choice As First-Class Optimization Variables

Routes affected: `5A`, `5B`, `6`.

Why this surfaced:

- Public route behavior says Polecat can start from Polecat Gulch or Cartwright Road and calls out counter-clockwise travel plus mud sensitivity.
- Current route set uses Cartwright heavily and has connector/repeat leads across `5B` and `6`.

Hypothesis:

Polecat/Cardwright work should compare start and direction as a route-quality variable, not assume the current Cartwright anchor is globally best.

Proof path:

- Verify current Polecat direction/signage.
- Compare Cartwright and Polecat Gulch starts for `5B` plus downstream `6`.
- Treat slow-drying/mud behavior as scheduling input.

## 7. Preserve Hillside / Harrison / West Climb Split Logic

Routes affected: `1A-1`, `1A-2`, Harrison-related completed/repeated context.

Why this surfaced:

- Public route behavior confirms Hillside to Hollow is confusing, exposed, and start-dependent.
- Local field history already showed Harrison/West Climb same-corridor confusion and accepted split/re-park behavior.

Hypothesis:

Do not let the optimizer collapse this area back to a single graph-valid route. Preserve split/re-park as a standing candidate pattern and compare Hillside Park, Harrison, West Climb, and 36th Street starts by p75 and confusion cost.

Proof path:

- Re-run the accepted split/re-park preservation audit after any route recalculation.
- Keep cue-level overlap warnings, but treat them as route-choice pressure, not just navigation text.

## Next Concrete Experiment

Start with `16A-2` and the Dry Creek/Shingle/Sweet Connie cluster. It has the strongest combination of public route behavior, high overhead, long non-credit legs, ascent constraints, and access-anchor proof sensitivity. A bounded repair experiment here is more likely to reveal a real route improvement than another global optimizer pass.

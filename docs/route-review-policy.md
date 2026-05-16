# 2026 Route Review Policy

Certification is not route acceptance.

A certified route proves that the route is runnable enough to produce coherent GPX, cues, and official-segment coverage. It does not prove that the chosen start is the right real-world outing for the credit target.

Accepted 2026 route promotion requires all of the following:

- The route is certified or explicitly marked provisional with the missing certification step.
- The route carries `start_justification` explaining why this start/parking anchor is used.
- Official-credit miles are separated from access, return, repeat, and non-credit miles.
- Exact official segment credit is compared against accepted, user-reviewed, or private-derived anchors.
- Single-segment routes receive the same exact-credit dominance review as larger route cards.
- Same-credit alternatives that save at least 0.25 on-foot miles or 10 p75 minutes block promotion unless a valid route/source-hashed waiver exists.

## Deterministic Gate

For each route card, the deterministic review asks:

1. What exact official segment set is being purchased?
2. What start anchor is used?
3. Is the start justified in route data?
4. Is there an accepted or certifiable start that earns the same segment set?
5. Does that same-credit alternative materially reduce human footmiles or p75 minutes?
6. If the current route is longer, is there a current waiver bound to this route/source hash?

Material dominance threshold:

- `>= 0.25` on-foot miles saved, or
- `>= 10` p75 minutes saved.

FD14D is the canonical regression: segment `1482` can be earned from the lower N 36th Street parking anchor with materially less human burden than the stale Full Sail start.

## AI Review

Codex review is local and explicit in phase 1. It should read the route-review pack as a hiker, not as another optimizer, and answer:

- Why this start?
- What official credit is being bought?
- How much movement is access, repeat, return, or non-credit burden?
- Would a real hiker choose this route from scratch?
- Is the card certified but dominated?

Live Codex review is not required in GitHub CI until the deterministic gate, schema, fixtures, and waiver behavior are stable.

## Waivers

A waiver is allowed only when the longer route is intentional because of safety, legality, closure, direction-rule, parking-confidence, cue-simplicity, or other documented field reality.

Waivers live in `years/2026/inputs/personal/private/route-review-waivers-2026.private.json` and must bind to:

- `route_label`
- `segment_ids`
- `route_source_hash`
- `reason`
- `approver`
- `date`
- `expires`

Expired, stale-hash, segment-mismatched, or reason-empty waivers do not unblock promotion.

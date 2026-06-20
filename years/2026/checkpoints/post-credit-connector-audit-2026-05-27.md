# Post-Credit Connector Audit

Generated: 2026-06-20T22:20:32Z
Status: `passed`

## Summary

- Routes audited: 19
- Failed routes: 0
- Findings: 0
- Warnings: 2
- Explicit post-credit connector proofs: 52
- Hidden official-cue exit findings: 0
- Hidden official-cue exit warnings: 2
- Shorter connector findings: 0
- Stale connector-savings metadata findings: 0
- Unproved connector findings: 0
- Source-gap proof blockers: 0
- Route-card/GPX mileage warnings: 0

## Findings

No findings.

## Warnings

| Route | Cue | Code | Miles | Feet | Message |
|---|---:|---|---:|---:|---|
| 1-2: 1A-2 | 5 | official_credit_cue_hides_post_credit_exit | 0.06 | 317 | Official-credit cue source geometry includes extra movement; review cue splitting, but connector shortest-path proof is handled by explicit post-credit connector cues. |
| 4-3: 4C | 6 | official_credit_cue_hides_post_credit_exit | 0.12 | 634 | Official-credit cue source geometry includes extra movement; review cue splitting, but connector shortest-path proof is handled by explicit post-credit connector cues. |

## Scope

- This audit proves generated field-packet cue intervals, not abstract segment lists.
- It excludes still-unearned route segments from connector alternatives so a shortcut cannot silently consume future official credit.
- Official-credit cue source paths with extra movement are carried as cue-splitting warnings; hard failures are reserved for missing, unproved, or non-shortest post-credit connector proofs.

# Post-Credit Connector Audit

Generated: 2026-06-23T16:37:25Z
Status: `passed`

## Summary

- Routes audited: 25
- Failed routes: 0
- Findings: 0
- Warnings: 6
- Explicit post-credit connector proofs: 88
- Hidden official-cue exit findings: 0
- Hidden official-cue exit warnings: 6
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
| 3-1: 3 | 7 | official_credit_cue_hides_post_credit_exit | 0.54 | 2851 | Official-credit cue source geometry includes extra movement; review cue splitting, but connector shortest-path proof is handled by explicit post-credit connector cues. |
| 3-1: 3 | 9 | official_credit_cue_hides_post_credit_exit | 0.25 | 1320 | Official-credit cue source geometry includes extra movement; review cue splitting, but connector shortest-path proof is handled by explicit post-credit connector cues. |
| 3-1: 3 | 23 | official_credit_cue_hides_post_credit_exit | 0.03 | 158 | Official-credit cue source geometry includes extra movement; review cue splitting, but connector shortest-path proof is handled by explicit post-credit connector cues. |
| 4-3: 4C | 6 | official_credit_cue_hides_post_credit_exit | 0.12 | 634 | Official-credit cue source geometry includes extra movement; review cue splitting, but connector shortest-path proof is handled by explicit post-credit connector cues. |
| 18-1: 18A | 3 | official_credit_cue_hides_post_credit_exit | 0.31 | 1637 | Official-credit cue source geometry includes extra movement; review cue splitting, but connector shortest-path proof is handled by explicit post-credit connector cues. |
| 18-1: 18A | 9 | official_credit_cue_hides_post_credit_exit | 1.75 | 9240 | Official-credit cue source geometry includes extra movement; review cue splitting, but connector shortest-path proof is handled by explicit post-credit connector cues. |

## Scope

- This audit proves generated field-packet cue intervals, not abstract segment lists.
- It excludes still-unearned route segments from connector alternatives so a shortcut cannot silently consume future official credit.
- Official-credit cue source paths with extra movement are carried as cue-splitting warnings; hard failures are reserved for missing, unproved, or non-shortest post-credit connector proofs.

# Public History Summary

Created: 2026-05-04

This is a derived, privacy-light rollup from the Boise Trails Challenge public history API. Raw history files include public participant identifiers/profile image URLs and remain ignored in `years/2026/inputs/official/api-pull-2026-05-04/history/`.

| Year | Target segments | Target trails | Target miles | Finishers | Scott public record |
| --- | ---: | ---: | ---: | ---: | --- |
| 2018 | 187 | 80 | 162.9 | 94 | not found |
| 2019 | 193 | 83 | 158.31 | 203 | not found |
| 2020 | 200 | 86 | 168.72 | 500 | not found |
| 2021 | 216 | 94 | 170.43 | 365 | not found |
| 2022 | 223 | 92 | 171.08 | 397 | not found |
| 2023 | 229 | 94 | 175.47 | 441 | not found |
| 2024 | 236 | 98 | 177.25 | 315 | 64.87% / 114.97 mi / rank 364 |
| 2025 | 245 | 98 | 164.73 | 376 | 41.82% / 68.89 mi / rank 491 |

## Notes

- These targets are inferred from public finisher rows because `GET /api/history/:year` does not return historical segment geometry.
- 2024 and 2025 contain public Scott Falconer records. Earlier years in this pull do not contain an exact `Scott Falconer` leaderboard match.
- 2025 has a known discrepancy: the local planner file has 247 segments / 100 trails / 169.354 mi, while public history completion math uses 245 segments / 98 trails / 164.73 mi. The user-reported 2025 result matches the public history value.
- The only historical official segment geometry currently preserved locally is the 2025 file copied to `archive/years/2025/inputs/official/local-legacy-2025/GETChallengeTrailData_v2.json`.
- Annual target drift can be legitimate. User-provided organizer email excerpts document 2024 Polecat fire / Heroes / Bogus construction issues and a 2025 Ridge Road / Mahalo closure. See `challenge-change-events-2026-05-04.md`.

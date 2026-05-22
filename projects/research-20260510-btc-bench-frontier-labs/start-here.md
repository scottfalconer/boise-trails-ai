# Start Here

This is a research bundle for the BTC-Bench idea:

**Field-executable route planning as a grounded benchmark for language agents.**

Your job:

- judge whether this project can credibly support an arXiv/preprint or frontier-lab pitch
- refine the research contribution and novelty boundary
- propose the minimum empirical benchmark/eval shape

Do not spend time on:

- re-solving the Boise Trails route plan
- using private planner state, private GPS traces, credentials, or raw dashboard data
- claiming global route-optimization novelty
- treating repo-local "skills" as the paper's central vocabulary

Read first:

- `research-answer`
- `related-work-matrix`
- `repo-readme`
- `btc-heuristics`
- `btc-field-packet-requirements`
- `btc-cases`
- `btc-behavior-evals`
- `field-test-2026-05-05-analysis`
- `field-test-2026-05-08-analysis`
- `field-tool-completion-audit`

Raw or archive-only artifacts:

- `bundle-assembly-transcript` is provenance for how this bundle was assembled.
- `public-literature-source-urls` and the `arxiv-*` / `openreview-*` snapshots are source verification, not prose drafts.
- The public literature snapshots are HTML source captures; use `related-work-matrix` first unless you need to inspect abstracts directly.

Already believed:

- The strongest contribution is benchmark/evaluation methodology, not a new route optimizer.
- The key failure distinction is graph-valid / GPX-valid / credit-correct versus human-executable.
- Harrison Hollow is the strongest field-backed case study.
- The paper should use neutral terms such as heuristics, procedural policies, verifier gates, repair protocols, and failure taxonomy.

Still needs judgment:

- Whether to scope the first benchmark release to route repair tasks, full route generation tasks, or both.
- How much field evidence is enough for a preprint versus a blog/position paper.
- Which public artifacts need more sanitization before third-party sharing.
- Which baseline agents and model setups are realistic to run.

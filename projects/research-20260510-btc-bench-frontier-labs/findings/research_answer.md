# Research Answer

## Question

Create a research-bundler evidence pack for the BTC-Bench / frontier-lab / arXiv framing:

> BTC-Bench: Field-executable route planning as a grounded evaluation for language agents.

## Conclusion

The project has a strong benchmark-shaped research candidate:

**BTC-Bench: evaluating whether language agents can produce and repair field-executable spatial plans while maintaining consistency across official rules, route geometry, generated artifacts, human constraints, field evidence, and recertification after reality changes.**

This overlaps almost completely with the frontier-lab framing from the conversation. The strongest contribution is not that an AI made a Boise route, and not that the repo invents a new global optimizer. The contribution is a grounded verifier methodology and failure taxonomy for agents operating in a physical-world planning task.

## Confidence

High for the framing and evidence fit. Medium for academic novelty until a deeper bibliography pass compares full methods and benchmark tasks rather than abstracts.

## Why This Has Legs

- The repo defines the core problem as edge coverage under human constraints, not waypoint visiting or map prettiness.
- The 2026 dataset is concrete: 101 official on-foot trails, 251 official segments, 164.43 official miles, and 23 ascent-only segments.
- The system already has executable evaluation surfaces: field-packet requirements, field-tool completion audit, field-route walkthrough audit, progress/recertification rules, and privacy checks.
- It has observed failures and repair loops: May 5 Harrison Hollow miss, May 8 repair/validation, Buena Vista repeat/connector issue, route-specific exception debt, and multi-start parking/access gates.
- It has seed benchmark artifacts: heuristics, failure modes, cases, behavior eval prompts, and repo-local procedural skills.

## Recommended Paper Shape

Title candidate:

**BTC-Bench: Field-Executable Route Planning as a Grounded Evaluation for Language Agents**

Core experiment:

1. Provide agents the public official segment set, challenge rules, and public-safe planning constraints.
2. Ask agents to produce, inspect, or repair route artifacts.
3. Score with deterministic checks: segment coverage, ascent direction, car-to-car continuity, access assumptions, cue completeness, artifact consistency, recertification behavior, and privacy leakage.
4. Compare prompt-only agents, doctrine/instruction-augmented agents, and verifier-gated agents.
5. Include field-backed case studies for the May 5 Harrison Hollow miss and May 8 repair.

## Important Boundary

Do not make "skills" central in the paper title or abstract. Use neutral language such as procedural policies, repair protocols, operational heuristics, verifier gates, or agent doctrine. Mention repo-local skills as one implementation form.

Do not claim broad outdoor-routing generality. Frame this as a benchmark prototype plus reproducible verifier method, with synthetic perturbations and sanitized failure cases as the expansion path.

## Best Next Step

Convert the current Markdown behavior eval seeds into a small executable eval subset:

- task prompt
- required input artifacts
- expected route/artifact change
- deterministic verifier command
- pass/fail rubric
- optional human-field-evidence label

That would turn the paper idea from a plausible essay into an empirical benchmark.

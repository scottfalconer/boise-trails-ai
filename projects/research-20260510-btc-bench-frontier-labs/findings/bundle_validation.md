# Bundle Validation

Validation run after `llm_bundle.json` build:

```text
manifest_items 40
bundle_artifacts 40
missing 0
has_bundle_brief True
entrypoint_artifact_id start-here
entrypoint_in_bundle True
missing_roles 0
has_conversation_context True
```

Bundle size:

```text
308K projects/research-20260510-btc-bench-frontier-labs/llm_bundle.json
```

Sensitive-token scan note:

- A broad token/credential regex found only false positives:
  - arXiv HTML uses `accesskey` attributes.
  - The exported bundle-assembly transcript includes the research-bundler skill text, which mentions credential-handling rules and environment variable names.
- No active credential value, bearer token, API key value, password, or private source file was intentionally bundled.

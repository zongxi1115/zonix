# Changelog

## 0.2.1

- Fixed the README logo URL so PyPI can render it from GitHub.
- Prepared the first PyPI distribution.

## 0.2.0

- Added provider-level structured output control:
  - OpenAI-compatible models now prefer strict JSON Schema response formats.
  - Anthropic-compatible models now use a final-output tool with `input_schema`.
  - Pydantic validation can request one output repair pass by default.
- Added Anthropic-compatible provider support.
- Fixed OpenAI-compatible streaming providers that send empty terminal chunks.
- Preserved assistant tool-call messages so tool results round-trip correctly.
- Added real provider examples and a full smoke script covering agents, streams,
  tools, HITL, resume, retry, timeout, fallback, workflow, team, router, memory,
  and wire adapters.
- Added a Chinese multi-chapter tutorial.

## 0.1.0

- Initial standalone repository.
- Added explicit `agent`, `workflow`, `team`, and `router` primitives.
- Added serializable run results, spans, usage, typed stream events, tool schemas,
  memory strategies, HITL checkpoints, and an AI SDK data stream adapter.

# Changelog

## 0.3.0

- Added model call inspection on `RunResult`, including raw upstream requests,
  raw upstream responses, provider status, finish reasons, and per-call usage.
- Expanded usage accounting with cached, cache creation, cache read, reasoning,
  and thinking token fields.
- Added chainable OpenAI Responses API controls for reasoning effort, reasoning
  summaries, verbosity, output limits, previous responses, and includes.
- Added Anthropic thinking, adaptive effort, service tier, metadata, native
  thinking stream handling, and richer usage details.
- Added a Gemini adapter with thinking budget, thought summaries, JSON output,
  safety settings, tool config, and cached-content controls.
- Added workflow and team graph exports as Mermaid, DOT, SVG, PNG, or PDF.
- Updated documentation around Zonix's simple-first, chainable, traceable
  design style and OpenAI-compatible providers such as DeepSeek.

## 0.2.2

- Added first-class manually supplied `message_history` for agents, workflows,
  and teams.
- Added message helper constructors for user, system, assistant, assistant
  tool-call, and tool-result messages.

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

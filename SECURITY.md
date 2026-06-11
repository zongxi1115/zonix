# Security Policy

Zonix can execute user-defined tools. Treat tool registration as code execution:

- Never register untrusted callables.
- Use `approval=True` for tools that mutate files, databases, cloud resources,
  or external systems.
- Persist checkpoints outside public directories.
- Redact secrets from prompts, trace attributes, and tool outputs before sharing
  run dumps.

Please report security issues privately to the repository owner.

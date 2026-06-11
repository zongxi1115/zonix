# Contributing

Zonix is intentionally small and explicit. Contributions should keep these
constraints intact:

- Prefer typed objects over magic strings.
- Keep run state serializable at node boundaries.
- Keep `__call__`, `.run()`, and `.stream()` on the same execution path.
- Add new model providers through `zonix.models.BaseChatModel`.
- Add new frontend protocols under `zonix.wire`.

Local development:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
python -m compileall src
```

Do not commit generated build artifacts.

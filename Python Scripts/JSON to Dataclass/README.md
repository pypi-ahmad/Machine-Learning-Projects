# JSON to Dataclass

Generate Python dataclasses, Pydantic models, TypedDicts, or attrs classes from a JSON example.

```powershell
uv sync --no-config
uv run --no-config python main.py --json '{"name":"Alice","age":30}'
```

Use `--style dataclass`, `pydantic`, `typeddict`, or `attrs`; Pydantic and attrs are imports in the generated code, not dependencies of this generator. `--url` fetches remote JSON, while `--file` and `--json` are local inputs.

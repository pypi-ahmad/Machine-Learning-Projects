# Conway Game of Life

Terminal implementation of Conway's Game of Life with random boards, preset patterns, custom coordinates, and animation.

```powershell
uv sync
uv run python main.py
```

Use Ctrl+C to stop an active animation. The board wraps at its edges, so cells on opposite edges are neighbors.

Verify syntax with `uv run python -m py_compile main.py`.

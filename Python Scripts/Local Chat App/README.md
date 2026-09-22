# Local Chat App

A peer-to-peer Tkinter chat application for a trusted local network. It uses Python sockets and does not require third-party packages.

```powershell
uv sync --no-config
uv run --no-config python main.py
```

Only connect to trusted hosts. Messages are sent over the local network; the GUI and socket listener were not started during headless verification.

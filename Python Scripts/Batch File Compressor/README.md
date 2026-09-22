# Batch File Compressor

Create ZIP, TAR.GZ, and TAR.BZ2 archives, list their contents, or extract them
through the interactive terminal menu.

```powershell
cd "Python Scripts/Batch File Compressor"
uv sync
uv run python main.py
```

Archive extraction rejects member paths that would escape the chosen output
directory. The project uses only the Python standard library.

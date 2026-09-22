# File Unzipper

A preview-first command-line tool for extracting one ZIP archive into a new or empty directory.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)

## Preview an archive

```powershell
uv sync --no-config
uv run --no-config python script.py archive.zip
```

Preview prints the destination, member count, and declared uncompressed size without creating files. By default, `archive.zip` would extract into an `archive` folder beside it.

## Extract

```powershell
uv run --no-config python script.py archive.zip --destination extracted --extract
```

Type `EXTRACT` when prompted. The destination must be new or empty, so existing files cannot be overwritten by this tool.

## Safety checks

- Rejects archive entries that resolve outside the destination directory (zip-slip protection).
- Rejects symlink entries.
- Rejects archives with more than 10,000 members or more than 2 GiB of declared uncompressed content.
- Does not extract anything unless both `--extract` and typed confirmation are provided.

Review archives from untrusted sources before extracting them. These checks reduce common risks but do not make untrusted content safe to open after extraction.

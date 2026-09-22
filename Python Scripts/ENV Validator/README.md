# ENV Validator

A dependency-free command-line tool for checking an explicitly named `.env` file against an optional schema. It reports variable names, counts, and validation results; it never prints environment-variable values.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)

## Setup

```powershell
uv sync --no-config
```

## Validate a file

```powershell
uv run --no-config python main.py .env
```

The tool checks basic formatting, warns about placeholder-like sensitive values, and reports schema violations without revealing values.

## Use a schema

Pass a schema explicitly, or save it beside the environment file as `.env.schema` for automatic discovery.

```powershell
uv run --no-config python main.py .env --schema .env.schema
```

Each non-comment schema line has this shape:

```text
KEY=type,required,min=1,max=65535,pattern=REGEX
```

Supported types are `string`, `int`, `float`, `bool`, `url`, `email`, and `port`.

## Compare key sets

```powershell
uv run --no-config python main.py .env.production --compare .env.staging
```

Comparison lists keys that are missing or have different values, but deliberately never prints either value.

## Safety

- Pass each file path explicitly. The tool does not scan directories for environment files.
- Treat `.env` files as sensitive and keep them out of version control.
- Validation can identify likely placeholders and malformed values; it does not verify that a credential works.

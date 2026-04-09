# Convert XML to JSON

A modern CLI tool to convert XML files to JSON.

Replaces the legacy `converter.py` script with a fully-featured,
tested, and type-safe implementation.

## What changed from the legacy script

| Aspect            | Legacy (`converter.py`)           | Modern (`xml2json`)                        |
| ----------------- | --------------------------------- | ------------------------------------------ |
| Input             | Hardcoded `input.xml`             | Any file path as CLI argument              |
| Output            | Hardcoded `output.json`           | stdout (default) or `--out <path>`         |
| Formatting        | Compact (no indent)               | `--pretty`, `--indent N`, `--sort-keys`    |
| Encoding          | System default                    | `--encoding` flag (default utf-8)          |
| Error handling    | Crashes on bad input              | Clear error messages, exit code 2          |
| CLI               | None                              | Typer with `--help`, flags, arguments      |
| Tests             | None                              | 30+ pytest tests                           |
| Packaging         | `requirements.txt`                | `pyproject.toml` (PEP 621) + src layout    |
| XML parser        | `xmltodict`                       | `xmltodict` (same, but pinned properly)    |

## Installation

```bash
cd Convert_XML_To_JSON
pip install -e .
```

## Usage

### Convert and print to stdout (compact)

```bash
xml2json input.xml
```

### Pretty-print with 2-space indent

```bash
xml2json input.xml --pretty
```

### Custom indent

```bash
xml2json input.xml --indent 4
```

### Write to file

```bash
xml2json input.xml --out output.json --pretty
```

### Sort keys alphabetically

```bash
xml2json input.xml --pretty --sort-keys
```

### Specify encoding

```bash
xml2json legacy.xml --encoding latin-1 --pretty
```

## Options

| Option                | Description                           | Default |
| --------------------- | ------------------------------------- | ------- |
| `INPUT_FILE`          | Path to the XML file (required)       | -       |
| `--out`, `-o`         | Write JSON to this file               | stdout  |
| `--pretty`, `-p`      | Pretty-print with 2-space indent      | off     |
| `--indent`, `-i`      | Number of indent spaces (overrides -p)| None    |
| `--sort-keys`, `-s`   | Sort keys alphabetically              | off     |
| `--encoding`, `-e`    | XML file encoding                     | utf-8   |
| `--verbose`, `-v`     | Enable debug logging                  | off     |

## Exit codes

| Code | Meaning                                |
| ---- | -------------------------------------- |
| 0    | Conversion successful                  |
| 2    | Invalid input, parse error, I/O error  |

## Development

```bash
pip install -e .
pip install ruff pytest
ruff check src tests
ruff format --check src tests
pytest -q
```

## Project structure

```
Convert_XML_To_JSON/
  pyproject.toml
  README.md
  input.xml              # sample input (kept from legacy)
  output.json            # sample output (kept from legacy)
  src/convert_xml_to_json/
    __init__.py           # version
    core.py               # XML parsing, JSON serialisation, file I/O
    cli.py                # Typer CLI entry point
  tests/
    test_core.py          # Unit tests: parsing, serialisation, file helpers
    test_cli.py           # CLI integration tests via CliRunner
```


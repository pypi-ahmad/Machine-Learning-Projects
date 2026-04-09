"""Typer CLI for xml2json."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from convert_xml_to_json.core import (
    convert_xml_to_json,
    read_xml_file,
    write_json_file,
)

app = typer.Typer(
    name="xml2json",
    help="Convert XML files to JSON.",
    add_completion=False,
)

logger = logging.getLogger("convert_xml_to_json")

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------
EXIT_OK = 0
EXIT_ERROR = 2


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level, force=True)


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------


@app.command()
def convert(
    input_file: Path = typer.Argument(  # noqa: B008
        ...,
        help="Path to the XML file to convert.",
        exists=True,
        readable=True,
    ),
    out: Path | None = typer.Option(  # noqa: B008
        None,
        "--out",
        "-o",
        help="Write JSON to this file instead of stdout.",
    ),
    pretty: bool = typer.Option(
        False,
        "--pretty",
        "-p",
        help="Pretty-print with 2-space indent (shorthand for --indent 2).",
    ),
    indent: int | None = typer.Option(
        None,
        "--indent",
        "-i",
        help="Number of spaces for indentation. Overrides --pretty.",
    ),
    sort_keys: bool = typer.Option(
        False,
        "--sort-keys",
        "-s",
        help="Sort JSON object keys alphabetically.",
    ),
    encoding: str = typer.Option(
        "utf-8",
        "--encoding",
        "-e",
        help="Character encoding of the XML file.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Convert an XML file to JSON."""
    _configure_logging(verbose)

    # Resolve indentation: --indent overrides --pretty
    resolved_indent: int | None
    if indent is not None:
        resolved_indent = indent
    elif pretty:
        resolved_indent = 2
    else:
        resolved_indent = None  # compact

    try:
        xml_text = read_xml_file(input_file, encoding=encoding)
        json_text = convert_xml_to_json(
            xml_text,
            encoding=encoding,
            indent=resolved_indent,
            sort_keys=sort_keys,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=EXIT_ERROR) from exc

    if out is not None:
        try:
            write_json_file(out, json_text, encoding=encoding)
        except OSError as exc:
            typer.echo(f"Error writing file: {exc}", err=True)
            raise typer.Exit(code=EXIT_ERROR) from exc
        typer.echo(f"Written to {out}")
    else:
        typer.echo(json_text)

    raise typer.Exit(code=EXIT_OK)

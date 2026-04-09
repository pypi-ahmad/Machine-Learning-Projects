"""Typer CLI for check-website-connectivity."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from check_website_connectivity.core import (
    check_urls,
    format_json,
    format_table,
    load_urls,
    normalize_url,
    validate_url,
    write_csv,
)

app = typer.App = typer.Typer(
    name="check_site",
    help="Check whether websites are reachable.",
    add_completion=False,
)

logger = logging.getLogger("check_website_connectivity")

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------
EXIT_OK = 0
EXIT_UNREACHABLE = 1
EXIT_INPUT_ERROR = 2


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=level,
        force=True,
    )


def _exit_code(all_ok: bool) -> int:
    return EXIT_OK if all_ok else EXIT_UNREACHABLE


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------


@app.command()
def check(
    urls: list[str] | None = typer.Argument(  # noqa: B008
        default=None,
        help="One or more URLs to check. Omit if using --file.",
    ),
    file: Path | None = typer.Option(  # noqa: B008
        None,
        "--file",
        "-f",
        help="Path to a .txt or .csv file containing URLs (one per line).",
        exists=True,
        readable=True,
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print results as JSON.",
    ),
    csv_output: Path | None = typer.Option(  # noqa: B008
        None,
        "--csv",
        help="Write results to a CSV file at this path.",
    ),
    timeout: float = typer.Option(
        10.0,
        "--timeout",
        "-t",
        help="Request timeout in seconds.",
    ),
    retries: int = typer.Option(
        2,
        "--retries",
        "-r",
        help="Number of retries on transient errors.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Check connectivity for one or more websites."""
    _configure_logging(verbose)

    # --- Collect URLs -------------------------------------------------------
    collected: list[str] = []

    if file is not None:
        logger.info("Loading URLs from %s", file)
        collected.extend(load_urls(file))

    if urls:
        for raw in urls:
            normalised = normalize_url(raw)
            if not normalised or not validate_url(normalised):
                logger.error("Invalid URL: %s", raw)
                typer.echo(f"Error: invalid URL '{raw}'", err=True)
                raise typer.Exit(code=EXIT_INPUT_ERROR)
            collected.append(normalised)

    if not collected:
        typer.echo("Error: no URLs provided. Pass URLs as arguments or use --file.", err=True)
        raise typer.Exit(code=EXIT_INPUT_ERROR)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for u in collected:
        if u not in seen:
            seen.add(u)
            unique.append(u)

    # --- Check URLs ---------------------------------------------------------
    results = check_urls(unique, timeout=timeout, retries=retries)

    # --- Output -------------------------------------------------------------
    if json_output:
        typer.echo(format_json(results))
    else:
        typer.echo(format_table(results))

    if csv_output is not None:
        write_csv(results, csv_output)
        logger.info("CSV written to %s", csv_output)
        if not json_output:
            typer.echo(f"\nCSV saved to {csv_output}")

    # --- Exit code ----------------------------------------------------------
    all_ok = all(r.ok for r in results)
    raise typer.Exit(code=_exit_code(all_ok))

"""Core logic: URL parsing, validation, checking, and output formatting."""

from __future__ import annotations

import csv
import io
import json
import logging
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from check_website_connectivity.models import CheckResult, Status

logger = logging.getLogger("check_website_connectivity")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT: float = 10.0
DEFAULT_RETRIES: int = 2
USER_AGENT = "check-website-connectivity/1.0 (https://github.com/Python-Projects)"


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def normalize_url(raw: str) -> str:
    """Ensure *raw* has a scheme; default to ``https://``.

    Strips whitespace. Returns the normalised URL string.
    """
    url = raw.strip()
    if not url:
        return ""
    parsed = urlparse(url)
    if not parsed.scheme:
        url = f"https://{url}"
        parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return ""  # unsupported scheme
    return url


def validate_url(url: str) -> bool:
    """Return *True* when *url* looks like a valid HTTP(S) URL."""
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------


def parse_urls_from_text(text: str) -> list[str]:
    """Extract non-empty, normalised URLs from newline-separated text."""
    urls: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        normalised = normalize_url(line)
        if normalised:
            urls.append(normalised)
    return urls


def parse_urls_from_csv(path: Path) -> list[str]:
    """Read URLs from a CSV file.

    Supported layouts:
    - Single column: each cell is a URL.
    - Multi-column: first column that looks like a URL wins.
    Falls back to treating every non-empty cell as a potential URL.
    """
    urls: list[str] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            for cell in row:
                cell = cell.strip()
                if not cell:
                    continue
                # Skip obvious header labels
                if cell.lower() in {"website", "url", "site", "status", "link"}:
                    continue
                normalised = normalize_url(cell)
                if normalised and validate_url(normalised):
                    urls.append(normalised)
                    break  # one URL per row
    return urls


def load_urls(source: Path) -> list[str]:
    """Load URLs from a file (txt or csv) based on extension."""
    if source.suffix.lower() == ".csv":
        return parse_urls_from_csv(source)
    # Default: plain-text, one URL per line
    return parse_urls_from_text(source.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Connectivity checking
# ---------------------------------------------------------------------------


def check_url(
    url: str,
    *,
    client: httpx.Client | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> CheckResult:
    """Check a single URL and return a :class:`CheckResult`.

    Retries on transient network errors.  Accepts an optional *client* to
    allow connection pooling and test injection.
    """
    own_client = client is None
    if own_client:
        transport = httpx.HTTPTransport(retries=retries)
        client = httpx.Client(transport=transport, follow_redirects=True)

    try:
        start = time.monotonic()
        resp = client.get(
            url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        )
        elapsed = (time.monotonic() - start) * 1000
        ok = 200 <= resp.status_code < 400
        return CheckResult(
            url=url,
            status=Status.REACHABLE if ok else Status.UNREACHABLE,
            status_code=resp.status_code,
            reason=resp.reason_phrase or "",
            response_time_ms=round(elapsed, 1),
        )
    except httpx.TimeoutException:
        logger.debug("Timeout checking %s", url)
        return CheckResult(url=url, status=Status.UNREACHABLE, reason="timeout")
    except httpx.HTTPError as exc:
        logger.debug("HTTP error checking %s: %s", url, exc)
        return CheckResult(url=url, status=Status.ERROR, reason=str(exc))
    finally:
        if own_client:
            client.close()


def check_urls(
    urls: list[str],
    *,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> list[CheckResult]:
    """Check multiple URLs with a shared connection pool."""
    transport = httpx.HTTPTransport(retries=retries)
    results: list[CheckResult] = []
    with httpx.Client(transport=transport, follow_redirects=True) as client:
        for url in urls:
            logger.info("Checking %s ...", url)
            results.append(check_url(url, client=client, timeout=timeout, retries=retries))
    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_table(results: list[CheckResult]) -> str:
    """Format results as a human-readable table."""
    if not results:
        return "No results."
    # Column widths
    url_width = max(len(r.url) for r in results)
    url_width = max(url_width, 3)  # minimum "URL"
    lines: list[str] = []
    header = f"{'URL':<{url_width}}  {'Status':<12}  {'Code':>4}  {'Time (ms)':>10}  Reason"
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        code = str(r.status_code) if r.status_code is not None else "-"
        ms = f"{r.response_time_ms:.0f}" if r.response_time_ms is not None else "-"
        line = f"{r.url:<{url_width}}  {r.status.value:<12}  {code:>4}  {ms:>10}  {r.reason}"
        lines.append(line)
    return "\n".join(lines)


def format_json(results: list[CheckResult]) -> str:
    """Format results as a JSON array."""
    return json.dumps([r.to_dict() for r in results], indent=2)


def write_csv(results: list[CheckResult], path: Path) -> None:
    """Write results to a CSV file."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Website", "Status", "Code", "Time_ms", "Checked_at"])
        for r in results:
            writer.writerow(
                [
                    r.url,
                    r.status_label(),
                    r.status_code or "",
                    r.response_time_ms or "",
                    r.checked_at,
                ]
            )


def format_csv_string(results: list[CheckResult]) -> str:
    """Return CSV content as a string (for testing)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Website", "Status", "Code", "Time_ms", "Checked_at"])
    for r in results:
        writer.writerow(
            [
                r.url,
                r.status_label(),
                r.status_code or "",
                r.response_time_ms or "",
                r.checked_at,
            ]
        )
    return buf.getvalue()

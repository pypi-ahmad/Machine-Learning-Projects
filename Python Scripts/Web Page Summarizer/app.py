"""Create short extractive summaries from HTML pages or a CSV of page URLs."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


USER_AGENT = "WebPageSummarizer/1.0"


class PageTextParser(HTMLParser):
    """Collect visible text while ignoring script and style content."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self.hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.hidden_depth:
            self.hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden_depth:
            self.parts.append(data)


def valid_url(value: str) -> str:
    """Accept only absolute HTTP(S) URLs."""
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise argparse.ArgumentTypeError("URL must use http or https and include a host.")
    return value


def fetch_html(url: str, timeout: float, max_bytes: int) -> str:
    """Fetch a bounded HTML response with a descriptive user agent."""
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"})
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get_content_type()
        if content_type not in {"text/html", "application/xhtml+xml"}:
            raise ValueError(f"Expected HTML, received {content_type}.")
        content = response.read(max_bytes + 1)
        encoding = response.headers.get_content_charset() or "utf-8"
    if len(content) > max_bytes:
        raise ValueError(f"Page exceeds the {max_bytes} byte limit.")
    return content.decode(encoding, errors="replace")


def page_text(html: str) -> str:
    """Return normalized visible text from an HTML document."""
    parser = PageTextParser()
    parser.feed(html)
    parser.close()
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def summarize_text(text: str, sentence_count: int) -> str:
    """Select frequent-word sentences while preserving their original order."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    usable_sentences: list[str] = []
    for sentence in sentences:
        if sentence.strip():
            usable_sentences.append(sentence.strip())
    if not usable_sentences:
        raise ValueError("The page did not contain readable text.")

    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    frequencies = Counter(words)
    ranked: list[tuple[float, int, str]] = []
    for position, sentence in enumerate(usable_sentences):
        sentence_words = re.findall(r"[A-Za-z0-9']+", sentence.lower())
        if sentence_words:
            score = sum(frequencies[word] for word in sentence_words) / len(sentence_words)
            ranked.append((score, position, sentence))
    ranked.sort(reverse=True)
    selected = ranked[:sentence_count]
    selected.sort(key=lambda item: item[1])
    return " ".join(item[2] for item in selected)


def summarize_url(url: str, sentence_count: int, timeout: float, max_bytes: int) -> str:
    """Fetch one web page and return an extractive summary."""
    return summarize_text(page_text(fetch_html(url, timeout, max_bytes)), sentence_count)


def summarize_csv(source: Path, output: Path, column: str, sentence_count: int, timeout: float, max_bytes: int) -> int:
    """Summarize URL rows and write summaries plus per-row errors to a new CSV."""
    if not source.is_file():
        raise FileNotFoundError(f"CSV does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose a new output path.")
    if source.resolve() == output.resolve():
        raise ValueError("The output CSV must differ from the input CSV.")

    with source.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames or column not in reader.fieldnames:
            raise ValueError(f"CSV must contain a {column!r} column.")
        fieldnames = [*reader.fieldnames, "summary", "error"]
        with output.open("x", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            rows_written = 0
            for row in reader:
                try:
                    row["summary"] = summarize_url(valid_url(row[column]), sentence_count, timeout, max_bytes)
                    row["error"] = ""
                except (HTTPError, OSError, URLError, ValueError) as error:
                    row["summary"] = ""
                    row["error"] = str(error)
                writer.writerow(row)
                rows_written += 1
    return rows_written


def add_common_options(parser: argparse.ArgumentParser) -> None:
    """Add shared summary controls to a subcommand parser."""
    parser.add_argument("--sentences", type=int, default=2, help="sentences to select (default: 2)")
    parser.add_argument("--timeout", type=float, default=10, help="request timeout in seconds (default: 10)")
    parser.add_argument("--max-bytes", type=int, default=2_000_000, help="maximum response size (default: 2000000)")


def main() -> None:
    """Parse a URL or CSV command and print the resulting summary status."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    url_command = commands.add_parser("url", help="summarize one URL")
    url_command.add_argument("url", type=valid_url)
    add_common_options(url_command)
    csv_command = commands.add_parser("csv", help="summarize URLs from a CSV")
    csv_command.add_argument("source", type=Path, help="CSV with a URL column")
    csv_command.add_argument("output", type=Path, help="new CSV path")
    csv_command.add_argument("--column", default="website", help="URL column name (default: website)")
    add_common_options(csv_command)
    args = parser.parse_args()
    if args.sentences < 1 or args.timeout <= 0 or args.max_bytes < 1:
        parser.error("--sentences, --timeout, and --max-bytes must be positive.")

    try:
        if args.command == "url":
            print(summarize_url(args.url, args.sentences, args.timeout, args.max_bytes))
        else:
            print(f"Wrote {summarize_csv(args.source, args.output, args.column, args.sentences, args.timeout, args.max_bytes)} row(s) to {args.output}.")
    except (HTTPError, OSError, URLError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()

"""Display trending records saved by youtube_scrapper.py."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from pymongo import MongoClient


def parse_args() -> argparse.Namespace:
    """Choose a CSV file or explicitly configured MongoDB source."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="CSV file produced by the scraper")
    source.add_argument("--mongo-uri", help="MongoDB URI used by the scraper")
    parser.add_argument("--database", default="youtube", help="MongoDB database name (default: youtube)")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read UTF-8 records from a scraper CSV output."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_mongo(uri: str, database: str) -> list[dict[str, str]]:
    """Read MongoDB records and discard the internal identifier."""
    with MongoClient(uri, serverSelectionTimeoutMS=5_000) as client:
        return list(client[database]["trending"].find({}, {"_id": False}))


def display(records: list[dict[str, str]]) -> None:
    """Print saved records in a readable text format."""
    for record in records:
        print(f"Section: {record.get('section', '')}")
        print(f"Title: {record.get('title', '')}")
        print(f"Link: {record.get('link', '')}")
        print(f"Channel: {record.get('channel', '')}")
        print(f"Views: {record.get('views', '')}")
        print(f"Time: {record.get('date', '')}")
        print("-" * 48)


def main() -> None:
    """Read and display one configured storage source."""
    args = parse_args()
    try:
        records = read_csv(args.csv) if args.csv else read_mongo(args.mongo_uri, args.database)
    except OSError as error:
        raise SystemExit(f"Read failed: {error}") from error
    display(records)


if __name__ == "__main__":
    main()

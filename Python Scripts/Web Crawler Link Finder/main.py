"""Crawl same-domain links from a user-supplied homepage."""

from __future__ import annotations

import argparse
import threading
from queue import Queue
from urllib.parse import urlparse

from domain import get_domain_name
from general import file_to_set
from spider import Spider

queue: Queue[str] = Queue()


def parse_args() -> argparse.Namespace:
    """Parse the crawl target and local output settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Homepage URL to crawl, including http:// or https://")
    parser.add_argument(
        "--project",
        default="crawl-output",
        help="Directory for queue.txt and crawled.txt (default: crawl-output)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Worker thread count (default: 4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and show the crawl configuration without network or file activity",
    )
    return parser.parse_args()


def create_threads(thread_count: int) -> None:
    """Start crawl worker threads."""
    for _ in range(thread_count):
        worker = threading.Thread(target=work, daemon=True)
        worker.start()


def work() -> None:
    """Process queued URLs until the process exits."""
    while True:
        url = queue.get()
        Spider.crawl_page(threading.current_thread().name, url)
        queue.task_done()


def create_jobs() -> None:
    """Queue all known links and wait until workers finish them."""
    for link in file_to_set(Spider.queue_file):
        queue.put(link)
    queue.join()
    crawl()


def crawl() -> None:
    """Continue while persisted queue entries remain."""
    if file_to_set(Spider.queue_file):
        print(f"Links left: {len(file_to_set(Spider.queue_file))}")
        create_jobs()


def validate_args(args: argparse.Namespace) -> str:
    """Return the registrable domain after validating user input."""
    parsed = urlparse(args.url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL must include an http:// or https:// scheme and hostname.")
    if args.threads < 1:
        raise ValueError("Thread count must be at least one.")
    domain = get_domain_name(args.url)
    if not domain:
        raise ValueError("Could not determine a crawl domain from the URL.")
    return domain


def main() -> None:
    """Configure and run a crawl only when explicitly invoked."""
    args = parse_args()
    try:
        domain = validate_args(args)
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error

    if args.dry_run:
        print(
            f"Would crawl {args.url} within {domain} using {args.threads} workers "
            f"and write state to {args.project!r}."
        )
        return

    Spider(args.project, args.url, domain)
    create_threads(args.threads)
    crawl()


if __name__ == "__main__":
    main()

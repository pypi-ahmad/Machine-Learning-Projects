"""Search Gmail for attachments and download confirmed results locally."""

import ezgmail
import argparse
from pathlib import Path


def attachment_query(query: str) -> str:
    """Ensure a Gmail query only returns messages with attachments."""
    return query if "has:attachment" in query.casefold() else f"{query} has:attachment"


def download_attachments(threads: list, output_dir: Path, overwrite: bool) -> int:
    """Download every message attachment and return the number of messages handled."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for thread in threads:
        for message in thread.messages:
            message.downloadAllAttachments(
                downloadFolder=str(output_dir), overwrite=overwrite
            )
            count += 1
    return count


def print_subjects(threads: list) -> None:
    """Print the first-message subject for each matching Gmail thread."""
    for thread in threads:
        if thread.messages:
            print(f"Email subject: {thread.messages[0].subject}")


def main() -> None:
    """Search Gmail and download matching attachments after confirmation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="Gmail search query")
    parser.add_argument("--output-dir", type=Path, default=Path("downloads"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--yes", action="store_true", help="download without prompting")
    args = parser.parse_args()

    query = args.query or input("Enter Gmail search query: ").strip()
    if not query:
        raise SystemExit("Error: a Gmail search query is required.")

    threads = ezgmail.search(attachment_query(query))
    if not threads:
        print("No matching attachments found.")
        return

    print(f"Found {len(threads)} matching thread(s):")
    print_subjects(threads)
    if not args.yes:
        answer = input("Download these attachments? [y/N]: ").strip().casefold()
        if answer not in {"y", "yes"}:
            print("Download cancelled.")
            return

    output_dir = args.output_dir.expanduser().resolve()
    message_count = download_attachments(threads, output_dir, args.overwrite)
    print(f"Downloaded attachments from {message_count} message(s) to {output_dir}")


if __name__ == "__main__":
    main()

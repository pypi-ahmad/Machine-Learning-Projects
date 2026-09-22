"""Search and retrieve Wikipedia article content from the command line."""

from __future__ import annotations

import argparse

import wikipedia


def parse_args() -> argparse.Namespace:
    """Parse the topic, language, and output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("topic", help="Wikipedia article title or search topic")
    parser.add_argument("--language", default="en", help="Wikipedia language code (default: en)")
    parser.add_argument("--search", action="store_true", help="Print matching article titles")
    parser.add_argument("--suggest", action="store_true", help="Print Wikipedia's suggestion")
    parser.add_argument("--full", action="store_true", help="Print the full article content")
    parser.add_argument("--links", action="store_true", help="Print article links")
    parser.add_argument("--images", action="store_true", help="Print article image URLs")
    parser.add_argument(
        "--sentences",
        type=int,
        default=3,
        help="Summary sentence count (default: 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the selected request without contacting Wikipedia",
    )
    return parser.parse_args()


def print_page(page: wikipedia.WikipediaPage, args: argparse.Namespace) -> None:
    """Print article metadata and explicitly requested large fields."""
    print(f"Title: {page.title}")
    print(f"URL: {page.url}")
    if args.full:
        print(f"\n{page.content}")
    if args.links:
        print("\nLinks:")
        print("\n".join(page.links))
    if args.images:
        print("\nImages:")
        print("\n".join(page.images))


def main() -> None:
    """Run the selected Wikipedia request."""
    args = parse_args()
    if args.sentences < 1:
        raise SystemExit("Error: --sentences must be at least one.")
    if args.dry_run:
        print(f"Would query {args.topic!r} in the {args.language!r} Wikipedia edition.")
        return

    wikipedia.set_lang(args.language)
    try:
        if args.search:
            print("\n".join(wikipedia.search(args.topic)))
            return
        if args.suggest:
            suggestion = wikipedia.suggest(args.topic)
            print(suggestion or "No suggestion found.")
            return
        if args.full or args.links or args.images:
            print_page(wikipedia.page(args.topic, auto_suggest=False), args)
            return
        print(wikipedia.summary(args.topic, sentences=args.sentences, auto_suggest=False))
    except wikipedia.DisambiguationError as error:
        options = ", ".join(error.options[:10])
        raise SystemExit(f"Ambiguous topic. Try one of: {options}") from error
    except wikipedia.PageError as error:
        raise SystemExit(f"Article not found: {error}") from error
    except Exception as error:
        raise SystemExit(f"Wikipedia request failed: {error}") from error


if __name__ == "__main__":
    main()

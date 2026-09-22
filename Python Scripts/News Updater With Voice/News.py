"""Read NewsAPI headlines aloud with Windows text-to-speech.

Set NEWS_API_KEY before running this script. By default it reads one update;
pass --interval to repeat at a chosen number of seconds.
"""

import argparse
import os
import time

import pyttsx3
from newsapi import NewsApiClient


def create_speech_engine():
    """Create a SAPI5 engine and choose a second voice when available."""
    engine = pyttsx3.init("sapi5")
    voices = engine.getProperty("voices")
    if voices:
        engine.setProperty("voice", voices[min(1, len(voices) - 1)].id)
    return engine


def fetch_headlines(api_key: str, query: str, country: str, limit: int) -> list[str]:
    """Return non-empty NewsAPI descriptions or titles."""
    response = NewsApiClient(api_key=api_key).get_top_headlines(
        q=query,
        country=country,
        language="en",
        page_size=limit,
    )
    headlines = []
    for article in response.get("articles", []):
        headline = article.get("description") or article.get("title")
        if headline:
            headlines.append(headline)
    return headlines


def read_update(engine, api_key: str, query: str, country: str, limit: int) -> None:
    """Fetch, print, and speak one news update."""
    headlines = fetch_headlines(api_key, query, country, limit)
    if not headlines:
        print("No headlines were returned.")
        return

    for index, headline in enumerate(headlines, start=1):
        print(f"{index}. {headline}")
        engine.say(f"Headline {index}. {headline}")
    engine.say("That is the latest update.")
    engine.runAndWait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Read NewsAPI headlines aloud.")
    parser.add_argument("--query", default="corona", help="NewsAPI search query")
    parser.add_argument("--country", default="in", help="Two-letter country code")
    parser.add_argument("--limit", type=int, default=5, help="Headlines per update (1-100)")
    parser.add_argument(
        "--interval",
        type=int,
        help="Repeat after this many seconds; omit for one update",
    )
    args = parser.parse_args()

    if not 1 <= args.limit <= 100:
        parser.error("--limit must be between 1 and 100")
    if args.interval is not None and args.interval < 1:
        parser.error("--interval must be at least 1 second")

    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        parser.error("NEWS_API_KEY is required in the process environment")

    engine = create_speech_engine()
    try:
        while True:
            try:
                read_update(engine, api_key, args.query, args.country, args.limit)
            except Exception as error:
                print(f"Unable to fetch or read the update: {error}")
            if args.interval is None:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

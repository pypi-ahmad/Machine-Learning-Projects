"""Joke Fetcher — CLI tool.

Fetch jokes from the JokeAPI (https://v2.jokeapi.dev).
Supports categories, filtering, and saving favourites locally.
Falls back to built-in jokes when offline.

Usage:
    python main.py
"""

import json
import random
import urllib.error
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------------
# Built-in fallback jokes
# ---------------------------------------------------------------------------

FALLBACK_JOKES = [
    {"type": "twopart", "setup": "Why do programmers prefer dark mode?",
     "delivery": "Because light attracts bugs."},
    {"type": "twopart", "setup": "Why did the developer go broke?",
     "delivery": "Because he used up all his cache."},
    {"type": "single",
     "joke": "A SQL query walks into a bar, walks up to two tables and asks... 'Can I join you?'"},
    {"type": "twopart", "setup": "How many programmers does it take to change a light bulb?",
     "delivery": "None — that's a hardware problem."},
    {"type": "twopart", "setup": "Why do Java developers wear glasses?",
     "delivery": "Because they don't C#."},
    {"type": "single",
     "joke": "There are 10 types of people: those who understand binary and those who don't."},
    {"type": "twopart", "setup": "What's a computer's favourite snack?",
     "delivery": "Microchips."},
    {"type": "twopart", "setup": "Why was the function sad?",
     "delivery": "It had too many arguments."},
    {"type": "single",
     "joke": "I would tell you a joke about UDP, but you might not get it."},
    {"type": "twopart", "setup": "Why don't scientists trust atoms?",
     "delivery": "Because they make up everything."},
    {"type": "twopart", "setup": "What did the ocean say to the beach?",
     "delivery": "Nothing — it just waved."},
    {"type": "twopart", "setup": "Why can't you explain puns to kleptomaniacs?",
     "delivery": "Because they always take things literally."},
]

FAVOURITES_FILE = Path("jokes_favourites.json")

CATEGORIES = ["Programming", "Misc", "Dark", "Pun", "Spooky", "Christmas"]
BASE_URL    = "https://v2.jokeapi.dev/joke"


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def fetch_joke(category: str = "Programming", lang: str = "en",
               safe: bool = True) -> dict | None:
    params = f"?lang={lang}"
    if safe:
        params += "&safe-mode"
    url = f"{BASE_URL}/{category}{params}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
            if not data.get("error"):
                return data
    except (urllib.error.URLError, json.JSONDecodeError, Exception):
        pass
    return None


def random_fallback() -> dict:
    return random.choice(FALLBACK_JOKES)


def format_joke(joke: dict) -> str:
    if joke.get("type") == "twopart":
        setup    = joke.get("setup",    "")
        delivery = joke.get("delivery", "")
        return f"  Q: {setup}\n  A: {delivery}"
    return f"  {joke.get('joke', joke.get('delivery', ''))}"


# ---------------------------------------------------------------------------
# Favourites
# ---------------------------------------------------------------------------

def load_favs() -> list[dict]:
    if FAVOURITES_FILE.exists():
        try:
            return json.loads(FAVOURITES_FILE.read_text())
        except Exception:
            pass
    return []


def save_fav(joke: dict) -> None:
    favs = load_favs()
    # Avoid duplicates by joke text
    key = joke.get("joke") or joke.get("setup") or ""
    if any((f.get("joke") or f.get("setup") or "") == key for f in favs):
        print("  Already in favourites.")
        return
    favs.append(joke)
    FAVOURITES_FILE.write_text(json.dumps(favs, indent=2))
    print("  Saved to favourites!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Joke Fetcher
------------
1. Random joke (Programming)
2. Choose category
3. View favourites
4. Clear favourites
0. Quit
"""


def main() -> None:
    print("Joke Fetcher  😄")
    last_joke: dict | None = None

    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice in ("1", "2"):
            if choice == "2":
                print(f"  Categories: {', '.join(CATEGORIES)}")
                cat = input("  Category (default Programming): ").strip().title() or "Programming"
                if cat not in CATEGORIES:
                    cat = "Programming"
            else:
                cat = "Programming"

            print(f"\n  Fetching {cat} joke...", end=" ", flush=True)
            joke = fetch_joke(cat)
            if not joke:
                print("(offline — using built-in)")
                joke = random_fallback()
            else:
                print("done.")

            print(f"\n{format_joke(joke)}\n")
            last_joke = joke

            fav = input("  Save to favourites? (y/n): ").strip().lower()
            if fav == "y" and last_joke:
                save_fav(last_joke)

        elif choice == "3":
            favs = load_favs()
            if not favs:
                print("  No favourites yet.")
            else:
                print(f"\n  {len(favs)} favourite(s):")
                for i, j in enumerate(favs, 1):
                    print(f"\n  [{i}] {format_joke(j)}")

        elif choice == "4":
            if FAVOURITES_FILE.exists():
                FAVOURITES_FILE.unlink()
                print("  Favourites cleared.")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()

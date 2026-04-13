"""Random Quote Generator — CLI tool.

Fetches random quotes from quotable.io API.
Falls back to a built-in quote bank when offline.
Filter by author or category, save favourites.

Usage:
    python main.py
    python main.py --author "Einstein"
"""

import argparse
import json
import urllib.request
from pathlib import Path

FAVOURITES_FILE = Path("quote_favourites.json")

FALLBACK_QUOTES = [
    {"content": "The only way to do great work is to love what you do.", "author": "Steve Jobs", "tags": ["work"]},
    {"content": "In the middle of every difficulty lies opportunity.", "author": "Albert Einstein", "tags": ["motivation"]},
    {"content": "Life is what happens when you're busy making other plans.", "author": "John Lennon", "tags": ["life"]},
    {"content": "The future belongs to those who believe in the beauty of their dreams.", "author": "Eleanor Roosevelt", "tags": ["motivation"]},
    {"content": "It is during our darkest moments that we must focus to see the light.", "author": "Aristotle", "tags": ["wisdom"]},
    {"content": "Whoever is happy will make others happy too.", "author": "Anne Frank", "tags": ["happiness"]},
    {"content": "Do not go where the path may lead; go instead where there is no path and leave a trail.", "author": "Ralph Waldo Emerson", "tags": ["inspiration"]},
    {"content": "You only live once, but if you do it right, once is enough.", "author": "Mae West", "tags": ["life"]},
    {"content": "In three words I can sum up everything I've learned about life: it goes on.", "author": "Robert Frost", "tags": ["life"]},
    {"content": "To be yourself in a world that is constantly trying to make you something else is the greatest accomplishment.", "author": "Ralph Waldo Emerson", "tags": ["wisdom"]},
    {"content": "Whether you think you can or you think you can't, you're right.", "author": "Henry Ford", "tags": ["motivation"]},
    {"content": "The best time to plant a tree was 20 years ago. The second best time is now.", "author": "Chinese Proverb", "tags": ["wisdom"]},
    {"content": "An unexamined life is not worth living.", "author": "Socrates", "tags": ["philosophy"]},
    {"content": "Spread love everywhere you go. Let no one ever come to you without leaving happier.", "author": "Mother Teresa", "tags": ["love"]},
    {"content": "When you reach the end of your rope, tie a knot in it and hang on.", "author": "Franklin D. Roosevelt", "tags": ["perseverance"]},
    {"content": "Always remember that you are absolutely unique, just like everyone else.", "author": "Margaret Mead", "tags": ["humor"]},
    {"content": "Do not dwell in the past, do not dream of the future, concentrate the mind on the present moment.", "author": "Buddha", "tags": ["wisdom"]},
    {"content": "Not everything that is faced can be changed, but nothing can be changed until it is faced.", "author": "James Baldwin", "tags": ["change"]},
    {"content": "It is never too late to be what you might have been.", "author": "George Eliot", "tags": ["motivation"]},
    {"content": "You miss 100% of the shots you don't take.", "author": "Wayne Gretzky", "tags": ["sports", "motivation"]},
]


def fetch_quote(author: str = "") -> dict | None:
    url = "https://api.quotable.io/random"
    if author:
        url += f"?author={urllib.parse.quote(author)}"
    try:
        with urllib.request.urlopen(url, timeout=4) as resp:
            data = json.loads(resp.read())
            return {"content": data["content"], "author": data["author"],
                    "tags": data.get("tags", [])}
    except Exception:
        return None


def load_favourites() -> list[dict]:
    if FAVOURITES_FILE.exists():
        try:
            return json.loads(FAVOURITES_FILE.read_text())
        except Exception:
            pass
    return []


def save_favourites(favs: list[dict]):
    FAVOURITES_FILE.write_text(json.dumps(favs, indent=2))


def display_quote(q: dict, num: int | None = None):
    prefix = f"[{num}] " if num else ""
    print(f'\n  {prefix}"{q["content"]}"')
    print(f'     — {q["author"]}')
    if q.get("tags"):
        print(f'     Tags: {", ".join(q["tags"])}')


def main():
    import urllib.parse
    parser = argparse.ArgumentParser(description="Random Quote Generator")
    parser.add_argument("--author", default="", help="Filter by author name")
    args = parser.parse_args()

    import random
    favs = load_favourites()

    if args.author:
        q = fetch_quote(args.author) or random.choice([
            x for x in FALLBACK_QUOTES if args.author.lower() in x["author"].lower()
        ] or FALLBACK_QUOTES)
        display_quote(q)
        return

    print("Random Quote Generator")
    print("──────────────────────────────")
    print("  n → new quote   f → favourites   s → save   q → quit")

    current: dict | None = None

    while True:
        cmd = input("\n> ").strip().lower()

        if cmd in ("", "n"):
            q = fetch_quote()
            if q is None:
                q = random.choice(FALLBACK_QUOTES)
                print("  (offline — using built-in quotes)")
            current = q
            display_quote(q)

        elif cmd == "s":
            if current:
                if current not in favs:
                    favs.append(current)
                    save_favourites(favs)
                    print("  Saved to favourites.")
                else:
                    print("  Already in favourites.")
            else:
                print("  No quote to save. Press 'n' first.")

        elif cmd == "f":
            if not favs:
                print("  No favourites yet.")
            else:
                print(f"\n  {len(favs)} favourite(s):")
                for i, q in enumerate(favs, 1):
                    display_quote(q, i)
                sub = input("\n  del <n> to remove, or Enter to go back: ").strip()
                if sub.startswith("del"):
                    try:
                        idx = int(sub.split()[1]) - 1
                        removed = favs.pop(idx)
                        save_favourites(favs)
                        print(f"  Removed: {removed['author']}")
                    except (IndexError, ValueError):
                        print("  Invalid.")

        elif cmd in ("q", "quit"):
            print("Bye!")
            break

        else:
            print("  Commands: n=new  f=favourites  s=save  q=quit")


if __name__ == "__main__":
    main()

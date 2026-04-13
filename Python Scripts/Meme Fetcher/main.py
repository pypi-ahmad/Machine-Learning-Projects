"""Meme Fetcher — CLI tool.

Fetch random memes from the Meme API (no key needed).
Filter by subreddit, save favorites, and show meme URLs.

Usage:
    python main.py
    python main.py --count 5
    python main.py --sub ProgrammerHumor
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path


MEME_API     = "https://meme-api.com/gimme"
FAVORITES    = Path("meme_favorites.json")

POPULAR_SUBS = [
    "memes", "dankmemes", "me_irl", "ProgrammerHumor",
    "AdviceAnimals", "terriblefacebookmemes", "OldSchoolCool",
    "wholesomememes", "Showerthoughts",
]


def fetch(url: str) -> dict | list:
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            return json.loads(resp.read())
    except Exception as e:
        raise ValueError(f"Network error: {e}")


def get_meme(subreddit: str = None) -> dict:
    url = MEME_API if not subreddit else f"{MEME_API}/{subreddit}"
    return fetch(url)


def get_memes(count: int, subreddit: str = None) -> list[dict]:
    url = f"{MEME_API}/{count}" if not subreddit else f"{MEME_API}/{subreddit}/{count}"
    data = fetch(url)
    return data.get("memes", [data]) if "memes" in data else [data]


def display_meme(m: dict, idx: int = 1) -> None:
    title  = m.get("title","—")[:80]
    sub    = m.get("subreddit","—")
    author = m.get("author","—")
    ups    = m.get("ups", 0)
    url    = m.get("url","—")
    nsfw   = "⚠️ NSFW" if m.get("nsfw") else ""

    print(f"\n  {'─'*60}")
    print(f"  [{idx}]  {title}")
    print(f"  r/{sub}  ·  u/{author}  ·  ⬆ {ups:,}  {nsfw}")
    print(f"  🔗  {url}")


def load_favorites() -> list:
    if FAVORITES.exists():
        try: return json.loads(FAVORITES.read_text())
        except: pass
    return []


def save_favorite(meme: dict) -> None:
    favs = load_favorites()
    url  = meme.get("url","")
    if not any(f.get("url") == url for f in favs):
        favs.append({"title": meme.get("title",""), "url": url,
                     "sub": meme.get("subreddit","")})
        FAVORITES.write_text(json.dumps(favs, indent=2))
        print("  ⭐ Saved!")


def interactive():
    print("=== Meme Fetcher ===")
    print(f"Popular subreddits: {', '.join(POPULAR_SUBS[:5])}…")
    print("Commands: get | sub | subs | favorites | quit\n")
    last_memes: list[dict] = []

    while True:
        cmd = input("> ").strip().lower()
        if cmd in ("quit", "q", "exit"): break

        elif cmd == "get":
            n = int(input("  How many memes? [3]: ").strip() or 3)
            try:
                memes = get_memes(min(n, 10))
                last_memes = memes
                for i, m in enumerate(memes, 1):
                    display_meme(m, i)
                print()
                if input("  Save a meme? Enter number (or blank): ").strip().isdigit():
                    idx = int(input("  Number: ").strip()) - 1
                    if 0 <= idx < len(memes): save_favorite(memes[idx])
            except ValueError as e: print(f"  Error: {e}")

        elif cmd == "sub":
            sub = input("  Subreddit name: ").strip()
            n   = int(input("  How many? [3]: ").strip() or 3)
            try:
                memes = get_memes(min(n, 10), sub)
                last_memes = memes
                for i, m in enumerate(memes, 1):
                    display_meme(m, i)
            except ValueError as e: print(f"  Error: {e}")

        elif cmd == "subs":
            print(f"  Popular subreddits: {', '.join(POPULAR_SUBS)}")

        elif cmd == "favorites":
            favs = load_favorites()
            if not favs: print("  No favorites yet.")
            else:
                print(f"  {len(favs)} saved meme(s):")
                for i, f in enumerate(favs, 1):
                    print(f"  [{i}]  {f['title'][:60]}  (r/{f.get('sub','')})")
                    print(f"       {f['url']}")

        else:
            print("  Commands: get | sub <subreddit> | subs | favorites | quit")
        print()


def main():
    parser = argparse.ArgumentParser(description="Meme Fetcher")
    parser.add_argument("--count", type=int, default=3, help="Number of memes")
    parser.add_argument("--sub",   metavar="SUBREDDIT",  help="Subreddit to fetch from")
    args = parser.parse_args()

    try:
        if args.sub or args.count != 3:
            memes = get_memes(min(args.count, 10), args.sub)
            for i, m in enumerate(memes, 1):
                display_meme(m, i)
        else:
            interactive()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    main()

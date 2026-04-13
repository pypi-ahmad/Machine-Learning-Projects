"""Book Search — CLI tool.

Search for books using the Open Library API (no key required).
View details, editions, ratings, and save favorites.

Usage:
    python main.py
    python main.py --search "dune"
    python main.py --isbn 9780441013593
"""

import argparse
import json
import sys
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path


FAVORITES_FILE = Path("book_favorites.json")
COVERS_BASE    = "https://covers.openlibrary.org/b/isbn/{}-M.jpg"


def api_get(url: str) -> dict | list:
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise ValueError(f"HTTP {e.code}")
    except Exception as e:
        raise ValueError(f"Network error: {e}")


def search_books(query: str, limit: int = 10) -> list[dict]:
    q   = urllib.parse.quote(query)
    url = f"https://openlibrary.org/search.json?q={q}&limit={limit}"
    data = api_get(url)
    return data.get("docs", [])


def display_search(docs: list[dict]) -> None:
    if not docs:
        print("  No results found.")
        return
    print(f"\n  {'─'*60}")
    for i, d in enumerate(docs):
        title   = d.get("title", "Unknown")[:50]
        authors = ", ".join(d.get("author_name", ["Unknown"])[:2])
        year    = d.get("first_publish_year", "—")
        key     = d.get("key", "")
        pages   = d.get("number_of_pages_median", "—")
        rating  = d.get("ratings_average", None)
        rating_s = f"  ⭐{rating:.1f}" if rating else ""
        print(f"  [{i+1:>2}]  {title}")
        print(f"        by {authors}  ({year})  {pages} pages{rating_s}")
        print(f"        Key: {key}")
    print()


def get_book_detail(work_key: str) -> dict:
    url = f"https://openlibrary.org{work_key}.json"
    return api_get(url)


def display_detail(doc: dict) -> None:
    d = get_book_detail(doc.get("key", ""))
    title   = d.get("title", doc.get("title",""))
    authors = ", ".join(doc.get("author_name", []))
    desc    = d.get("description", "")
    if isinstance(desc, dict): desc = desc.get("value", "")
    subjects = d.get("subjects", [])[:8]

    print(f"\n  📖  {title}")
    print(f"  ✍️   {authors}")
    if doc.get("first_publish_year"):
        print(f"  📅  First published: {doc['first_publish_year']}")
    if doc.get("number_of_pages_median"):
        print(f"  📏  Pages: {doc['number_of_pages_median']}")
    if doc.get("ratings_average"):
        print(f"  ⭐  Rating: {doc['ratings_average']:.2f}/5 ({doc.get('ratings_count',0):,} ratings)")
    if subjects:
        print(f"  🏷   Subjects: {', '.join(subjects)}")
    if desc:
        print(f"\n  📝  Description:")
        print(f"  {desc[:400]}")
    print()


def lookup_isbn(isbn: str) -> dict:
    url = f"https://openlibrary.org/isbn/{isbn}.json"
    data = api_get(url)
    return data


def display_isbn(data: dict) -> None:
    print(f"\n  📖  {data.get('title','—')}")
    authors = data.get("authors", [])
    if authors:
        print(f"  ✍️   {', '.join(a.get('key','') for a in authors)}")
    print(f"  📅  Published: {data.get('publish_date','—')}")
    print(f"  📏  Pages: {data.get('number_of_pages','—')}")
    print(f"  🏷   ISBN: {', '.join(data.get('isbn_13',[]) + data.get('isbn_10',[]))}")
    print()


def load_favorites() -> list:
    if FAVORITES_FILE.exists():
        try: return json.loads(FAVORITES_FILE.read_text())
        except: pass
    return []


def save_favorite(doc: dict) -> None:
    favs = load_favorites()
    key  = doc.get("key","")
    if not any(f.get("key") == key for f in favs):
        favs.append({"key": key, "title": doc.get("title",""),
                     "author": ", ".join(doc.get("author_name",[]))})
        FAVORITES_FILE.write_text(json.dumps(favs, indent=2))
        print(f"  ⭐ Saved to favorites!")


def interactive():
    print("=== Book Search ===")
    print("Commands: search | isbn | favorites | quit\n")
    last_results: list[dict] = []

    while True:
        cmd = input("> ").strip().lower()
        if cmd in ("quit", "q", "exit"): break
        elif cmd == "search":
            q = input("  Search query: ").strip()
            if not q: continue
            try:
                docs = search_books(q)
                last_results = docs
                display_search(docs)
                if docs:
                    sel = input("  Enter number for details (or blank to skip): ").strip()
                    if sel.isdigit() and 1 <= int(sel) <= len(docs):
                        book = docs[int(sel)-1]
                        try: display_detail(book)
                        except: pass
                        if input("  Save to favorites? (y/n): ").strip().lower() == "y":
                            save_favorite(book)
            except ValueError as e: print(f"  Error: {e}")
        elif cmd == "isbn":
            isbn = input("  ISBN: ").strip().replace("-","").replace(" ","")
            try:
                data = lookup_isbn(isbn)
                display_isbn(data)
            except ValueError as e: print(f"  Error: {e}")
        elif cmd == "favorites":
            favs = load_favorites()
            if not favs: print("  No favorites yet.")
            else:
                print(f"  {len(favs)} favorite(s):")
                for f in favs:
                    print(f"  ⭐  {f['title']} — {f['author']}")
        else:
            print("  Commands: search | isbn | favorites | quit")
        print()


def main():
    parser = argparse.ArgumentParser(description="Book Search via Open Library")
    parser.add_argument("--search", metavar="QUERY", help="Search query")
    parser.add_argument("--isbn",   metavar="ISBN",  help="Look up by ISBN")
    parser.add_argument("--limit",  type=int, default=10)
    args = parser.parse_args()

    try:
        if args.search:
            docs = search_books(args.search, args.limit)
            display_search(docs)
        elif args.isbn:
            data = lookup_isbn(args.isbn)
            display_isbn(data)
        else:
            interactive()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    main()

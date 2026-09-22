# Book Search

Search Open Library by title or ISBN and optionally save local favorites.

```powershell
cd "Python Scripts/Book Search"
uv sync
uv run python main.py --search "dune"
```

Favorites are stored beside the script in `book_favorites.json`, which is
excluded from Git. Live Open Library requests are required for search results.

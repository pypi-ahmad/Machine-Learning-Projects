# Instagram Scraper

Inspect public profile metadata or explicitly download a bounded number of posts. The default command is local preview only and makes no Instagram request.

```powershell
uv sync --no-config
uv run --no-config python "Instagram Scapper In Python.py" example_account
```

Request public metadata deliberately:

```powershell
uv run --no-config python "Instagram Scapper In Python.py" example_account --metadata
```

Downloads require a finite limit, exact confirmation, and a new output directory:

```powershell
uv run --no-config python "Instagram Scapper In Python.py" example_account `
  --download-posts --max-posts 1 --output downloads\example_account `
  --confirm DOWNLOAD_POSTS
```

Use only for content you are authorized to retrieve. Instagram may limit automated access; live requests and downloads were not exercised during local verification.

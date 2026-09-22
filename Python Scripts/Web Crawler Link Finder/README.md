# Web Crawler Link Finder

A standard-library crawler that follows HTML links within the target domain and writes the pending and crawled URLs to local text files.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)

The crawler has no third-party runtime dependencies.

## Run

Start with a dry-run to confirm the target and output directory without making network requests or creating files:

```powershell
cd "Python Scripts\Web Crawler Link Finder"
uv run python .\main.py "https://example.com" --project .\example-crawl --dry-run
```

Run a crawl with a small worker count:

```powershell
uv run python .\main.py "https://example.com" --project .\example-crawl --threads 2
```

The project directory contains:

- `queue.txt` for links waiting to be crawled.
- `crawled.txt` for links already processed.

## Behavior and limits

- Only links whose URL contains the target’s derived domain are added to the queue.
- Relative links are resolved against the page that contains them.
- This is a basic crawler. It does not interpret JavaScript, enforce `robots.txt`, impose a page limit, or provide a crawl-delay setting.
- Crawl only sites you own or are authorized to crawl, and use low worker counts to reduce load.

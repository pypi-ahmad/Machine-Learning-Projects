# Instagram Profile

> Scrapes Instagram profile information and downloads profile pictures using HTTP requests and HTML parsing.

## Overview

A collection of scripts for viewing public Instagram Open Graph profile metadata and optionally downloading the public profile image. `InstgramProfile.py` remains as a compatible entry point.

## Features

- Fetch public profile title, description, and image URL
- Download public profile pictures with progress bar
- Optional image download; no automatic image display
- CLI interface accepting username as argument
- Pretty-printed output using `pprint`
- One maintained implementation (`main.py`) with a compatible legacy entry point

## Project Structure

```
Instagram_profile/
├── main.py              # Enhanced scraper + profile pic download
├── InstgramProfile.py   # Standalone profile scraper
├── profilepic.py        # HD profile picture downloader
├── pyproject.toml       # Project metadata and dependencies
├── uv.lock              # Locked dependency versions
├── output.png           # Sample output screenshot
└── Readme.md
```

## Requirements

- Python 3.13+
- `requests`, `lxml`, and `tqdm`, managed by uv in `pyproject.toml`

## Installation

```bash
cd Instagram_profile
uv sync
```

## Usage

```bash
# View public metadata
uv run python main.py <username>

# View metadata and save the profile image to a new path
uv run python main.py <username> --download-image profile.jpg

# Compatible legacy entry point
uv run python InstgramProfile.py <username>
```

**Example:**
```bash
uv run python main.py cristiano
```

**Output:**
```python
{'description': '...', 'name': '...', 'profile_image_url': '...', 'username': '...'}
```

## How It Works

### Profile Scraping (`main.py` / `InstgramProfile.py`)

1. Validates the username and requests the public profile page with a timeout.
2. Uses `lxml.html` XPath to read available Open Graph title, description, and image metadata.
3. Returns the public metadata or reports an unavailable profile.

### Profile Picture Download (`profilepic.py`)

1. Receives the public Open Graph image URL from the profile parser.
2. Downloads the image with a bounded request and `tqdm` progress bar.
3. Saves to the explicitly requested new path without opening an image viewer.

## Configuration

- **Profile URL**: Instagram's public profile pages at `instagram.com/<username>/?hl=en`.
- No authentication required (relies on public page meta tags).

## Limitations

- Relies on Instagram's public HTML structure and meta tags — may break if Instagram changes their page layout.
- Instagram can change or restrict public page metadata at any time.
- Private or unavailable profiles may not expose metadata or a downloadable image.
- Existing image paths are not overwritten.

## Security Notes

- No credentials are required or stored — relies on publicly accessible data.
- No security concerns identified.

## License

Not specified.

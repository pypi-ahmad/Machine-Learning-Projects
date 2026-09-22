# Facebook DP Downloader

> Downloads the profile picture of a public Facebook profile using its numeric user ID.

## Overview

This script uses the Facebook Graph API to fetch and save a user's large profile picture from a numeric user ID. Images are saved in a local `fb_dps` folder.

## Features

- Fetches profile pictures via the Facebook Graph API (`graph.facebook.com`)
- Downloads the large-size variant of the profile picture
- Automatically creates a `fb_dps` output directory if it doesn't exist
- Saves images as `{facebook_id}_img.jpg`
- Interactive prompt or command-line argument for Facebook user IDs

## Project Structure

```
Facebook-DP-Downloader/
├── fb_dp_downloader.py   # Main script to download Facebook profile pictures
├── pyproject.toml        # Project metadata and dependencies
└── uv.lock               # Locked dependency versions
```

## Requirements

- Python 3.13+
- `requests`, managed by uv in `pyproject.toml`

## Installation

```bash
cd "Facebook-DP-Downloader"
uv sync
```

The uv project correctly treats `os` as part of Python and installs `requests`.

## Usage

```bash
uv run python fb_dp_downloader.py
```

Or provide an ID and output directory directly:

```bash
uv run python fb_dp_downloader.py 4 --output-dir fb_dps
```

When prompted, enter a valid numeric Facebook user ID:

```
Enter the Facebook-id to download it's profile picture: 4
```

The profile picture will be saved to `fb_dps/4_img.jpg`.

## How it works

1. Constructs a URL using the Facebook Graph API: `https://graph.facebook.com/{id}/picture?type=large`
2. Validates the numeric Facebook user ID.
3. Sends a bounded GET request to the Graph API URL.
4. Verifies that the response is an image.
5. Creates the output directory if needed and writes the image to `{id}_img.jpg`.

## Configuration

No configuration files. Use `--output-dir` to select the destination directory.

## Limitations

- Only works for **public** profiles without profile picture guard enabled
- Facebook user IDs below 4 do not correspond to valid profiles
- The Graph API endpoint may require authentication or may be rate-limited by Facebook

## Security Notes

No sensitive credentials in the code. However, the Facebook Graph API may deprecate or restrict this unauthenticated endpoint at any time.

## License

Not specified.

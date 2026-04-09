# Facebook DP Downloader

> Downloads the profile picture of any public Facebook profile using its numeric Facebook user ID.

## Overview

This script uses the Facebook Graph API endpoint to fetch and save the profile picture (large size) of a Facebook user, given their numeric user ID. Downloaded images are saved to a local `fb_dps` folder.

## Features

- Fetches profile pictures via the Facebook Graph API (`graph.facebook.com`)
- Downloads the large-size variant of the profile picture
- Automatically creates a `fb_dps` output directory if it doesn't exist
- Saves images as `{facebook_id}_img.jpg`
- Interactive prompt for entering Facebook user IDs

## Project Structure

```
Facebook-DP-Downloader/
├── fb_dp_downloader.py   # Main script to download Facebook profile pictures
└── requirements.txt      # Python dependencies
```

## Requirements

- Python 3.x
- `requests`

## Installation

```bash
cd "Facebook-DP-Downloader"
pip install requests
```

> **Note:** The `requirements.txt` lists `os` and `request`, but `os` is a built-in module and the correct package name is `requests` (with an 's').

## Usage

```bash
python fb_dp_downloader.py
```

When prompted, enter a valid numeric Facebook user ID:

```
Enter the Facebook-id to download it's profile picture: 4
```

The profile picture will be saved to `fb_dps/4_img.jpg`.

## How It Works

1. Constructs a URL using the Facebook Graph API: `https://graph.facebook.com/{id}/picture?type=large`
2. Checks if a `fb_dps` directory exists in the current working directory; creates it if not.
3. Prompts the user for a numeric Facebook user ID.
4. Sends a GET request to the Graph API URL.
5. Writes the response content (image bytes) to `fb_dps/{id}_img.jpg`.

## Configuration

No configuration files. The Graph API URL is hardcoded in the script.

## Limitations

- Only works for **public** profiles without profile picture guard enabled
- Facebook user IDs below 4 do not correspond to valid profiles
- The Graph API endpoint may require authentication or may be rate-limited by Facebook
- Bare `except` clause catches all errors with a generic message
- The `int()` cast on input will crash on non-numeric input with an unhandled `ValueError`
- The `requirements.txt` is incorrect (`os` is not a pip package, `request` should be `requests`)

## Security Notes

No sensitive credentials in the code. However, the Facebook Graph API may deprecate or restrict this unauthenticated endpoint at any time.

## License

Not specified.

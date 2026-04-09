# Instagram Profile

> Scrapes Instagram profile information and downloads profile pictures using HTTP requests and HTML parsing.

## Overview

A collection of scripts for fetching Instagram profile metadata (followers, following, posts, bio) and downloading profile pictures in HD. The project includes two versions of the profile scraper (`main.py` with profile picture download, `InstgramProfile.py` standalone) and a dedicated profile picture downloader (`profilepic.py`).

## Features

- Fetch profile details: name, username, followers, following, post count, bio
- Download HD profile pictures with progress bar
- Automatic profile picture display after download
- CLI interface accepting username as argument
- Pretty-printed output using `pprint`
- Two scraper versions: basic (`InstgramProfile.py`) and enhanced (`main.py` with profile picture download)

## Project Structure

```
Instagram_profile/
├── main.py              # Enhanced scraper + profile pic download
├── InstgramProfile.py   # Standalone profile scraper
├── profilepic.py        # HD profile picture downloader
├── requirements.txt     # Dependencies
├── output.png           # Sample output screenshot
└── Readme.md
```

## Requirements

- Python 3.x
- `requests`
- `lxml`
- `Pillow` (PIL)
- `tqdm`

### requirements.txt

```
appdirs, beautifulsoup4, bs4, CacheControl, certifi, chardet, colorama,
contextlib2, distlib, distro, html5lib, idna, ipaddr, lockfile, lxml,
msgpack, packaging, pep517, Pillow, progress, pyparsing, pytoml,
requests, retrying, six, soupsieve, tqdm, urllib3, webencodings
```

## Installation

```bash
cd Instagram_profile
pip install requests lxml Pillow tqdm
```

## Usage

```bash
# Enhanced version (scrapes profile + downloads profile picture)
python main.py <username>

# Basic version (scrapes profile info only)
python InstgramProfile.py <username>
```

**Example:**
```bash
python main.py cristiano
```

**Output:**
```python
{'profile': {'aboutinfo': '...', 'followers': '...', 'following': '...',
             'name': '...', 'posts': '...', 'profileurl': '...', 'username': '...'},
 'success': True}
```

## How It Works

### Profile Scraping (`main.py` / `InstgramProfile.py`)

1. Constructs the Instagram profile URL: `https://www.instagram.com/<username>/?hl=en`.
2. Fetches the page HTML via `requests.get()`.
3. Uses `lxml.html` XPath to extract the `<meta name="description">` tag content containing follower/following/post counts.
4. Parses the counts and name from the HTML using regular expressions.
5. Returns a dictionary with profile details or `{'success': False}` if the profile is not found.

### Profile Picture Download (`profilepic.py`)

1. Constructs the Instagram profile URL and appends `?__a=1` to get JSON data.
2. Validates the URL format using multiple regex patterns.
3. Fetches the page and extracts `profile_pic_url_hd` using regex.
4. Downloads the image with a `tqdm` progress bar and saves as `<username>.jpg`.
5. Opens the image using `PIL.Image.show()`.

## Configuration

- **Profile URL**: Instagram's public profile pages at `instagram.com/<username>/?hl=en`.
- No authentication required (relies on public page meta tags).

## Limitations

- Relies on Instagram's public HTML structure and meta tags — may break if Instagram changes their page layout.
- The `?__a=1` JSON endpoint used in `profilepic.py` has been deprecated by Instagram and may not work.
- `main.py` and `InstgramProfile.py` have largely duplicated code.
- Regex-based parsing is fragile and may fail on unusual profile data.
- No error handling for network failures or private profiles.
- `requirements.txt` includes many packages not directly used by the project.

## Security Notes

- No credentials are required or stored — relies on publicly accessible data.
- No security concerns identified.

## License

Not specified.

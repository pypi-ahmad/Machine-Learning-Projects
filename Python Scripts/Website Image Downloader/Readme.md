# Website image downloader

`scrap-img.py` downloads images referenced by a web page's server-rendered HTML.
It reads `src` and `data-src` attributes from `<img>` tags, resolves relative
URLs, and saves the image responses to a new local directory.

## Install

```powershell
cd "Python Scripts/Website Image Downloader"
uv sync
```

The script uses only the Python standard library, so no third-party packages
are installed.

## Usage

```powershell
uv run python scrap-img.py https://example.com --output downloaded_images --limit 20
```

The default limit is 25 images. Each saved file is named in source order, such
as `image_0001.jpg`. File extensions come from the response content type.

Request controls are available when a page or its images need different limits:

```powershell
uv run python scrap-img.py https://example.com --output downloaded_images --timeout 20 --page-max-bytes 500000 --image-max-bytes 5000000
```

## Safety and limits

- Only absolute `http` and `https` page URLs are accepted.
- The output directory must be new. Existing directories are never cleared or
  reused.
- Each response must identify itself as HTML or an image as appropriate.
- Failed image downloads are reported and do not stop later images.

The tool does not run JavaScript, log in, bypass bot checks, or find CSS
background images. Pages that populate images in the browser may return no
downloadable images here.

Run `uv run python scrap-img.py --help` for the command reference.

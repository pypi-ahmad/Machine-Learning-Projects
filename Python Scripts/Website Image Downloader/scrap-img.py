"""Download images referenced by the server-rendered HTML of one web page."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen


USER_AGENT = "WebsiteImageDownloader/1.0"
CONTENT_EXTENSIONS = {
    "image/gif": ".gif",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


class ImageParser(HTMLParser):
    """Collect image source attributes from HTML img tags."""

    def __init__(self) -> None:
        super().__init__()
        self.sources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "img":
            return
        attributes = dict(attrs)
        source = attributes.get("src") or attributes.get("data-src")
        if source:
            self.sources.append(source)


def valid_url(value: str) -> str:
    """Accept only absolute HTTP(S) page URLs."""
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise argparse.ArgumentTypeError("URL must use http or https and include a host.")
    return value


def request(url: str, timeout: float):
    """Open a URL with a descriptive user agent."""
    return urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout)


def image_urls(html: str, page_url: str) -> list[str]:
    """Return unique absolute image URLs in source order."""
    parser = ImageParser()
    parser.feed(html)
    parser.close()
    urls: list[str] = []
    for source in parser.sources:
        absolute_url = urljoin(page_url, source)
        if absolute_url.startswith(("http://", "https://")) and absolute_url not in urls:
            urls.append(absolute_url)
    return urls


def fetch_page(url: str, timeout: float, max_bytes: int) -> str:
    """Fetch bounded HTML from the source page."""
    with request(url, timeout) as response:
        content_type = response.headers.get_content_type()
        if content_type not in {"text/html", "application/xhtml+xml"}:
            raise ValueError(f"Expected HTML, received {content_type}.")
        content = response.read(max_bytes + 1)
        encoding = response.headers.get_content_charset() or "utf-8"
    if len(content) > max_bytes:
        raise ValueError(f"Page exceeds the {max_bytes} byte limit.")
    return content.decode(encoding, errors="replace")


def download_image(url: str, output_path: Path, timeout: float, max_bytes: int) -> str:
    """Download one image and return the extension chosen from its content type."""
    with request(url, timeout) as response:
        content_type = response.headers.get_content_type()
        if not content_type.startswith("image/"):
            raise ValueError(f"Expected an image, received {content_type}.")
        content = response.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError(f"Image exceeds the {max_bytes} byte limit.")
    extension = CONTENT_EXTENSIONS.get(content_type, ".img")
    output_path.with_suffix(extension).write_bytes(content)
    return extension


def download_images(page_url: str, output_directory: Path, limit: int, timeout: float, page_max_bytes: int, image_max_bytes: int) -> tuple[int, int]:
    """Download selected page images and return saved and failed counts."""
    if output_directory.exists():
        raise FileExistsError(f"{output_directory} already exists; choose a new output directory.")
    sources = image_urls(fetch_page(page_url, timeout, page_max_bytes), page_url)
    output_directory.mkdir(parents=True)
    saved = 0
    failed = 0
    for source in sources[:limit]:
        try:
            download_image(source, output_directory / f"image_{saved + failed + 1:04d}", timeout, image_max_bytes)
            saved += 1
        except (HTTPError, OSError, URLError, ValueError) as error:
            failed += 1
            print(f"Skipped {source}: {error}")
    return saved, failed


def main() -> None:
    """Fetch one page, download its image sources, and report the result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", type=valid_url, help="web page URL")
    parser.add_argument("--output", type=Path, default=Path("output"), help="new image directory")
    parser.add_argument("--limit", type=int, default=25, help="maximum images to attempt (default: 25)")
    parser.add_argument("--timeout", type=float, default=15, help="request timeout in seconds (default: 15)")
    parser.add_argument("--page-max-bytes", type=int, default=2_000_000, help="maximum page size (default: 2000000)")
    parser.add_argument("--image-max-bytes", type=int, default=10_000_000, help="maximum image size (default: 10000000)")
    args = parser.parse_args()
    if args.limit < 1 or args.timeout <= 0 or args.page_max_bytes < 1 or args.image_max_bytes < 1:
        parser.error("--limit, --timeout, --page-max-bytes, and --image-max-bytes must be positive.")

    try:
        saved, failed = download_images(
            args.url,
            args.output,
            args.limit,
            args.timeout,
            args.page_max_bytes,
            args.image_max_bytes,
        )
    except (HTTPError, OSError, URLError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
    print(f"Saved {saved} image(s); skipped {failed}.")


if __name__ == "__main__":
    main()

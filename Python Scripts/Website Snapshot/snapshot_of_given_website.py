"""Save a full-page PNG snapshot of an HTTP(S) website."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
from urllib.parse import urlsplit

from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options


def valid_url(value: str) -> str:
    """Accept only absolute HTTP(S) URLs."""
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise argparse.ArgumentTypeError("URL must use http or https and include a host.")
    return value


def chrome_options() -> Options:
    """Create quiet headless Chrome options suitable for screenshots."""
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1440,1200")
    return options


def capture_snapshot(url: str, output: Path, timeout: float, driver_factory=webdriver.Chrome) -> None:
    """Capture a full-page PNG through Chrome DevTools without overwriting files."""
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose a new output path.")
    driver = driver_factory(options=chrome_options())
    try:
        driver.set_page_load_timeout(timeout)
        driver.get(url)
        result = driver.execute_cdp_cmd("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True})
        output.write_bytes(base64.b64decode(result["data"], validate=True))
    finally:
        driver.quit()


def main() -> None:
    """Parse snapshot arguments and save the requested page image."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", type=valid_url, help="website URL")
    parser.add_argument("--output", type=Path, default=Path("screenshot.png"), help="new PNG path")
    parser.add_argument("--timeout", type=float, default=30, help="page-load timeout in seconds (default: 30)")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive.")
    try:
        capture_snapshot(args.url, args.output, args.timeout)
    except (OSError, ValueError, WebDriverException) as error:
        raise SystemExit(f"Error: {error}") from error
    print(f"Saved {args.output}.")


if __name__ == "__main__":
    main()

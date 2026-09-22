"""Preview or explicitly download images visible on an Instagram profile."""

from __future__ import annotations

import argparse
from getpass import getpass
from pathlib import Path

from bs4 import BeautifulSoup
import requests


CONFIRMATION = "DOWNLOAD_IMAGES"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview or download visible Instagram profile images.")
    parser.add_argument("profile", help="Instagram profile username.")
    parser.add_argument("--output", type=Path, help="New destination directory.")
    parser.add_argument("--max-images", type=int, default=20, help="Maximum images to download (default: 20).")
    parser.add_argument("--apply", action="store_true", help="Open Chrome and download image files.")
    parser.add_argument("--confirm", help=f"Required with --apply: {CONFIRMATION}")
    return parser.parse_args()


def validate_profile(profile: str) -> str:
    profile = profile.strip().lstrip("@")
    if not profile.replace("_", "").replace(".", "").isalnum():
        raise ValueError("Profile names may contain letters, numbers, underscores, and periods only.")
    return profile


def visible_image_urls(page_source: str, limit: int) -> list[str]:
    """Extract distinct HTTPS image URLs from already-loaded profile HTML."""
    if limit <= 0:
        raise ValueError("--max-images must be greater than zero.")
    urls = [image.get("src") for image in BeautifulSoup(page_source, "html.parser").select("img[src]")]
    return list(dict.fromkeys(url for url in urls if url and url.startswith("https://")))[:limit]


def download_images(profile: str, output: Path, limit: int) -> int:
    """Log in interactively, collect visible images, and save them into a new directory."""
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    username = input("Instagram username: ").strip()
    password = getpass("Instagram password: ")
    if not username or not password:
        raise ValueError("Instagram username and password are required.")

    driver = webdriver.Chrome()
    try:
        wait = WebDriverWait(driver, 30)
        driver.get("https://www.instagram.com/")
        wait.until(EC.presence_of_element_located((By.NAME, "username"))).send_keys(username)
        driver.find_element(By.NAME, "password").send_keys(password)
        driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
        driver.get(f"https://www.instagram.com/{profile}/")
        urls = visible_image_urls(driver.page_source, limit)
    finally:
        driver.quit()

    if not urls:
        raise ValueError("No visible image URLs were found.")
    output.mkdir(parents=True)
    for index, url in enumerate(urls, start=1):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        (output / f"image-{index:03}.jpg").write_bytes(response.content)
    return len(urls)


def main() -> None:
    args = parse_args()
    try:
        profile = validate_profile(args.profile)
        if args.max_images <= 0:
            raise ValueError("--max-images must be greater than zero.")
    except ValueError as error:
        raise SystemExit(f"Invalid input: {error}") from error

    output = args.output or Path("downloads") / profile
    if not args.apply:
        print(f"Preview only for @{profile}. No browser, login, network request, or download will occur.")
        print(f"Would save up to {args.max_images} visible images to {output}.")
        print(f"To apply, rerun with --apply --confirm {CONFIRMATION}.")
        return
    if args.confirm != CONFIRMATION:
        raise SystemExit(f"Refusing to download. Use --confirm {CONFIRMATION} with --apply.")
    if output.exists():
        raise SystemExit(f"Output already exists: {output}. Choose a new --output path.")

    try:
        count = download_images(profile, output, args.max_images)
    except (OSError, requests.RequestException, ValueError) as error:
        raise SystemExit(f"Download failed: {error}") from error
    print(f"Downloaded {count} image(s) to {output}.")


if __name__ == "__main__":
    main()

"""Preview or explicitly like a bounded number of Instagram posts."""

from __future__ import annotations

import argparse
from getpass import getpass


CONFIRMATION = "LIKE_POSTS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview Instagram likes before any browser action.")
    parser.add_argument("profile", help="Instagram profile username.")
    parser.add_argument("--max-likes", type=int, default=1, help="Maximum posts to like (default: 1).")
    parser.add_argument("--apply", action="store_true", help="Open Chrome and like posts.")
    parser.add_argument("--confirm", help=f"Required with --apply: {CONFIRMATION}")
    return parser.parse_args()


def validate_profile(profile: str) -> str:
    profile = profile.strip().lstrip("@")
    if not profile.replace("_", "").replace(".", "").isalnum():
        raise ValueError("Profile names may contain letters, numbers, underscores, and periods only.")
    return profile


def apply_likes(profile: str, max_likes: int) -> int:
    """Log in interactively and like no more than ``max_likes`` visible posts."""
    from selenium import webdriver
    from selenium.common.exceptions import TimeoutException
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
        liked = 0
        while liked < max_likes:
            try:
                wait.until(EC.element_to_be_clickable((By.XPATH, "//article//a"))).click()
                wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='Like']"))).click()
                liked += 1
                driver.back()
            except TimeoutException:
                break
        return liked
    finally:
        driver.quit()


def main() -> None:
    args = parse_args()
    try:
        profile = validate_profile(args.profile)
        if args.max_likes <= 0:
            raise ValueError("--max-likes must be greater than zero.")
    except ValueError as error:
        raise SystemExit(f"Invalid input: {error}") from error

    if not args.apply:
        print(f"Preview only for @{profile}. No browser or Instagram action will occur.")
        print(f"Would like at most {args.max_likes} post(s).")
        print(f"To apply, rerun with --apply --confirm {CONFIRMATION}.")
        return
    if args.confirm != CONFIRMATION:
        raise SystemExit(f"Refusing to like posts. Use --confirm {CONFIRMATION} with --apply.")

    try:
        liked = apply_likes(profile, args.max_likes)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(f"Liked {liked} post(s) for @{profile}.")


if __name__ == "__main__":
    main()

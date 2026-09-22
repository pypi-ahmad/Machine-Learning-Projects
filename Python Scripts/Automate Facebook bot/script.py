"""Plan authorized Facebook group posts or execute them with explicit consent."""

from __future__ import annotations

import argparse
import getpass

LOGIN_URL = "https://www.facebook.com/login.php"


def group_urls(group_ids: str) -> list[str]:
    """Validate comma-separated group identifiers and return their URLs."""
    identifiers = [group_id.strip() for group_id in group_ids.split(",") if group_id.strip()]
    if not identifiers or any("/" in group_id or "?" in group_id for group_id in identifiers):
        raise ValueError("Provide one or more plain group identifiers separated by commas.")
    return [f"https://www.facebook.com/groups/{group_id}" for group_id in identifiers]


def post(urls: list[str], message: str, selectors: dict[str, str], timeout: int) -> None:
    """Log in and post to configured groups using Selenium 4 waits."""
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as conditions
    from selenium.webdriver.support.ui import WebDriverWait

    email = input("Facebook email: ").strip()
    password = getpass.getpass("Facebook password: ")
    driver = webdriver.Chrome()
    try:
        wait = WebDriverWait(driver, timeout)
        driver.get(LOGIN_URL)
        wait.until(conditions.element_to_be_clickable((By.CSS_SELECTOR, selectors["email"]))).send_keys(email)
        wait.until(conditions.element_to_be_clickable((By.CSS_SELECTOR, selectors["password"]))).send_keys(password)
        wait.until(conditions.element_to_be_clickable((By.CSS_SELECTOR, selectors["login"]))).click()
        for url in urls:
            driver.get(url)
            wait.until(conditions.element_to_be_clickable((By.CSS_SELECTOR, selectors["composer"]))).click()
            driver.switch_to.active_element.send_keys(message)
            wait.until(conditions.element_to_be_clickable((By.CSS_SELECTOR, selectors["post"]))).click()
    finally:
        driver.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", required=True, help="comma-separated group identifiers")
    parser.add_argument("--message", required=True)
    parser.add_argument("--post", action="store_true", help="perform posts; default is preview")
    parser.add_argument("--timeout", type=int, default=15)
    for name in ("email", "password", "login", "composer", "post"):
        parser.add_argument(f"--{name}-selector")
    args = parser.parse_args()
    try:
        urls = group_urls(args.groups)
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error
    print("Planned groups:")
    for url in urls:
        print(url)
    if not args.post:
        print("Preview only. Use --post only for groups you are authorized to manage.")
        return
    selectors = {name: getattr(args, f"{name}_selector") for name in ("email", "password", "login", "composer", "post")}
    if not all(selectors.values()) or args.timeout <= 0:
        raise SystemExit("Error: --post requires all selector options and a positive timeout.")
    post(urls, args.message, selectors, args.timeout)


if __name__ == "__main__":
    main()

"""Export email addresses that are visibly shared in one LinkedIn post's comments."""

from __future__ import annotations

import argparse
import csv
import getpass
import re
from pathlib import Path
from urllib.parse import urlparse

from email_validator import EmailNotValidError, validate_email
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as expected
from selenium.webdriver.support.ui import WebDriverWait


EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
WAIT_SECONDS = 15


def linkedin_post_url(value: str) -> str:
    """Reject non-LinkedIn URLs before a browser session starts."""
    parsed = urlparse(value)
    is_linkedin = parsed.hostname == "linkedin.com" or (
        parsed.hostname is not None and parsed.hostname.endswith(".linkedin.com")
    )
    if parsed.scheme != "https" or not is_linkedin:
        raise argparse.ArgumentTypeError("post URL must be an HTTPS LinkedIn URL")
    return value


def comment_record(comment: webdriver.remote.webelement.WebElement) -> tuple[str, str] | None:
    """Return the first valid email visibly shared in one comment."""
    match = EMAIL_PATTERN.search(comment.text)
    if match is None:
        return None
    try:
        email = validate_email(match.group(), check_deliverability=False).normalized
        name = comment.find_element(By.CSS_SELECTOR, ".hoverable-link-text").text.strip()
    except (EmailNotValidError, NoSuchElementException):
        return None
    return name, email


def expand_comments(driver: webdriver.Chrome, maximum: int) -> None:
    """Load at most ``maximum`` batches of earlier comments when available."""
    for _ in range(maximum):
        buttons = driver.find_elements(
            By.CSS_SELECTOR, ".comments-comments-list__show-previous-container button"
        )
        if not buttons:
            return
        buttons[0].click()


def scrape_post(post_url: str, account_email: str, password: str, maximum_load_more: int) -> dict[str, str]:
    """Log in, read visible post comments, and return name-to-email records."""
    driver = webdriver.Chrome()
    wait = WebDriverWait(driver, WAIT_SECONDS)
    try:
        driver.get("https://www.linkedin.com/login")
        wait.until(expected.visibility_of_element_located((By.ID, "username"))).send_keys(account_email)
        driver.find_element(By.ID, "password").send_keys(password)
        driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
        wait.until(expected.presence_of_element_located((By.TAG_NAME, "body")))

        driver.get(post_url)
        wait.until(expected.presence_of_element_located((By.TAG_NAME, "article")))
        expand_comments(driver, maximum_load_more)
        entries = map(comment_record, driver.find_elements(By.TAG_NAME, "article"))
        return dict(filter(None, entries))
    except TimeoutException as error:
        raise RuntimeError("LinkedIn did not load the requested page in time.") from error
    finally:
        driver.quit()


def write_csv(records: dict[str, str], output: Path) -> None:
    """Write records using CSV quoting and an explicit UTF-8 encoding."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("name", "email"))
        writer.writerows(records.items())


def main() -> None:
    """Parse CLI options, prompt for credentials, and export discovered records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("post_url", type=linkedin_post_url, help="HTTPS URL for the LinkedIn post")
    parser.add_argument("--output", type=Path, default=Path("emails.csv"), help="CSV output path")
    parser.add_argument(
        "--max-load-more", type=int, default=3, help="maximum earlier-comment batches to load"
    )
    args = parser.parse_args()
    if args.max_load_more < 0:
        parser.error("--max-load-more must be zero or greater")

    account_email = input("LinkedIn email: ").strip()
    password = getpass.getpass("LinkedIn password: ")
    if not account_email or not password:
        parser.error("LinkedIn email and password are required")

    try:
        records = scrape_post(args.post_url, account_email, password, args.max_load_more)
    except WebDriverException as error:
        raise SystemExit(f"Browser error: {error.msg}") from error
    except RuntimeError as error:
        raise SystemExit(f"Error: {error}") from error

    write_csv(records, args.output)
    print(f"Saved {len(records)} record(s) to {args.output}")


if __name__ == "__main__":
    main()

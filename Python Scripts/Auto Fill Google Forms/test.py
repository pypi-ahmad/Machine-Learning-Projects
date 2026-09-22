"""Validate CSV rows or submit them to a form you are authorized to automate."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REQUIRED_COLUMNS = ("name", "email", "phone_number")


def load_rows(path: Path) -> list[dict[str, str]]:
    """Load and validate required contact fields from a CSV file."""
    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None or any(column not in reader.fieldnames for column in REQUIRED_COLUMNS):
            raise ValueError(f"CSV requires columns: {', '.join(REQUIRED_COLUMNS)}")
        rows = [{column: row[column].strip() for column in REQUIRED_COLUMNS} for row in reader]
    if any(not all(row.values()) for row in rows):
        raise ValueError("CSV contains an empty required value.")
    return rows


def submit_rows(url: str, rows: list[dict[str, str]], selectors: dict[str, str], timeout: int) -> None:
    """Submit rows with Selenium 4 explicit waits."""
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as conditions
    from selenium.webdriver.support.ui import WebDriverWait

    driver = webdriver.Firefox()
    try:
        for row in rows:
            driver.get(url)
            wait = WebDriverWait(driver, timeout)
            for field in REQUIRED_COLUMNS:
                wait.until(conditions.element_to_be_clickable((By.XPATH, selectors[field]))).send_keys(row[field])
            wait.until(conditions.element_to_be_clickable((By.XPATH, selectors["submit"]))).click()
    finally:
        driver.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--form-url")
    parser.add_argument("--name-xpath")
    parser.add_argument("--email-xpath")
    parser.add_argument("--phone-xpath")
    parser.add_argument("--submit-xpath")
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--submit", action="store_true", help="submit rows; default validates only")
    args = parser.parse_args()
    try:
        rows = load_rows(args.csv)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
    print(f"Validated {len(rows)} row(s).")
    if not args.submit:
        print("Dry run: no browser was opened and no form was submitted.")
        return
    selectors = {"name": args.name_xpath, "email": args.email_xpath, "phone_number": args.phone_xpath, "submit": args.submit_xpath}
    if not args.form_url or not all(selectors.values()) or args.timeout <= 0:
        raise SystemExit("Error: --submit requires a URL, all XPath options, and a positive timeout.")
    submit_rows(args.form_url, rows, selectors, args.timeout)


if __name__ == "__main__":
    main()

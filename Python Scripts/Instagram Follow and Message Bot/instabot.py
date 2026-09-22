"""Preview or explicitly apply Instagram follow and direct-message actions."""

from __future__ import annotations

import argparse
from getpass import getpass


CONFIRMATION = "FOLLOW_AND_MESSAGE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview Instagram follow and message actions.")
    parser.add_argument("targets", nargs="+", help="Instagram usernames to process.")
    parser.add_argument("--apply", action="store_true", help="Open Chrome and perform the requested actions.")
    parser.add_argument(
        "--confirm",
        help=f"Required with --apply. Enter exactly {CONFIRMATION!r}.",
    )
    return parser.parse_args()


def validate_targets(targets: list[str]) -> list[str]:
    """Return trimmed usernames after basic local validation."""
    cleaned = [target.strip().lstrip("@") for target in targets]
    if any(not target.replace("_", "").replace(".", "").isalnum() for target in cleaned):
        raise ValueError("Usernames may contain letters, numbers, underscores, and periods only.")
    return cleaned


def preview(targets: list[str]) -> None:
    print("Preview only. No browser will open and no Instagram action will occur.")
    print(f"Would request a follow and one direct message for {len(targets)} account(s):")
    for target in targets:
        print(f"- @{target}")
    print(f"To apply, rerun with --apply --confirm {CONFIRMATION}")


def apply_actions(targets: list[str]) -> None:
    """Log in interactively and apply each requested social action."""
    from selenium import webdriver
    from selenium.common.exceptions import TimeoutException, WebDriverException
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    username = input("Instagram username: ").strip()
    password = getpass("Instagram password: ")
    message = input("Message to send to every listed account: ").strip()
    if not username or not password or not message:
        raise ValueError("Username, password, and message are all required.")

    try:
        driver = webdriver.Chrome()
    except WebDriverException as error:
        raise RuntimeError(f"Could not start Chrome: {error}") from error

    try:
        wait = WebDriverWait(driver, 30)
        driver.get("https://www.instagram.com/")
        wait.until(EC.presence_of_element_located((By.NAME, "username"))).send_keys(username)
        driver.find_element(By.NAME, "password").send_keys(password)
        driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()

        for target in targets:
            driver.get(f"https://www.instagram.com/{target}/")
            try:
                wait.until(
                    EC.element_to_be_clickable((By.XPATH, "//button[normalize-space()='Follow']"))
                ).click()
                wait.until(
                    EC.element_to_be_clickable((By.XPATH, "//button[normalize-space()='Message']"))
                ).click()
                message_box = wait.until(EC.presence_of_element_located((By.TAG_NAME, "textarea")))
                message_box.send_keys(message, Keys.ENTER)
                print(f"Applied requested actions for @{target}")
            except TimeoutException:
                print(f"Could not find the required controls for @{target}; no message was sent.")
    finally:
        driver.quit()


def main() -> None:
    args = parse_args()
    try:
        targets = validate_targets(args.targets)
    except ValueError as error:
        raise SystemExit(f"Invalid target: {error}") from error

    if not args.apply:
        preview(targets)
        return
    if args.confirm != CONFIRMATION:
        raise SystemExit(f"Refusing to act. Use --confirm {CONFIRMATION} with --apply.")

    try:
        apply_actions(targets)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()

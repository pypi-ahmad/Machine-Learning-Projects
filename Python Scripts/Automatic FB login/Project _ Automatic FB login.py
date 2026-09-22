"""Preview or explicitly start a Facebook login browser session."""

from __future__ import annotations

import argparse
import getpass

LOGIN_URL = "https://www.facebook.com/login.php"


def login(email_selector: str, password_selector: str, submit_selector: str, timeout: int) -> None:
    """Open Chrome and submit credentials entered only at runtime."""
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as conditions
    from selenium.webdriver.support.ui import WebDriverWait

    email = input("Facebook email or phone: ").strip()
    password = getpass.getpass("Facebook password: ")
    driver = webdriver.Chrome()
    try:
        wait = WebDriverWait(driver, timeout)
        driver.get(LOGIN_URL)
        wait.until(conditions.element_to_be_clickable((By.CSS_SELECTOR, email_selector))).send_keys(email)
        wait.until(conditions.element_to_be_clickable((By.CSS_SELECTOR, password_selector))).send_keys(password)
        wait.until(conditions.element_to_be_clickable((By.CSS_SELECTOR, submit_selector))).click()
    finally:
        driver.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--login", action="store_true", help="open Chrome and submit credentials")
    parser.add_argument("--email-selector", default="input[name='email']")
    parser.add_argument("--password-selector", default="input[name='pass']")
    parser.add_argument("--submit-selector", default="button[name='login']")
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args()
    if not args.login:
        print(f"Preview only. Login URL: {LOGIN_URL}")
        print("Re-run with --login only for an account you are authorized to access.")
        return
    if args.timeout <= 0:
        raise SystemExit("Error: --timeout must be greater than zero.")
    login(args.email_selector, args.password_selector, args.submit_selector, args.timeout)


if __name__ == "__main__":
    main()

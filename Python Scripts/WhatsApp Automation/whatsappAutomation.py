"""Send one WhatsApp Web message after explicit confirmation."""

from __future__ import annotations

import argparse

from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as expected
from selenium.webdriver.support.ui import WebDriverWait

WHATSAPP_WEB_URL = "https://web.whatsapp.com/"


def parse_args() -> argparse.Namespace:
    """Parse the recipient, message, and browser options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contact", help="Exact WhatsApp contact name")
    parser.add_argument("message", help="Message text to send")
    parser.add_argument(
        "--send",
        action="store_true",
        help="Open WhatsApp Web and send the message",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Seconds to wait for login and the contact (default: 60)",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Leave the browser open after the send attempt",
    )
    return parser.parse_args()


def contact_xpath(contact: str) -> str:
    """Return a contact-title selector for names without quote characters."""
    if "'" in contact or '"' in contact:
        raise ValueError("Contact names containing quotes are not supported.")
    return f"//span[@title='{contact}']"


def send_message(contact: str, message: str, timeout: int, keep_open: bool) -> None:
    """Open WhatsApp Web, wait for a contact, and send one message."""
    if timeout < 1:
        raise ValueError("Timeout must be at least one second.")
    selector = contact_xpath(contact)
    driver = webdriver.Chrome()
    try:
        driver.get(WHATSAPP_WEB_URL)
        wait = WebDriverWait(driver, timeout)
        contact_element = wait.until(
            expected.element_to_be_clickable((By.XPATH, selector))
        )
        contact_element.click()
        message_box = wait.until(
            expected.element_to_be_clickable((By.CSS_SELECTOR, "#main footer [contenteditable='true']"))
        )
        message_box.send_keys(message, Keys.ENTER)
    finally:
        if not keep_open:
            driver.quit()


def main() -> None:
    """Preview or send the requested message."""
    args = parse_args()
    if not args.send:
        print(f"Would send a message to {args.contact!r}: {args.message!r}")
        print("Pass --send to open WhatsApp Web and send it.")
        return
    try:
        send_message(args.contact, args.message, args.timeout, args.keep_open)
    except (ValueError, WebDriverException) as error:
        raise SystemExit(f"Error: {error}") from error
    print(f"Message sent to {args.contact}.")


if __name__ == "__main__":
    main()

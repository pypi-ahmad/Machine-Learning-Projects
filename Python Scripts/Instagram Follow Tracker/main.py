"""Preview or run a read-only Instagram follow-back check."""

from __future__ import annotations

import argparse
from getpass import getpass

from prettytable import PrettyTable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find followed accounts that do not follow back.")
    parser.add_argument("username", help="Instagram account to inspect.")
    parser.add_argument("--run", action="store_true", help="Open Chrome, log in, and read follower lists.")
    return parser.parse_args()


def validate_username(username: str) -> str:
    username = username.strip().lstrip("@")
    if not username.replace("_", "").replace(".", "").isalnum():
        raise ValueError("Username may contain letters, numbers, underscores, and periods only.")
    return username


class InstaBot:
    """Read the signed-in account's follower and following lists through Chrome."""

    def __init__(self, username: str, password: str):
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        self.driver = webdriver.Chrome()
        self.username = username
        self.by = By
        self.wait = WebDriverWait(self.driver, 30)
        self.driver.get("https://www.instagram.com/")
        self.wait.until(EC.presence_of_element_located((By.NAME, "username"))).send_keys(username)
        self.driver.find_element(By.NAME, "password").send_keys(password)
        self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()

    def get_unfollowers(self) -> list[str]:
        self.driver.get(f"https://www.instagram.com/{self.username}/")
        following = self._read_names("following")
        followers = self._read_names("followers")
        return sorted(set(following) - set(followers))

    def _read_names(self, relation: str) -> list[str]:
        from selenium.webdriver.support import expected_conditions as EC

        self.driver.get(f"https://www.instagram.com/{self.username}/{relation}/")
        dialog = self.wait.until(
            EC.presence_of_element_located((self.by.CSS_SELECTOR, "div[role='dialog']"))
        )
        previous_height = -1
        current_height = 0
        while current_height != previous_height:
            previous_height = current_height
            current_height = self.driver.execute_script(
                "arguments[0].scrollTop = arguments[0].scrollHeight; return arguments[0].scrollHeight;", dialog
            )
        links = dialog.find_elements(self.by.TAG_NAME, "a")
        return [link.text for link in links if link.text]

    def close(self) -> None:
        self.driver.quit()


def print_results(unfollowers: list[str]) -> None:
    table = PrettyTable(["Does not follow back"])
    for username in unfollowers:
        table.add_row([username])
    print(table)


def main() -> None:
    args = parse_args()
    try:
        username = validate_username(args.username)
    except ValueError as error:
        raise SystemExit(f"Invalid username: {error}") from error

    if not args.run:
        print(f"Preview only for @{username}. No browser will open and no Instagram data will be read.")
        print("Rerun with --run to open Chrome and perform the read-only check.")
        return

    password = getpass("Instagram password: ")
    if not password:
        raise SystemExit("A password is required.")

    bot = InstaBot(username, password)
    try:
        print_results(bot.get_unfollowers())
    finally:
        bot.close()


if __name__ == "__main__":
    main()

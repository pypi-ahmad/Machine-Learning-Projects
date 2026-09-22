"""List Instagram accounts you follow that do not follow you back."""

import argparse
from getpass import getpass
import time

from selenium import webdriver
from selenium.webdriver.common.by import By


class InstaBot:
    """Browser automation for the current user's Instagram follow lists."""

    def __init__(self, username: str, password: str) -> None:
        self.username = username
        self.password = password
        self.driver = webdriver.Chrome()
        self.following: list[str] = []
        self.followers: list[str] = []

    def start(self) -> None:
        self.driver.get('https://www.instagram.com/')
        time.sleep(2)

    def login(self) -> None:
        self.driver.find_element(By.XPATH, '//*[@id="loginForm"]/div/div[1]/div/label/input').send_keys(
            self.username
        )
        self.driver.find_element(By.XPATH, '//*[@id="loginForm"]/div/div[2]/div/label/input').send_keys(
            self.password
        )
        self.driver.find_element(By.XPATH, '//*[@id="loginForm"]/div/div[3]/button/div').click()
        time.sleep(3)

    def open_profile(self) -> None:
        self.driver.find_element(
            By.XPATH, '//*[@id="react-root"]/section/main/section/div[3]/div[1]/div/div[2]/div[1]/a'
        ).click()
        time.sleep(2)

    def scroll_list(self, xpath: str) -> list[str]:
        time.sleep(2)
        scroll_box = self.driver.find_element(By.XPATH, xpath)
        last_height, height = 0, 1
        while last_height != height:
            last_height = height
            time.sleep(1)
            height = self.driver.execute_script(
                'arguments[0].scrollTo(0, arguments[0].scrollHeight); return arguments[0].scrollHeight;',
                scroll_box,
            )

        names = [link.text for link in scroll_box.find_elements(By.TAG_NAME, 'a') if link.text]
        self.driver.find_element(By.XPATH, '/html/body/div[4]/div/div/div[1]/div/div[2]/button/div').click()
        return names

    def get_following(self) -> None:
        self.driver.find_element(By.XPATH, '/html/body/div[1]/section/main/div/header/section/ul/li[3]/a').click()
        self.following = self.scroll_list('/html/body/div[4]/div/div/div[2]')

    def get_followers(self) -> None:
        self.driver.find_element(By.XPATH, '//*[@id="react-root"]/section/main/div/header/section/ul/li[2]/a').click()
        self.followers = self.scroll_list('/html/body/div[4]/div/div/div[2]')

    def get_unfollowers(self) -> list[str]:
        follower_names = set(self.followers)
        return [name for name in self.following if name not in follower_names]

    def close(self) -> None:
        self.driver.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description='List Instagram accounts that do not follow you back.')
    parser.add_argument('--username', help='Instagram username; omitted values are prompted')
    args = parser.parse_args()
    username = args.username or input('Enter your username: ')
    password = getpass('Enter your password (will not appear as you type): ')
    if not username or not password:
        parser.error('A username and password are required')

    bot = InstaBot(username, password)
    try:
        bot.start()
        bot.login()
        bot.open_profile()
        bot.get_following()
        bot.get_followers()
        for account in bot.get_unfollowers():
            print(account)
    finally:
        bot.close()


if __name__ == '__main__':
    main()

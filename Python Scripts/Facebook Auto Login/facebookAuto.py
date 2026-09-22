"""Open Facebook and submit an interactive login form with Selenium."""

import argparse
from getpass import getpass

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as expected
from selenium.webdriver.support.ui import WebDriverWait


def main() -> None:
    parser = argparse.ArgumentParser(description='Open Facebook and submit a login form.')
    parser.add_argument('--username', help='Facebook email address or username')
    args = parser.parse_args()

    username = args.username or input('Facebook email or username: ').strip()
    password = getpass('Facebook password: ')
    if not username or not password:
        parser.error('A username and password are required')

    driver = webdriver.Chrome()
    driver.get('https://www.facebook.com')
    wait = WebDriverWait(driver, 15)
    wait.until(expected.presence_of_element_located((By.ID, 'email'))).send_keys(username)
    driver.find_element(By.ID, 'pass').send_keys(password)
    driver.find_element(By.NAME, 'login').click()


if __name__ == '__main__':
    main()

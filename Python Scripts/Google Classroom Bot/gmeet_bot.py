import argparse
from time import sleep
from configparser import ConfigParser, Error as ConfigError
from datetime import datetime
import csv
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By

classTime = ["09:20", "11:40", "14:25"]


class ClassAutomation:
    def __init__(self, username: str, password: str, schedule_path: Path, profile_path: str | None):
        self.username = username
        self.password = password
        self.schedule_path = schedule_path
        self.profile_path = profile_path
        self.count = 0
        self.findCount()
        # Runs endlessly and calls initClass method when it's class time
        while True:
            if datetime.now().strftime("%H:%M") in classTime:
                print(datetime.now().strftime("%H:%M"))
                self.initClass()
            sleep(30)

    # Initiates Class
    def initClass(self):
        className = self.findClass()
        if className is None:
            return
        print("Initiating...")
        self.login()
        self.driver.find_element(By.XPATH, "//div[text()='{}']".format(className)).click()
        sleep(10)
        link = self.driver.find_element(By.PARTIAL_LINK_TEXT, 'https://meet.google.com/lookup/').text
        self.driver.get(link)
        sleep(10)
        self.driver.find_element(By.XPATH, "//span[text()='Join now']").click()
        sleep(60 * 60)
        print("Quitting...")
        sleep(5)
        self.driver.quit()
        if self.count < 2:
            self.count = self.count + 1
        else:
            self.count = 0
        self.findCount()

    # Returns the ClassName for the current time
    def findClass(self):
        with self.schedule_path.open('r', encoding='utf-8', newline='') as csvFile:
            reader = csv.DictReader(csvFile)
            for row in reader:
                if row["Day"] == datetime.now().strftime("%a"):
                    return row[classTime[self.count]]
            return None

    # Determines the current time position in the classTime list

    def findCount(self):
        if self.findClass() is None:
            print("No Class Today")
            return
        currentTime = datetime.now().strftime("%H:%M")
        currentHour = int(currentTime.split(":")[0])
        currentMin = int(currentTime.split(":")[1])
        for i in classTime:
            if currentHour == int(
                    i.split(":")[0]) and currentMin < int(i.split(":")[1]):
                self.count = classTime.index(i)
                print("Next Class at", classTime[self.count], "Today")
                break
            elif currentHour < int(i.split(":")[0]):
                self.count = classTime.index(i)
                print("Next Class at", classTime[self.count], "Today")
                break
            else:
                if classTime.index(i) == 2:
                    self.count = 0
                    print("Next Class at", classTime[self.count], "Tomorrow")
                    break
                continue

    # Logs into the google classroom with the account credentials
    def login(self):
        options = webdriver.FirefoxOptions()
        if self.profile_path:
            options.profile = self.profile_path
        self.driver = webdriver.Firefox(options=options)
        self.driver.get("https://accounts.google.com/")
        sleep(2)
        try:
            self.driver.find_element(By.NAME, 'identifier').send_keys(self.username)
            sleep(1)
        except:
            self.driver.find_element(By.NAME, 'Email').send_keys(self.username)
            sleep(1)
        try:
            self.driver.find_element(By.ID, 'identifierNext').click()
            sleep(4)
        except:
            self.driver.find_element(By.ID, 'next').click()
            sleep(4)
        try:
            self.driver.find_element(By.NAME, 'password').send_keys(self.password)
            sleep(1)
        except:
            self.driver.find_element(By.NAME, 'Passwd').send_keys(self.password)
            sleep(1)
        try:
            self.driver.find_element(By.ID, 'passwordNext').click()
            sleep(4)
        except:
            self.driver.find_element(By.ID, 'trustDevice').click()
            self.driver.find_element(By.ID, 'submit').click()
            sleep(4)
        self.driver.get("https://classroom.google.com/")
        sleep(6)


def load_credentials(config_path: Path) -> tuple[str, str]:
    config = ConfigParser()
    if not config.read(config_path):
        raise FileNotFoundError(f'Configuration file not found: {config_path}')
    return config.get('AUTH', 'USERNAME'), config.get('AUTH', 'PASSWORD')


def main() -> None:
    parser = argparse.ArgumentParser(description='Join scheduled Google Classroom meetings.')
    parser.add_argument('--config', type=Path, default=Path('config.ini'))
    parser.add_argument('--schedule', type=Path, default=Path('schedule.csv'))
    parser.add_argument('--firefox-profile', help='Optional Firefox profile directory')
    args = parser.parse_args()
    try:
        username, password = load_credentials(args.config)
    except (ConfigError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))
    if not args.schedule.is_file():
        parser.error(f'Schedule file not found: {args.schedule}')

    ClassAutomation(username, password, args.schedule, args.firefox_profile)


if __name__ == '__main__':
    main()

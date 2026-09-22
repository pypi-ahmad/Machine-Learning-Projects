import argparse
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from eng_hindi import eth
from plyer import notification

DATA_URL = 'https://www.medtalks.in/live-corona-counter-india'
ICON_PATH = Path(__file__).with_name('Notify_icon.ico')
REQUEST_TIMEOUT = 15


def notify_user(title: str, message: str) -> None:
    notification.notify(
        title=title,
        message=message,
        app_icon=str(ICON_PATH),
        timeout=5,
    )


def get_info(url: str) -> str:
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Send periodic desktop notifications for selected Indian states.'
    )
    parser.parse_args()

    interval = int(input('Enter interval in secs: '))
    state_names = input('Enter name of states: ').split(',')
    states = [f'{name} ( {eth(name)} )' for name in state_names]

    while True:
        html_data = get_info(DATA_URL)
        soup = BeautifulSoup(html_data, 'html.parser')

        table_body = soup.find('tbody')
        if table_body is None:
            raise RuntimeError('The data source did not contain the expected table')

        table_data = ''.join(row.get_text() for row in table_body.find_all('tr'))[1:]
        items = table_data.split('\n\n')

        for item in items[:-2]:
            data = item.split('\n')
            if data[0] in states:
                title = 'Cases of Covid-19'
                message = (
                    f'State: {data[0]}: Total: {data[1]}\n'
                    f' Active: {data[2]}\n Death: {data[3]}'
                )
                notify_user(title, message)
                time.sleep(2)

        time.sleep(interval)


if __name__ == '__main__':
    main()

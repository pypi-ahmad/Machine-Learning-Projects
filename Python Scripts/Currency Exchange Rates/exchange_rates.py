"""Display an exchange-rate table for a user-selected source currency."""

import sys

from bs4 import BeautifulSoup
import requests as req

REQUEST_TIMEOUT_SECONDS = 15


def fetch_page(url):
    """Fetch a page or print a concise connection error."""
    try:
        response = req.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except req.RequestException as error:
        print(f"Could not reach x-rates.com: {error}")
        return None
    return response.text


def main():
    """Prompt for a source currency and print its conversion table."""
    page = fetch_page('https://www.x-rates.com/')
    if page is None:
        return 1

    currencies = []
    soup = BeautifulSoup(page, 'html.parser')
    options = soup.find_all('option')[:-11]

    for option in options:
        currency_short = option.text[:(option.text.find(" "))]
        currency_name = option.text[(option.text.find(" ") + 3):]
        current_element = {'name': currency_name, 'short': currency_short}
        currencies.append(current_element)
        print('{}. {} ({})'.format(len(currencies), current_element['name'],
                                   current_element['short']))

    if not currencies:
        print('No currencies were found. The x-rates.com page may have changed.')
        return 1

    try:
        currency_index = int(input('Enter your currency\'s position number: ')) - 1
        currency = currencies[currency_index]
    except (ValueError, IndexError):
        print('Enter a number from the displayed currency list.')
        return 1

    amount = input('Enter the amount of {}s: '.format(currency['name'].lower()))
    try:
        if float(amount) <= 0:
            raise ValueError
    except ValueError:
        print('Enter an amount greater than zero, using a dot for decimals.')
        return 1

    currencies_table_url = 'https://www.x-rates.com/table/?from={}&amount={}'.format(
        currency['short'], amount)
    currencies_table_page = fetch_page(currencies_table_url)
    if currencies_table_page is None:
        return 1

    soup = BeautifulSoup(currencies_table_page, 'html.parser')
    table = soup.findChild('table', attrs={'class': 'tablesorter'})
    if table is None:
        print('No exchange-rate table was found. The x-rates.com page may have changed.')
        return 1

    table_rows = table.findChildren('tr')[1:]
    print('For {} {}s you will get:'.format(amount, currency['name'].lower()))

    for table_row in table_rows:
        row_data = table_row.findChildren('td')
        exchange_rate = {
            'currency': row_data[0].text,
            'amount': float(row_data[1].text),
        }
        print('{:.3f} {}s'.format(exchange_rate['amount'],
                                  exchange_rate['currency']))
    return 0



if __name__ == '__main__':
    sys.exit(main())

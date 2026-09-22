"""Convert an address to coordinates with the LocationIQ search API."""

import argparse
from getpass import getpass

import requests

API_URL = 'https://us1.locationiq.com/v1/search.php'
REQUEST_TIMEOUT = 15


def geocode(address: str, token: str) -> tuple[str, str]:
    """Return latitude and longitude for the first LocationIQ result."""
    response = requests.get(
        API_URL,
        params={'key': token, 'q': address, 'format': 'json'},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    results = response.json()
    if not results:
        raise ValueError('No matching location was found.')
    return results[0]['lat'], results[0]['lon']


def main() -> None:
    parser = argparse.ArgumentParser(description='Geocode an address with LocationIQ.')
    parser.add_argument('address', nargs='?', help='Address to geocode')
    parser.add_argument('--token', help='LocationIQ API token')
    args = parser.parse_args()

    address = args.address or input('Input the address: ').strip()
    token = args.token or getpass('LocationIQ API token: ')
    if not address or not token:
        parser.error('An address and LocationIQ API token are required')

    try:
        latitude, longitude = geocode(address, token)
    except (KeyError, ValueError, requests.RequestException) as error:
        parser.error(f'Geocoding failed: {error}')

    print(f'The latitude of the given address is: {latitude}')
    print(f'The longitude of the given address is: {longitude}')


if __name__ == '__main__':
    main()

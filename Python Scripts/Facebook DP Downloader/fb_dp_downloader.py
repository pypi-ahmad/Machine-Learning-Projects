"""Download the large profile image for a public Facebook numeric ID."""

import argparse
from pathlib import Path

import requests

URL = 'https://graph.facebook.com/{}/picture?type=large'
REQUEST_TIMEOUT = 15


def main() -> None:
    parser = argparse.ArgumentParser(description='Download a public Facebook profile image.')
    parser.add_argument('facebook_id', nargs='?', help='Numeric Facebook user ID')
    parser.add_argument('--output-dir', type=Path, default=Path.cwd() / 'fb_dps')
    args = parser.parse_args()

    facebook_id = args.facebook_id or input('Enter the Facebook ID to download its profile picture: ')
    if not facebook_id.isdecimal():
        parser.error('facebook_id must contain digits only')

    try:
        response = requests.get(URL.format(facebook_id), timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as error:
        parser.error(f'profile image request failed: {error}')

    content_type = response.headers.get('Content-Type', '')
    if not content_type.startswith('image/'):
        parser.error(f'expected an image response, received {content_type or "unknown content"}')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f'{facebook_id}_img.jpg'
    output_path.write_bytes(response.content)
    print(f'Saved profile image to {output_path}')


if __name__ == '__main__':
    main()

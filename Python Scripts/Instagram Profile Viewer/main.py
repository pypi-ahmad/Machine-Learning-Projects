"""View public Instagram Open Graph profile metadata."""

import argparse
import re
from pathlib import Path
import pprint

from lxml import html
import requests

from profilepic import download_profile_image

PROFILE_URL = 'https://www.instagram.com/{}/'
REQUEST_TIMEOUT = 15
USERNAME_PATTERN = re.compile(r'^[A-Za-z0-9._]{1,30}$')


def parse_profile_page(page_content: bytes, username: str) -> dict[str, str]:
    """Extract public Open Graph metadata from an Instagram page."""
    tree = html.fromstring(page_content)
    title = tree.xpath('//meta[@property="og:title"]/@content')
    description = tree.xpath('//meta[@property="og:description"]/@content')
    image = tree.xpath('//meta[@property="og:image"]/@content')
    if not title and not description:
        raise ValueError('No public profile metadata was found.')

    profile = {
        'username': username,
        'name': title[0] if title else 'Unavailable',
        'description': description[0] if description else 'Unavailable',
        'profile_image_url': image[0] if image else '',
    }
    return profile


def fetch_profile(username: str) -> dict[str, str]:
    """Request and parse one public Instagram profile page."""
    if not USERNAME_PATTERN.fullmatch(username):
        raise ValueError('Username must contain only letters, digits, periods, or underscores.')
    response = requests.get(
        PROFILE_URL.format(username),
        headers={'User-Agent': 'InstagramProfileViewer/1.0 (public metadata utility)'},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return parse_profile_page(response.content, username)


def main() -> None:
    parser = argparse.ArgumentParser(description='View public Instagram profile metadata.')
    parser.add_argument('username', help='Public Instagram username')
    parser.add_argument('--download-image', type=Path, help='New path for the profile image')
    args = parser.parse_args()

    try:
        profile = fetch_profile(args.username)
        if args.download_image:
            if not profile['profile_image_url']:
                raise ValueError('No public profile image URL was found.')
            download_profile_image(profile['profile_image_url'], args.download_image)
    except (OSError, ValueError, requests.RequestException) as error:
        parser.error(str(error))

    pprint.pprint(profile)
    if args.download_image:
        print(f'Profile image saved to {args.download_image}')


if __name__ == '__main__':
    main()

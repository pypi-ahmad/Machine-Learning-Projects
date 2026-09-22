"""Display local image metadata and optional GPS-derived location details."""

import argparse
from datetime import datetime
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS

from author_utils import get_author
from gps_utils import get_coordinates, get_location


def inspect_image(path: Path) -> dict[str, object]:
    """Return local metadata without making a network request."""
    with Image.open(path) as image:
        width, height = image.size
        labeled_exif = {TAGS.get(key, str(key)): value for key, value in image.getexif().items()}

    metadata: dict[str, object] = {
        'ImageName': path.name,
        'Size': f'{width}x{height}',
        'FileExtension': path.suffix,
        'ImageWidth': labeled_exif.get('ExifImageWidth', 'No ImageWidth'),
        'ImageHeight': labeled_exif.get('ExifImageHeight', 'No ImageHeight'),
        'DateTimeOriginal': labeled_exif.get('DateTimeOriginal', 'No DateTimeOriginal'),
        'CreateDate': datetime.fromtimestamp(path.stat().st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
    }
    try:
        metadata['Author'] = get_author(str(path))
    except OSError:
        metadata['Author'] = 'Unavailable'
    try:
        metadata['Coordinates'] = get_coordinates(path)
    except ValueError:
        metadata['Coordinates'] = 'No GPS metadata'
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description='Display image metadata and optional GPS location.')
    parser.add_argument('image', type=Path, help='Image file to inspect')
    parser.add_argument('--reverse-geocode', action='store_true', help='Look up GPS coordinates through Nominatim')
    parser.add_argument('--nominatim-user-agent', help='Required Nominatim user agent for reverse geocoding')
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f'Image file not found: {args.image}')
    if args.reverse_geocode and not args.nominatim_user_agent:
        parser.error('--nominatim-user-agent is required with --reverse-geocode')

    try:
        metadata = inspect_image(args.image)
    except OSError as error:
        parser.error(f'Unable to inspect image: {error}')
    for label, value in metadata.items():
        print(f'{label}: {value}')

    if args.reverse_geocode:
        try:
            print(f'Location: {get_location(args.image, args.nominatim_user_agent)}')
        except (OSError, ValueError) as error:
            print(f'Location: unavailable ({error})')


if __name__ == '__main__':
    main()

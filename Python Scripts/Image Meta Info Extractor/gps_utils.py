"""Extract and optionally reverse-geocode GPS coordinates from image EXIF data."""

from pathlib import Path

import exifread
from geopy.geocoders import Nominatim


def _to_decimal(values) -> float:
    degrees, minutes, seconds = (float(value) for value in values)
    return degrees + minutes / 60 + seconds / 3600


def get_coordinates(filename: Path) -> tuple[float, float]:
    """Return signed latitude and longitude or raise ValueError when absent."""
    with filename.open('rb') as image_file:
        tags = exifread.process_file(image_file, details=False)

    try:
        latitude = _to_decimal(tags['GPS GPSLatitude'].values)
        longitude = _to_decimal(tags['GPS GPSLongitude'].values)
        if str(tags['GPS GPSLatitudeRef']) == 'S':
            latitude = -latitude
        if str(tags['GPS GPSLongitudeRef']) == 'W':
            longitude = -longitude
    except KeyError as error:
        raise ValueError('Image does not contain complete GPS metadata.') from error
    return latitude, longitude


def get_location(filename: Path, user_agent: str) -> str:
    """Reverse-geocode image GPS coordinates through Nominatim."""
    latitude, longitude = get_coordinates(filename)
    location = Nominatim(user_agent=user_agent).reverse((latitude, longitude), timeout=15)
    if location is None:
        raise ValueError('No address was returned for the image coordinates.')
    return location.address

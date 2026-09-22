"""Download a public profile image returned by an Instagram page."""

from pathlib import Path

import requests
from tqdm import tqdm

REQUEST_TIMEOUT = 15


def download_profile_image(url: str, output_path: Path) -> None:
    """Download an image URL without overwriting an existing file."""
    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite existing file: {output_path}')

    response = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    if not response.headers.get('Content-Type', '').startswith('image/'):
        raise ValueError('Profile image response did not contain an image.')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_size = int(response.headers.get('Content-Length', 0))
    with output_path.open('wb') as image_file, tqdm(
        total=total_size or None, unit='B', unit_scale=True, desc=output_path.name
    ) as progress:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                image_file.write(chunk)
                progress.update(len(chunk))

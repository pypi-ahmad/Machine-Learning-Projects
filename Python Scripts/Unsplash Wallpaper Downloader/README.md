# Unsplash Wallpaper Downloader

Platform-specific scripts that download an image from the configured Unsplash random-image URL and set it as the desktop wallpaper.

## Setup

```powershell
uv sync
```

## Run

On Windows:

```powershell
uv run python background_windows.py
```

On Linux, with the Nitrogen wallpaper manager installed:

```powershell
uv run python background_linux.py
```

## Important behavior

Running either script downloads an image, writes `random.jpg` in the current working directory, and changes the active desktop wallpaper immediately. It overwrites an existing `random.jpg` without prompting. Review the source URL and use a disposable working folder if you do not want to replace a local image.

The image source is an external service. Its availability, content, redirects, and usage terms are controlled by that provider. No API key is configured by these scripts.

## Dependencies

- Python 3.14+
- requests
- Linux only: Nitrogen

# Random Unsplash Wallpaper Setter

> Download a random image from Unsplash and automatically set it as your desktop wallpaper on Linux or Windows.

## Overview

This project provides two platform-specific Python scripts that download a random image from the Unsplash random image API and set it as the desktop background. The Linux version uses the Nitrogen wallpaper manager, while the Windows version uses the Windows API via `ctypes`.

## Features

- Downloads a random high-quality image from `https://source.unsplash.com/random`
- Saves the image as `random.jpg` in the current working directory
- **Linux**: Sets wallpaper using Nitrogen (`nitrogen --set-auto`)
- **Windows**: Sets wallpaper using `ctypes.windll.user32.SystemParametersInfo`
- Windows script auto-detects 32-bit vs 64-bit Python to call the correct API variant (`SystemParametersInfoW` vs `SystemParametersInfoA`)
- Error handling on the Windows script

## Project Structure

```
Write_a_script_to_download_a_random_image_from_unsplash_and_set_it_as_wallpaper/
├── background_linux.py      # Linux wallpaper setter
├── background_windows.py    # Windows wallpaper setter
└── README.md
```

## Requirements

- Python 3.x
- `requests`

### Platform-specific

- **Linux**: [Nitrogen](https://wiki.archlinux.org/index.php/Nitrogen) wallpaper manager installed
- **Windows**: No additional tools (uses built-in `ctypes`)

## Installation

```bash
cd "Write_a_script_to_download_a_random_image_from_unsplash_and_set_it_as_wallpaper"
pip install requests
```

## Usage

### Linux

```bash
python background_linux.py
```

### Windows

```bash
python background_windows.py
```

Both scripts will download a random image and immediately set it as the desktop wallpaper.

## How It Works

### Linux (`background_linux.py`)
1. **`download(url, file_name)`** — Makes a GET request to the Unsplash random URL and writes the binary response content to `random.jpg`.
2. **`setup(pathtofile)`** — Calls `nitrogen --set-auto` via `os.system()` with the full path to the downloaded image.

### Windows (`background_windows.py`)
1. **`download(url, file_name)`** — Same as Linux version.
2. **`is_64bit()`** — Checks `sys.maxsize > 2**32` to determine architecture.
3. **`setup(pathtofile, version)`** — Uses `ctypes.windll.user32.SystemParametersInfoW` (64-bit) or `SystemParametersInfoA` (32-bit) with `SPI_SETDESKWALLPAPER` (value `20`) to set the wallpaper.

## Configuration

| Value | Default | Location |
|-------|---------|----------|
| Image URL | `https://source.unsplash.com/random` | Top of both scripts |
| Output filename | `random.jpg` | Top of both scripts |

## Limitations

- The Unsplash random source URL (`source.unsplash.com/random`) may be deprecated or rate-limited; no API key is used
- Overwrites `random.jpg` on every run without backup
- Linux script uses `os.system()` for shell commands (not subprocess)
- Windows `setup()` function signature has an unused `version` parameter
- Windows script raises `NotImplementedError` on any exception, which is misleading
- No image format validation; the response is assumed to always be a valid JPEG
- No command-line arguments for customizing resolution or search terms

## Security Notes

No API keys or credentials are used. The Unsplash random endpoint is accessed without authentication.

## License

Not specified.

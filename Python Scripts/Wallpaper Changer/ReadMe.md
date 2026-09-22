# Wallpaper Changer

Download a random landscape wallpaper from Unsplash and, only when requested, apply it as the Windows desktop wallpaper. A second command handles an image already stored locally.

## Requirements

- Windows 11 for applying a desktop wallpaper
- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)
- An Unsplash access key in `UNSPLASH_ACCESS_KEY` for downloads

## Install

```powershell
cd "Python Scripts\Wallpaper Changer"
uv sync
```

Set the access key in the environment available to the process that runs the command. Do not put it in source code or a committed `.env` file.

```powershell
$env:UNSPLASH_ACCESS_KEY = "your-access-key"
```

## Download a wallpaper

Download a random landscape image without changing the desktop:

```powershell
uv run python .\wallpapers.py --query "mountain landscape" --output .\wallpaper.jpg
```

Apply the downloaded image as the Windows wallpaper:

```powershell
uv run python .\wallpapers.py --query "mountain landscape" --output .\wallpaper.jpg --apply
```

Preview the planned operation without calling Unsplash or changing the desktop:

```powershell
uv run python .\wallpapers.py --query "mountain landscape" --apply --dry-run
```

## Use a local image

Preview a local-image operation:

```powershell
uv run python .\test.py "C:\Pictures\wallpaper.jpg"
```

Pass `--apply` only when ready to change the desktop wallpaper:

```powershell
uv run python .\test.py "C:\Pictures\wallpaper.jpg" --apply
```

## Notes

- Downloads use the Unsplash API and require network access.
- The main command is one-shot. Use Windows Task Scheduler if you want a scheduled rotation.
- The tools never change the desktop unless `--apply` is supplied.

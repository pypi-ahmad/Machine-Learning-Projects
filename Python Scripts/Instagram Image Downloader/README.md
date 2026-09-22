# Instagram Image Downloader

Preview or explicitly download images currently visible on an Instagram profile page. The default command does not open Chrome, log in, contact Instagram, or write files.

## Preview

```powershell
uv sync --no-config
uv run --no-config python instagram.py example_account
```

## Download

Use the apply form only for content you are authorized to save. It requires an exact confirmation phrase and a new output directory:

```powershell
uv run --no-config python instagram.py example_account `
  --output downloads\example_account `
  --max-images 10 `
  --apply --confirm DOWNLOAD_IMAGES
```

The script asks for credentials only after confirmation and does not save them. Chrome is used to load the profile; only image URLs present in the loaded page are downloaded. Existing output directories are rejected to prevent accidental overwrites.

Instagram can change its interface or restrict automated access. Follow applicable rules, permissions, and account policies. The live login and download flow was not run during local verification.

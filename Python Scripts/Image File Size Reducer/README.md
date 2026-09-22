# Image File Size Reducer

Resize one image from the command line. Smaller dimensions generally produce a smaller file; JPEG output also supports an explicit quality setting.

## Run

```powershell
uv sync --no-config
uv run --no-config python reduce_image_size.py input.jpg resized.jpg --scale 5 --quality 85
```

`--scale` divides both dimensions. For example, `--scale 5` turns a 2,500 x 1,500 image into 500 x 300 pixels. `--quality` applies only when the output filename ends in `.jpg` or `.jpeg`.

The script requires different input and output paths and creates the output directory when needed. It does not open a GUI window, so it is suitable for headless use.

## Included sample

The folder includes `input.jpg` and a previously generated `resized_output_image.jpg`. To generate a fresh copy without replacing the included output, use another destination name:

```powershell
uv run --no-config python reduce_image_size.py input.jpg reduced-copy.jpg
```

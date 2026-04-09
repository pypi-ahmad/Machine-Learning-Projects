# Wikipedia Search Wordcloud

> Generate a word cloud image from any Wikipedia article via an interactive CLI prompt.

## Overview

This script prompts the user for a search topic, fetches the corresponding Wikipedia article, and generates a word cloud visualization from the article content. The resulting image is saved as a PNG file and can optionally be displayed on screen.

## Features

- Interactive CLI prompt for Wikipedia topic search
- Fetches full Wikipedia article content via the `wikipedia` Python library
- Generates a word cloud with up to 200 words on a black background (600×350 px)
- Filters common English stopwords plus the `==` section header artifact
- Saves the word cloud as `wordcloud.png` in the current directory
- Optional on-screen display using `matplotlib`
- Error handling for failed Wikipedia searches

## Project Structure

```
Wikipedia_search_wordcloud/
├── wiki-search-cloud.py      # Main script
├── requirements.txt           # Pinned dependencies
├── wordcloud.png              # Example output image
├── script_execution.jpg       # Screenshot of script execution
└── README.md
```

## Requirements

- Python 3.x
- `wordcloud` (1.8.0)
- `matplotlib` (3.3.1)
- `wikipedia` (1.4.0)
- `Pillow` (7.2.0)
- `numpy` (1.19.1)
- `beautifulsoup4` (4.9.1)
- `requests` (2.24.0)

All dependencies are pinned in `requirements.txt`.

## Installation

```bash
cd "Wikipedia_search_wordcloud"
pip install -r requirements.txt
```

## Usage

```bash
python wiki-search-cloud.py
```

The script will:
1. Prompt: `What do you want to search:`
2. Fetch the Wikipedia article for your query
3. Save `wordcloud.png` to the current directory
4. Ask: `Do you wish to see the output(y/n):`  — enter `y` to display the image via matplotlib

## How It Works

1. **`gen_cloud(topic)`** — Calls `wikipedia.page(topic).content` to get the full article text. Builds a `WordCloud` object with STOPWORDS filtering (plus `==` removal), max 200 words, black background, 600×350 resolution.
2. **`save_cloud(wordcloud)`** — Saves the word cloud to `./wordcloud.png` using `WordCloud.to_file()`.
3. **`show_cloud(wordcloud)`** — Displays the image using `matplotlib.pyplot.imshow()` with bilinear interpolation and hidden axes.

## Configuration

- **Max words**: Hardcoded to `200` in the `WordCloud()` constructor
- **Background color**: Hardcoded to `"black"`
- **Image dimensions**: Hardcoded to `width=600, height=350`
- **Output filename**: Hardcoded to `./wordcloud.png`

Edit these values directly in `wiki-search-cloud.py` if needed.

## Limitations

- Uses a bare `except` clause that catches all exceptions, only printing a generic error message
- The `wikipedia` library may raise disambiguation errors for ambiguous queries; these are silently caught
- Output filename is hardcoded — running the script again overwrites the previous word cloud
- No command-line argument support; interaction is prompt-based only
- Dependency versions in `requirements.txt` are dated (2020 era)

## Security Notes

No security concerns identified.

## License

Not specified.

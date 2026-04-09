# Wikipedia Summary GUI

> A Tkinter-based GUI application that fetches and displays Wikipedia article summaries using the MediaWiki API.

## Overview

This application provides a graphical interface where users can enter a topic keyword and retrieve the Wikipedia summary for that topic. It uses the `pymediawiki` library to query the MediaWiki API and displays the results in a scrollable text area.

## Features

- Tkinter GUI with entry field, search button, and scrollable text area
- Fetches Wikipedia article summaries via `pymediawiki`
- Error handling with popup error dialogs (`showerror`)
- Scrollbar for long summaries
- Fixed window size (770x650) with non-resizable layout

## Project Structure

```
Wikipedia-Summary-GUI/
├── README.md
├── requirements.txt
└── summary.py
```

## Requirements

- Python 3.x
- `pymediawiki` — Python wrapper for the MediaWiki API
- `tkinter` (stdlib) — GUI framework

## Installation

```bash
cd Wikipedia-Summary-GUI
pip install -r requirements.txt
```

## Usage

```bash
python summary.py
```

1. Enter a topic in the text field (e.g., "Python programming language").
2. Click **Get Summary**.
3. The Wikipedia summary will appear in the text area below.
4. Enter a new topic and click again to get a different summary (the text area is cleared automatically).

## How It Works

1. Creates a Tkinter window (770x650, non-resizable, dark grey background).
2. The top frame contains an `Entry` widget for keyword input and a **Get Summary** button.
3. The bottom frame contains a `Text` widget with a `Scrollbar` for displaying summaries.
4. `get_summary()` is called when the button is clicked:
   - Clears the text area with `answer.delete(1.0, END)`.
   - Gets the keyword from the entry field.
   - Calls `wikipedia.page(topic)` using `MediaWiki()` to fetch the page.
   - Inserts `p.summary` into the text area.
   - On error, displays a popup via `showerror("Error", error)`.

## Configuration

No configuration files or environment variables. The MediaWiki API is accessed with default settings (English Wikipedia).

## Limitations

- No language selection — defaults to English Wikipedia only.
- No loading indicator while fetching data from the API.
- The `MediaWiki()` instance is created at module level as a global variable named `wikipedia`.
- Disambiguation pages or ambiguous queries may raise errors.
- No input validation — empty searches will cause an error.
- The window is non-resizable at a fixed 770x650 size.

## Security Notes

No sensitive data or credentials in the code.

## License

Not specified.

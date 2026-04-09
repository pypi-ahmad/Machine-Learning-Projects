# Dictionary GUI

## Overview

A GUI-based dictionary application built with Python's Tkinter library and the PyDictionary package. Users can type a word and retrieve its meaning (Noun, Verb, or Adjective) displayed directly in the window.

**Type:** GUI Application

## Features

- Text input field for entering words to look up
- "Search Word" button to trigger the lookup
- Displays the first meaning found, prioritizing: Noun → Verb → Adjective
- Error popup via `messagebox.showinfo` when an invalid or unrecognized word is entered
- Layout with centered heading ("DICTIONARY" in blue, Helvetica 35 bold)
- Fixed window size of 500×400 pixels

## Dependencies

- `tkinter` (Python standard library)
- `PyDictionary` — word definition lookup

### requirements.txt

```
PyDictionary
```

## How It Works

1. The application creates a Tkinter window (500×400) with a heading label, an entry field, a search button, and a meaning display label.
2. When the user clicks "Search Word", the `getMeaning()` function is called.
3. `PyDictionary().meaning()` is invoked with the entered word, returning a dictionary of parts of speech mapped to definitions.
4. The function checks for `Noun`, `Verb`, then `Adjective` keys in the response, displaying the first definition found.
5. If the response is `None` or the word is unrecognized, an error messagebox is shown.

## Project Structure

```
Dictionary-GUI/
├── dictionary.py      # Main application script
├── requirements.txt   # Python dependencies
└── README.md
```

## Setup & Installation

1. Ensure Python 3.x and pip are installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

```bash
python dictionary.py
```

## Configuration

No configuration required.

## Testing

No formal test suite present.

## Limitations

- Only displays the first definition for one part of speech (Noun, Verb, or Adjective); other parts of speech (e.g., Adverb, Pronoun) are not handled and will display "Invalid word".
- If a word has definitions only under a part of speech other than Noun, Verb, or Adjective, it shows "Invalid word" even though the word exists.
- The `meaning` variable may be referenced before assignment if `response` is truthy but contains no recognized keys and is not `None`.
- Requires an active internet connection (PyDictionary fetches definitions from online sources).
- No search history or autocomplete functionality.



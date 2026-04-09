# Language Translator

> An interactive command-line translator supporting 16 languages, powered by the Google Translate API via the `googletrans` library.

## Overview

This script provides a CLI-based translation tool. Users select a target language from 16 supported options using language codes, then enter text to translate in a continuous loop. The script displays the translated text, its pronunciation, and the detected source language.

## Features

- Supports 16 target languages: Bangla, English, Korean, French, German, Hebrew, Hindi, Italian, Japanese, Latin, Malay, Nepali, Russian, Arabic, Chinese, Spanish
- Interactive language code selection with a viewable options menu (type `options`)
- Input validation for language codes
- Continuous translation loop — translate multiple texts without restarting
- Displays translated text, pronunciation, and detected source language
- Exit by typing `close`

## Project Structure

```
Language_translator/
├── requirements.txt
├── Shot.png
└── translator.py
```

## Requirements

- Python 3.x
- `googletrans==3.0.0`

## Installation

```bash
cd "Language_translator"
pip install -r requirements.txt
```

## Usage

```bash
python translator.py
```

**Example session:**
```
Please input desired language code. To see the language code list enter 'options'
options
Code : Language
bn => Bangla
en => English
ko => Koren
...
Please input desired language code. To see the language code list enter 'options'
fr
You have selected French

Write the text you want to translate:
To exit the program write 'close'
Hello, how are you?

French translation: Bonjour comment allez-vous?
Pronunciation : ...
Translated from : English
```

### Supported Language Codes

| Code | Language |
|------|----------|
| bn   | Bangla   |
| en   | English  |
| ko   | Korean   |
| fr   | French   |
| de   | German   |
| he   | Hebrew   |
| hi   | Hindi    |
| it   | Italian  |
| ja   | Japanese |
| la   | Latin    |
| ms   | Malay    |
| ne   | Nepali   |
| ru   | Russian  |
| ar   | Arabic   |
| zh   | Chinese  |
| es   | Spanish  |

## How It Works

1. **Language Selection Loop:** Prompts the user for a language code. Typing `options` displays the full language code table. The loop continues until a valid code is entered.
2. **Translation Loop:** Prompts the user for text input. Typing `close` exits the program. Otherwise, it calls `translator.translate(string, dest=user_code)` from the `googletrans` library.
3. **Output:** Prints the translated text (`translated.text`), pronunciation (`translated.pronunciation`), and source language (looked up from the local `language` dictionary).

## Configuration

- **Supported languages:** Defined in the `language` dictionary at the top of the script. Add or remove entries to change available languages.

## Limitations

- The `googletrans` library (version 3.0.0) is an unofficial Google Translate API wrapper and can break if Google changes its internal API.
- Only 16 languages are supported, though `googletrans` itself supports many more — the script's hardcoded dictionary limits the selection.
- Source language detection only prints the language name if it matches one of the 16 in the local dictionary; otherwise it prints nothing for the source.
- There is a typo in the language dictionary: "Koren" should be "Korean".
- No error handling for translation failures or network issues.
- Pronunciation may return `None` for some languages.

## Security Notes

No security concerns identified.

## License

Not specified.

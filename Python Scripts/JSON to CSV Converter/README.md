# Convert JSON to CSV

## Overview

A Python script that reads a JSON file containing an array of objects and converts it into a CSV file. Uses only the Python standard library.

**Type:** CLI Utility

## Features

- Reads a JSON array from `input.json`
- Extracts column headers automatically from the keys of the first JSON object
- Writes comma-separated values to `output.csv`
- Exception handling with error message output

## Dependencies

- Python 3.x (no external dependencies)
- `json` — standard library module

## How It Works

1. Opens and reads `input.json` using `json.loads()`.
2. Extracts the keys from the first object in the JSON array to build the CSV header row.
3. Iterates through each object, formatting values as comma-separated rows. The fields accessed are `Name`, `age`, and `birthyear` (hardcoded in the f-string).
4. Writes the header and all rows to `output.csv`.
5. Catches and prints any exceptions that occur.

## Project Structure

```
Convert_JSON_to_CSV/
├── converter.py    # Main conversion script
├── input.json      # Sample JSON input (array of objects with Name, age, birthyear)
├── output.csv      # Generated CSV output
└── README.md
```

## Setup & Installation

No installation required. Only a working Python 3 interpreter is needed.

## How to Run

1. Place your JSON data in `input.json` (must be an array of objects).
2. Run:
   ```bash
   python converter.py
   ```
3. The output will be written to `output.csv` in the same directory.

## Testing

No formal test suite present.

## Limitations

- The field extraction in the data rows is hardcoded to `Name`, `age`, and `birthyear` — it does not dynamically extract values for arbitrary JSON keys.
- The CSV header is generated dynamically from keys, but the row values are not, creating an inconsistency.
- Does not handle nested JSON objects or arrays within the data.
- No command-line arguments — input and output filenames are hardcoded.
- No CSV library usage; commas inside values would break the output.

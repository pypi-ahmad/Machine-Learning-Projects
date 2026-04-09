# Lyrics Genius API

> A Python script that downloads song lyrics for specified artists using the Genius API and saves them to a text file.

## Overview

This script uses the `lyricsgenius` library to search for artists on the Genius music platform, retrieve their most popular songs' lyrics, and write them to a text file. Users provide a filename and a list of artist names interactively, and the script fetches up to 3 songs per artist by default.

## Features

- Fetches song lyrics from the Genius API via the `lyricsgenius` library
- Interactive prompts for output filename and artist names
- Searches for multiple artists in a single run (space-separated input)
- Retrieves songs sorted by popularity
- Skips non-song entries (e.g., tracklists, album art)
- Excludes remix and live versions via configurable terms
- Removes section headers (e.g., [Chorus], [Bridge]) from lyrics
- Writes all lyrics to a text file with a custom delimiter (`<|endoftext|>`)

## Project Structure

```
Lyrics_Genius_API/
├── lyrics.py
└── lyrics.txt
```

## Requirements

- Python 3.x
- `lyricsgenius`
- A Genius API Client Access Token (free — sign up at [genius.com/api-clients](https://genius.com/api-clients))

## Installation

```bash
cd "Lyrics_Genius_API"
pip install lyricsgenius
```

## Usage

1. **Get a Genius API token:** Sign up at [Genius API Clients](https://genius.com/api-clients) and generate a Client Access Token.
2. **Add your token:** Replace `'Client_Access_Token_Goes_Here'` in `lyrics.py` with your actual token.
3. **Run the script:**

```bash
python lyrics.py
```

4. **Interactive prompts:**
   - Enter a filename (or press Enter for default `Lyrics.txt`).
   - Enter artist names separated by spaces (e.g., `Eminem Drake Adele`).

5. The lyrics will be saved to the specified file.

## How It Works

1. **Setup:** Creates a `Genius` API client with the provided access token. Configures it to skip non-songs, exclude terms containing "(Remix)" or "(Live)", and remove section headers.
2. **`get_lyrics(arr, max_song)`**: For each artist name in the list:
   - Calls `genius.search_artist(name, max_songs=max_song, sort='popularity')` to find the artist and their top songs.
   - Extracts the `.lyrics` attribute from each song object.
   - Writes all lyrics to the output file, separated by the delimiter `<|endoftext|>`.
   - Prints the number of songs grabbed for each artist.
3. **Default max_songs:** Set to 3 in the function call `get_lyrics(artists, 3)`.

## Configuration

- **API Token:** Must be set in the `lg.Genius('Client_Access_Token_Goes_Here')` call.
- **Max songs per artist:** Currently set to `3` in the `get_lyrics(artists, 3)` call.
- **Excluded terms:** `["(Remix)", "(Live)"]` — songs with these terms in their title are skipped.
- **Output delimiter:** `<|endoftext|>` — used to separate lyrics from different songs in the output file.
- **Default filename:** `Lyrics.txt` (used when user presses Enter without input).

## Limitations

- Artist names are split by spaces, so multi-word artist names (e.g., "Kanye West") cannot be entered — each word is treated as a separate artist.
- The file is opened with `open(filename, "w+")` but never explicitly closed (no `with` statement or `file.close()` at the end).
- Bare `except` blocks hide specific API errors or rate-limiting issues.
- The `get_lyrics` function returns the counter `c` but the return value is never used.
- No validation of the API token before making requests.

## Security Notes

- **Hardcoded API token placeholder:** The script contains a placeholder for a Genius API token. Ensure you do not commit your actual token to version control. Consider using environment variables instead.

## License

Not specified.

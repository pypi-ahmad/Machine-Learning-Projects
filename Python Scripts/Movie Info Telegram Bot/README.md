# Movie Info Telegram Bot

> A Telegram bot that scrapes IMDB to provide movie genres, ratings, and genre-based movie lists via chat commands.

## Overview

This bot uses the `python-telegram-bot` library to interact with users on Telegram. It scrapes IMDB search results to provide movie information (genre and rating) by name, and can list top movies/TV shows belonging to a specified genre. The bot token is loaded from a `.env` file using `python-decouple`.

## Features

- `/start` command — displays bot capabilities and usage instructions
- `/help` command — sends a help message
- `/name <movie_name>` — returns up to 3 IMDB search results with title, genre, rating, and IMDB link
- `/genre <genre_name>` — returns a list of top movies/TV shows for a given IMDB genre
- Logging of bot activity
- Error logging for failed updates

## Project Structure

```
Movie-Info-Telegram-Bot/
├── bot.py             # Main bot script with all handlers and scraping logic
├── requirements.txt   # Python dependencies (pinned versions)
└── .env.example       # Template for environment variables
```

## Requirements

- Python 3.x
- python-telegram-bot == 13.1
- beautifulsoup4 == 4.9.3
- requests == 2.25.1
- python-decouple == 3.4
- APScheduler == 3.6.3
- certifi, python-dateutil, pytz, six, soupsieve, tornado, urllib3 (transitive dependencies)

## Installation

```bash
cd Movie-Info-Telegram-Bot
pip install -r requirements.txt
```

## Usage

### 1. Create a Telegram Bot

1. Open Telegram and message [@BotFather](https://t.me/BotFather)
2. Send `/start`, then `/newbot`
3. Choose a name and username for your bot
4. Copy the API token provided

### 2. Configure Environment

Create a `.env` file based on the example:

```bash
cp .env.example .env
```

Edit `.env` and paste your token:

```
API_KEY = "your-telegram-bot-token-here"
```

### 3. Run the Bot

```bash
python bot.py
```

### 4. Chat with the Bot

- `/start` — introduction and usage guide
- `/name The Dark Knight` — get genre and rating for the movie
- `/genre comedy` — get a list of comedy movies/shows from IMDB

## How It Works

### Bot Architecture
- Uses `python-telegram-bot` `Updater` with polling to receive messages
- Registers `CommandHandler`s for `/start`, `/help`, `/name`, and `/genre`
- All handlers receive `update` and `context` via the context-based callback API

### `/name` Command (`name()` → `get_info()`)
1. Extracts movie name from the message text (characters after `/name `)
2. Searches IMDB (`/find?q=<movie>`)
3. Iterates through results, follows up to 3 unique `/title/` links
4. For each result, extracts: title, genre (via regex on genre links), IMDB rating (via `<strong>` tag), and the IMDB URL
5. Returns formatted text to the user

### `/genre` Command (`genre()`)
1. Extracts genre name from message text (characters after `/genre `)
2. Queries IMDB's advanced search (`/search/title/?genres=<genre>`)
3. Parses all `<a>` tags matching title links via regex
4. Returns the list of movie titles, or an error if the genre is invalid

## Configuration

| Variable | Location | Description |
|----------|----------|-------------|
| `API_KEY` | `.env` file | Telegram Bot API token from BotFather |

## Limitations

- Scrapes IMDB HTML directly — will break if IMDB changes their page structure
- Bare `except` clauses throughout suppress all exceptions silently
- The `/name` command uses string slicing (`str(update.message.text)[6:]`) rather than proper argument parsing
- The `/genre` command uses slicing (`str(update.message.text)[7:]`) — will break if command prefix changes
- No rate limiting on IMDB requests
- Uses `python-telegram-bot` v13.1 which is outdated (v20+ uses async)
- The `help` function name shadows Python's built-in `help()`
- The `rstring` variable may be referenced before assignment if no rating `<strong>` tag is found

## Security Notes

- The `.env.example` file contains a placeholder token — ensure the actual `.env` file is in `.gitignore`
- The API key is loaded via `python-decouple` rather than hardcoded, which is good practice

## License

Not specified.

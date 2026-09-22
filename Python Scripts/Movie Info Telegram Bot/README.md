# Movie Info Telegram Bot

Telegram bot that searches public IMDb pages for movie details and genre listings.

## Setup

Requirements: Python 3.11+ and a Telegram bot token from [@BotFather](https://t.me/BotFather).

```powershell
cd "Movie Info Telegram Bot"
uv sync
Copy-Item .env.example .env
```

Edit the local `.env` file and set `TELEGRAM_BOT_TOKEN` to the BotFather token. Keep `.env` private; it must never be committed.

## Run

```powershell
uv run python bot.py
```

Commands:

- `/start`: Show available commands.
- `/help`: Show examples.
- `/name MOVIE TITLE`: Return details for up to three IMDb matches.
- `/genre GENRE`: Return up to ten IMDb titles for a genre, for example `/genre comedy`.

## Behavior

The bot uses the async `python-telegram-bot` polling API. Its IMDb work runs outside Telegram's event loop, uses a 20-second request timeout, and reports a brief chat-safe failure message if IMDb is unavailable.

IMDb can change its markup, block or limit automated requests, or return incomplete information. This bot does not bypass access controls, rate limits, paywalls, or CAPTCHA checks.

## Project files

```text
Movie Info Telegram Bot/
├── .env.example
├── bot.py
├── pyproject.toml
└── uv.lock
```

"""Telegram bot that returns public IMDb search and genre results."""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from decouple import config
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes


IMDB_URL = "https://www.imdb.com"
REQUEST_TIMEOUT_SECONDS = 20
MAX_RESULTS = 3
USER_AGENT = "Movie-Info-Telegram-Bot/0.1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def fetch_soup(path: str, params: dict[str, str]) -> BeautifulSoup:
    """Fetch one IMDb HTML page with a timeout and explicit user agent."""
    response = requests.get(
        urljoin(IMDB_URL, path),
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def movie_details(soup: BeautifulSoup, url: str) -> str:
    """Format visible title, genres, and rating from one IMDb title page."""
    title = soup.select_one("h1")
    rating = soup.select_one("[data-testid='hero-rating-bar__aggregate-rating__score'] span")
    genres = soup.select("a[href*='/search/title/?genres=']")
    title_text = title.get_text(" ", strip=True) if title else "Untitled"
    rating_text = rating.get_text(" ", strip=True) if rating else "Not available"
    genre_text = ", ".join(genre.get_text(" ", strip=True) for genre in genres) or "Not available"
    return f"{title_text}\nGenres: {genre_text}\nIMDb rating: {rating_text}\n{url}"


def search_movies(query: str) -> list[str]:
    """Return details for up to three unique IMDb title search results."""
    soup = fetch_soup("/find/", {"q": query})
    links = soup.select("a[href*='/title/']")
    urls: list[str] = []
    for link in links:
        href = link.get("href", "")
        match = re.search(r"/title/(tt\d+)/", href)
        if match:
            url = urljoin(IMDB_URL, match.group())
            if url not in urls:
                urls.append(url)
        if len(urls) == MAX_RESULTS:
            break
    return [movie_details(fetch_soup(url, {}), url) for url in urls]


def genre_movies(genre: str) -> list[str]:
    """Return up to ten visible title names for a valid IMDb genre query."""
    if not re.fullmatch(r"[a-z-]+", genre):
        raise ValueError("Genre may contain lowercase letters and hyphens only.")
    soup = fetch_soup("/search/title/", {"genres": genre})
    titles = soup.select("a.ipc-title-link-wrapper h3")
    return [re.sub(r"^\d+\.\s*", "", title.get_text(" ", strip=True)) for title in titles[:10]]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Describe bot commands."""
    del context
    await update.effective_message.reply_text(
        "Use /name MOVIE TITLE for up to three IMDb matches.\n"
        "Use /genre GENRE for up to ten IMDb genre results."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show brief command help."""
    del context
    await update.effective_message.reply_text("Example: /name The Dark Knight\nExample: /genre comedy")


async def name_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reply with IMDb details for a title search."""
    query = " ".join(context.args).strip()
    if not query:
        await update.effective_message.reply_text("Usage: /name MOVIE TITLE")
        return
    try:
        results = await asyncio.to_thread(search_movies, query)
    except requests.RequestException:
        logger.exception("IMDb title search failed")
        await update.effective_message.reply_text("IMDb could not be reached. Please try again later.")
        return
    await update.effective_message.reply_text("\n\n".join(results) if results else "No IMDb title matches found.")


async def genre_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reply with IMDb title names for a genre."""
    genre = "-".join(context.args).lower()
    if not genre:
        await update.effective_message.reply_text("Usage: /genre GENRE")
        return
    try:
        results = await asyncio.to_thread(genre_movies, genre)
    except ValueError as error:
        await update.effective_message.reply_text(str(error))
        return
    except requests.RequestException:
        logger.exception("IMDb genre search failed")
        await update.effective_message.reply_text("IMDb could not be reached. Please try again later.")
        return
    await update.effective_message.reply_text("\n".join(results) if results else "No IMDb titles found for that genre.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Telegram update errors without exposing tokens."""
    logger.error("Update %s caused %s", update, context.error)


def main() -> None:
    """Read the local token and start Telegram polling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    token = config("TELEGRAM_BOT_TOKEN", default="").strip()
    if not token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in .env before starting the bot.")

    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("name", name_command))
    application.add_handler(CommandHandler("genre", genre_command))
    application.add_error_handler(error_handler)
    application.run_polling()


if __name__ == "__main__":
    main()

"""News Dashboard — Streamlit app.

Fetch top headlines from NewsAPI (free tier) or RSS feeds.
Filter by category, keyword search, and save articles.

Usage:
    streamlit run main.py
    Set NEWS_API_KEY env var for NewsAPI, or use RSS mode.
"""

import json
import os
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="News Dashboard", layout="wide")
st.title("📰 News Dashboard")

SAVED_FILE = Path("saved_articles.json")

RSS_FEEDS = {
    "BBC World":       "http://feeds.bbci.co.uk/news/world/rss.xml",
    "Reuters":         "https://feeds.reuters.com/reuters/topNews",
    "Hacker News":     "https://news.ycombinator.com/rss",
    "TechCrunch":      "https://techcrunch.com/feed/",
    "CNN":             "http://rss.cnn.com/rss/edition.rss",
}


def load_saved() -> list[dict]:
    if SAVED_FILE.exists():
        try:
            return json.loads(SAVED_FILE.read_text())
        except Exception:
            pass
    return []


def save_article(article: dict):
    saved = load_saved()
    if not any(a["url"] == article["url"] for a in saved):
        saved.append(article)
        SAVED_FILE.write_text(json.dumps(saved, indent=2))


@st.cache_data(ttl=600)
def fetch_rss(url: str) -> list[dict]:
    try:
        with urllib.request.urlopen(url, timeout=6) as resp:
            root = ET.fromstring(resp.read())
        articles = []
        for item in root.findall(".//item")[:20]:
            def tag(t): return (item.findtext(t) or "").strip()
            articles.append({
                "title":       tag("title"),
                "description": tag("description")[:200],
                "url":         tag("link"),
                "published":   tag("pubDate"),
                "source":      url,
            })
        return articles
    except Exception as e:
        return []


@st.cache_data(ttl=600)
def fetch_newsapi(api_key: str, query: str = "", category: str = "general",
                  country: str = "us") -> list[dict]:
    params = {"apiKey": api_key, "pageSize": 20}
    if query:
        endpoint = "https://newsapi.org/v2/everything"
        params["q"]       = query
        params["language"]= "en"
    else:
        endpoint = "https://newsapi.org/v2/top-headlines"
        params["category"]= category
        params["country"] = country
    url = endpoint + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=6) as resp:
            data = json.loads(resp.read())
        return [{"title": a["title"] or "", "description": (a["description"] or "")[:200],
                 "url": a["url"], "published": a["publishedAt"],
                 "source": a["source"]["name"]} for a in data.get("articles", [])]
    except Exception:
        return []


# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("Source")
api_key = st.sidebar.text_input("NewsAPI key (optional)", type="password",
                                  value=os.environ.get("NEWS_API_KEY", ""))
mode = "newsapi" if api_key else "rss"

if mode == "rss":
    source_name = st.sidebar.selectbox("RSS Feed", list(RSS_FEEDS.keys()))
else:
    category = st.sidebar.selectbox("Category", ["general","business","technology","science","health","sports","entertainment"])
    country  = st.sidebar.selectbox("Country", ["us","gb","au","ca","in"])

search = st.sidebar.text_input("🔍 Search keyword")

# ── Fetch ────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Headlines", "Saved"])

with tab1:
    if mode == "rss":
        articles = fetch_rss(RSS_FEEDS[source_name])
        st.caption(f"Source: {source_name}")
    else:
        articles = fetch_newsapi(api_key, search, category, country)
        st.caption(f"Source: NewsAPI — {category} / {country}")

    if search and mode == "rss":
        articles = [a for a in articles if search.lower() in a["title"].lower()
                    or search.lower() in a["description"].lower()]

    if not articles:
        st.info("No articles found. Check your connection or API key.")
    else:
        for i, a in enumerate(articles):
            with st.container(border=True):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"**[{a['title']}]({a['url']})**")
                    if a.get("description"):
                        st.caption(a["description"])
                    if a.get("published"):
                        st.caption(f"📅 {a['published'][:16]}  ·  {a.get('source','')}")
                with col2:
                    if st.button("🔖 Save", key=f"save_{i}"):
                        save_article(a)
                        st.success("Saved!")

with tab2:
    saved = load_saved()
    if not saved:
        st.info("No saved articles yet. Click 🔖 Save on any article.")
    else:
        st.caption(f"{len(saved)} saved article(s)")
        for i, a in enumerate(saved):
            with st.expander(a["title"]):
                st.write(a.get("description", ""))
                st.markdown(f"[Read more]({a['url']})")
                if st.button("🗑️ Remove", key=f"del_{i}"):
                    saved.pop(i)
                    SAVED_FILE.write_text(json.dumps(saved, indent=2))
                    st.rerun()

"""Movie Watchlist — Streamlit app.

Track movies to watch, watching, and watched.
Rate, review, filter by genre, and see watch stats.

Usage:
    streamlit run main.py
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Movie Watchlist", layout="wide")
st.title("🎬 Movie Watchlist")

DATA_FILE = Path("movies.json")
STATUSES  = ["Want to Watch", "Watching", "Watched", "Dropped"]
GENRES    = ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Romance",
             "Thriller", "Animation", "Documentary", "Fantasy", "Other"]


def load_movies() -> list[dict]:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text())
        except Exception:
            pass
    return []


def save_movies(movies: list[dict]):
    DATA_FILE.write_text(json.dumps(movies, indent=2))


if "movies" not in st.session_state:
    st.session_state.movies = load_movies()
movies = st.session_state.movies

tab1, tab2, tab3 = st.tabs(["Add Movie", "My List", "Stats & Reviews"])

with tab1:
    with st.form("add_movie"):
        st.subheader("Add a Movie / Show")
        c1, c2    = st.columns(2)
        title     = c1.text_input("Title")
        year      = c2.number_input("Year", 1900, date.today().year + 2, date.today().year)
        c1, c2    = st.columns(2)
        genre     = c1.selectbox("Genre", GENRES)
        status    = c2.selectbox("Status", STATUSES)
        c1, c2    = st.columns(2)
        rating    = c1.slider("Rating (0 = unrated)", 0, 10, 0)
        date_wtch = c2.date_input("Date Watched", value=None)
        director  = st.text_input("Director (optional)")
        review    = st.text_area("Review / Notes", height=80)
        submit    = st.form_submit_button("Add to List", type="primary")

    if submit and title:
        movies.append({
            "title":        title,
            "year":         int(year),
            "genre":        genre,
            "status":       status,
            "rating":       int(rating),
            "date_watched": str(date_wtch) if date_wtch else "",
            "director":     director,
            "review":       review,
            "date_added":   str(date.today()),
        })
        save_movies(movies)
        st.success(f"Added: {title} ({year})")

with tab2:
    if not movies:
        st.info("Your watchlist is empty. Add some movies!")
    else:
        c1, c2    = st.columns(2)
        s_filter  = c1.multiselect("Status", STATUSES, default=STATUSES)
        g_filter  = c2.multiselect("Genre",  GENRES,   default=GENRES)
        search    = st.text_input("Search title")

        filtered = [m for m in movies
                    if m["status"] in s_filter and m["genre"] in g_filter
                    and (not search or search.lower() in m["title"].lower())]
        filtered = sorted(filtered, key=lambda x: x["date_added"], reverse=True)

        rows = []
        for m in filtered:
            stars = ("⭐" * (m["rating"] // 2)) + ("½" if m["rating"] % 2 else "") if m["rating"] else "—"
            rows.append({"Title": m["title"], "Year": m["year"], "Genre": m["genre"],
                         "Status": m["status"], "Rating": f"{m['rating']}/10" if m["rating"] else "—",
                         "Stars": stars, "Watched": m.get("date_watched") or "—"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"{len(filtered)} title(s)")

        # Quick update
        st.subheader("Quick Update")
        titles = [m["title"] for m in movies]
        sel = st.selectbox("Select movie", [""] + titles)
        if sel:
            movie = next(m for m in movies if m["title"] == sel)
            new_status = st.selectbox("New status", STATUSES, index=STATUSES.index(movie["status"]),
                                      key="upd_status")
            new_rating = st.slider("New rating", 0, 10, movie.get("rating", 0), key="upd_rating")
            if st.button("Update", type="primary"):
                movie["status"] = new_status
                movie["rating"] = new_rating
                if new_status == "Watched" and not movie.get("date_watched"):
                    movie["date_watched"] = str(date.today())
                save_movies(movies)
                st.success("Updated!")
                st.rerun()

with tab3:
    if not movies:
        st.info("No movies yet.")
    else:
        watched  = [m for m in movies if m["status"] == "Watched"]
        want     = [m for m in movies if m["status"] == "Want to Watch"]
        watching = [m for m in movies if m["status"] == "Watching"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total",         len(movies))
        c2.metric("Watched",       len(watched))
        c3.metric("Watching",      len(watching))
        c4.metric("Want to Watch", len(want))

        if watched:
            rated   = [m for m in watched if m["rating"] > 0]
            avg_r   = sum(m["rating"] for m in rated) / len(rated) if rated else 0
            st.metric("Average Rating", f"{avg_r:.1f}/10")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("By Genre")
            gc = {}
            for m in movies: gc[m["genre"]] = gc.get(m["genre"], 0) + 1
            st.bar_chart(pd.Series(gc).sort_values(ascending=False))

        with col2:
            st.subheader("By Status")
            sc = {s: sum(1 for m in movies if m["status"] == s) for s in STATUSES}
            st.bar_chart(pd.Series({k: v for k, v in sc.items() if v > 0}))

        if watched:
            st.subheader("Top Rated")
            top = sorted(watched, key=lambda x: x["rating"], reverse=True)[:10]
            for m in top:
                st.markdown(f"**{m['title']}** ({m['year']}) — {m['rating']}/10"
                            + (f"\n> {m['review']}" if m.get("review") else ""))

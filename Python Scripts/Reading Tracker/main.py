"""Reading Tracker — Streamlit app.

Track books you've read, are reading, or want to read.
Log progress, ratings, notes, and see reading stats.

Usage:
    streamlit run main.py
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Reading Tracker", layout="wide")
st.title("📚 Reading Tracker")

DATA_FILE = Path("books.json")
STATUSES  = ["Want to Read", "Reading", "Finished", "Abandoned"]
GENRES    = ["Fiction", "Non-Fiction", "Sci-Fi", "Fantasy", "Biography",
             "Self-Help", "History", "Mystery", "Other"]


def load_books() -> list[dict]:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text())
        except Exception:
            pass
    return []


def save_books(books: list[dict]):
    DATA_FILE.write_text(json.dumps(books, indent=2))


if "books" not in st.session_state:
    st.session_state.books = load_books()
books = st.session_state.books

tab1, tab2, tab3, tab4 = st.tabs(["Add Book", "My Library", "Stats", "Update Progress"])

with tab1:
    with st.form("add_book"):
        st.subheader("Add a Book")
        c1, c2   = st.columns(2)
        title    = c1.text_input("Title")
        author   = c2.text_input("Author")
        c1, c2   = st.columns(2)
        genre    = c1.selectbox("Genre", GENRES)
        status   = c2.selectbox("Status", STATUSES)
        c1, c2, c3 = st.columns(3)
        pages    = c1.number_input("Total Pages", 0, 5000, 0)
        rating   = c2.slider("Rating (0=unrated)", 0, 5, 0)
        date_fin = c3.date_input("Date Finished", value=None)
        notes    = st.text_area("Notes / Review", height=80)
        submit   = st.form_submit_button("Add Book", type="primary")

    if submit and title:
        books.append({
            "title":      title,
            "author":     author,
            "genre":      genre,
            "status":     status,
            "pages":      int(pages),
            "pages_read": int(pages) if status == "Finished" else 0,
            "rating":     int(rating),
            "date_added": str(date.today()),
            "date_finished": str(date_fin) if date_fin else "",
            "notes":      notes,
        })
        save_books(books)
        st.success(f"Added: '{title}' by {author}")

with tab2:
    if not books:
        st.info("No books yet. Add your first book!")
    else:
        status_filter = st.multiselect("Filter by status", STATUSES, default=STATUSES)
        genre_filter  = st.multiselect("Filter by genre",  GENRES,   default=GENRES)
        filtered = [b for b in books
                    if b["status"] in status_filter and b["genre"] in genre_filter]
        filtered = sorted(filtered, key=lambda x: x["date_added"], reverse=True)

        rows = []
        for b in filtered:
            stars = "⭐" * b["rating"] if b["rating"] else "—"
            rows.append({"Title": b["title"], "Author": b["author"], "Genre": b["genre"],
                         "Status": b["status"], "Pages": b["pages"] or "—",
                         "Rating": stars, "Finished": b.get("date_finished") or "—"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"{len(filtered)} book(s)")

with tab3:
    if not books:
        st.info("No books yet.")
    else:
        finished = [b for b in books if b["status"] == "Finished"]
        reading  = [b for b in books if b["status"] == "Reading"]
        want     = [b for b in books if b["status"] == "Want to Read"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Books",   len(books))
        c2.metric("Finished",      len(finished))
        c3.metric("Reading",       len(reading))
        c4.metric("Want to Read",  len(want))

        if finished:
            rated = [b for b in finished if b["rating"] > 0]
            avg_rating = sum(b["rating"] for b in rated) / len(rated) if rated else 0
            total_pages = sum(b["pages"] for b in finished if b["pages"])
            st.metric("Avg Rating", f"{avg_rating:.1f} ⭐")
            st.metric("Total Pages Read", f"{total_pages:,}")

        st.subheader("Books by Genre")
        genre_counts: dict[str, int] = {}
        for b in books:
            genre_counts[b["genre"]] = genre_counts.get(b["genre"], 0) + 1
        st.bar_chart(pd.Series(genre_counts).sort_values(ascending=False))

        st.subheader("Books by Status")
        status_counts = {s: sum(1 for b in books if b["status"] == s) for s in STATUSES}
        st.bar_chart(pd.Series({k: v for k, v in status_counts.items() if v > 0}))

        if finished:
            st.subheader("Rating Distribution")
            rating_counts = {str(i): sum(1 for b in finished if b["rating"] == i)
                             for i in range(1, 6)}
            st.bar_chart(pd.Series(rating_counts))

with tab4:
    in_progress = [b for b in books if b["status"] == "Reading"]
    if not in_progress:
        st.info("No books currently being read.")
    else:
        titles   = [b["title"] for b in in_progress]
        sel_title = st.selectbox("Book", titles)
        book     = next(b for b in in_progress if b["title"] == sel_title)

        pages_read = st.number_input("Pages read so far",
                                     0, max(book["pages"], 1),
                                     value=book.get("pages_read", 0))
        if book["pages"] > 0:
            pct = pages_read / book["pages"] * 100
            st.progress(min(pct / 100, 1.0))
            st.caption(f"{pages_read}/{book['pages']} pages — {pct:.1f}%")

        new_status = st.selectbox("Update status", STATUSES, index=STATUSES.index(book["status"]))
        new_rating = st.slider("Update rating", 0, 5, book.get("rating", 0))
        new_notes  = st.text_area("Notes", value=book.get("notes", ""), height=80)

        if st.button("Save Update", type="primary"):
            book["pages_read"] = int(pages_read)
            book["status"]     = new_status
            book["rating"]     = new_rating
            book["notes"]      = new_notes
            if new_status == "Finished" and not book.get("date_finished"):
                book["date_finished"] = str(date.today())
            save_books(books)
            st.success("Updated!")
            st.rerun()

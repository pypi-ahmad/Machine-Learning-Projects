"""Daily Journal — Streamlit app.

Write, browse, search, and export daily journal entries.
Supports mood tracking and tags.  Data stored as JSON.

Usage:
    streamlit run main.py
"""

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Daily Journal", layout="wide")
st.title("📔 Daily Journal")

DATA_FILE = Path("journal.json")

MOODS = ["😊 Happy", "😌 Calm", "😐 Neutral", "😟 Sad", "😤 Frustrated", "😴 Tired", "🤩 Excited"]


def load_entries() -> list[dict]:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text())
        except Exception:
            pass
    return []


def save_entries(entries: list[dict]) -> None:
    DATA_FILE.write_text(json.dumps(entries, indent=2))


if "entries" not in st.session_state:
    st.session_state.entries = load_entries()

entries = st.session_state.entries

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Write Entry", "Browse", "Stats & Export"])

with tab1:
    st.subheader("New Entry")
    entry_date  = st.date_input("Date", value=date.today())
    entry_title = st.text_input("Title (optional)")
    entry_mood  = st.selectbox("Mood", MOODS)
    entry_text  = st.text_area("Journal entry", height=280, placeholder="Write your thoughts here…")
    entry_tags  = st.text_input("Tags (comma-separated)", placeholder="personal, work, gratitude")

    if st.button("💾 Save Entry", type="primary"):
        if entry_text.strip():
            new_entry = {
                "date":  str(entry_date),
                "title": entry_title.strip(),
                "mood":  entry_mood,
                "text":  entry_text.strip(),
                "tags":  [t.strip() for t in entry_tags.split(",") if t.strip()],
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            }
            # Replace existing entry for the same date if present
            existing_idx = next(
                (i for i, e in enumerate(entries) if e["date"] == str(entry_date)), None
            )
            if existing_idx is not None:
                entries[existing_idx] = new_entry
                st.success("Entry updated.")
            else:
                entries.append(new_entry)
                st.success("Entry saved!")
            entries.sort(key=lambda e: e["date"], reverse=True)
            st.session_state.entries = entries
            save_entries(entries)
        else:
            st.warning("Please write something before saving.")

with tab2:
    if not entries:
        st.info("No entries yet. Start writing!")
    else:
        search  = st.text_input("🔍 Search entries")
        all_tags = sorted({t for e in entries for t in e.get("tags", [])})
        tag_filter = st.multiselect("Filter by tag", all_tags)
        mood_filter = st.multiselect("Filter by mood", MOODS)

        filtered = [
            e for e in entries
            if (not search or search.lower() in e.get("text","").lower() or search.lower() in e.get("title","").lower())
            and (not tag_filter or any(t in e.get("tags",[]) for t in tag_filter))
            and (not mood_filter or e.get("mood") in mood_filter)
        ]

        st.caption(f"Showing {len(filtered)} of {len(entries)} entries")
        for e in filtered:
            with st.expander(f"**{e['date']}** — {e.get('title') or e['mood']}  {e['mood'].split()[0]}"):
                st.write(f"**Mood:** {e['mood']}")
                if e.get("tags"):
                    st.caption("🏷️ " + ", ".join(e["tags"]))
                st.write(e["text"])
                if st.button("🗑️ Delete", key=f"del_{e['date']}"):
                    st.session_state.entries = [x for x in entries if x["date"] != e["date"]]
                    save_entries(st.session_state.entries)
                    st.rerun()

with tab3:
    if not entries:
        st.info("No entries yet.")
    else:
        df = pd.DataFrame(entries)
        df["date"] = pd.to_datetime(df["date"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Entries", len(df))
        c2.metric("First Entry", df["date"].min().date())
        c3.metric("Last Entry",  df["date"].max().date())

        st.subheader("Mood Distribution")
        mood_counts = df["mood"].value_counts()
        st.bar_chart(mood_counts)

        st.subheader("Entries per Month")
        df["month"] = df["date"].dt.to_period("M").astype(str)
        monthly = df.groupby("month").size().rename("Entries")
        st.line_chart(monthly)

        st.divider()
        csv = df.drop(columns=["date"], errors="ignore").to_csv(index=False).encode()
        st.download_button("📥 Export as CSV", csv, "journal_export.csv", "text/csv")

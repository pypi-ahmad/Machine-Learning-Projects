"""Simple Poll Creator — Streamlit app.

Create polls with multiple options, cast votes, and see live results
as bar charts.  Multiple polls supported; data saved locally.

Usage:
    streamlit run main.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Poll Creator", layout="centered")
st.title("🗳️ Simple Poll Creator")

DATA_FILE = Path("polls.json")


def load() -> dict:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text())
        except Exception:
            pass
    return {"polls": {}}


def save(data: dict) -> None:
    DATA_FILE.write_text(json.dumps(data, indent=2))


if "data" not in st.session_state:
    st.session_state.data = load()

data = st.session_state.data

# ---------------------------------------------------------------------------
# Sidebar — create poll
# ---------------------------------------------------------------------------
st.sidebar.header("Create New Poll")
poll_q = st.sidebar.text_input("Question")
opt_str = st.sidebar.text_area("Options (one per line)")

if st.sidebar.button("Create Poll") and poll_q.strip() and opt_str.strip():
    opts = [o.strip() for o in opt_str.splitlines() if o.strip()]
    if len(opts) >= 2:
        data["polls"][poll_q.strip()] = {o: 0 for o in opts}
        save(data)
        st.sidebar.success("Poll created!")
        st.rerun()
    else:
        st.sidebar.error("Need at least 2 options.")

# ---------------------------------------------------------------------------
# Display polls
# ---------------------------------------------------------------------------
polls = data.get("polls", {})

if not polls:
    st.info("No polls yet. Create one using the sidebar.")
    st.stop()

for question, options in list(polls.items()):
    with st.container(border=True):
        st.subheader(question)
        total_votes = sum(options.values())
        tab_vote, tab_results = st.tabs(["Vote", "Results"])

        with tab_vote:
            choice = st.radio("Your vote:", list(options.keys()),
                               key=f"vote_{question}", index=None)
            if st.button("Submit Vote", key=f"submit_{question}"):
                if choice:
                    data["polls"][question][choice] += 1
                    save(data)
                    st.success(f"Voted for: {choice}")
                    st.rerun()
                else:
                    st.warning("Please select an option.")

        with tab_results:
            if total_votes == 0:
                st.info("No votes yet.")
            else:
                vote_df = pd.DataFrame({
                    "Option": list(options.keys()),
                    "Votes":  list(options.values()),
                })
                vote_df["Percent"] = (vote_df["Votes"] / total_votes * 100).round(1)
                st.bar_chart(vote_df.set_index("Option")["Votes"])
                st.dataframe(vote_df, use_container_width=True, hide_index=True)
                st.caption(f"Total votes: {total_votes}")

        if st.button("🗑️ Delete poll", key=f"del_{question}"):
            del data["polls"][question]
            save(data)
            st.rerun()

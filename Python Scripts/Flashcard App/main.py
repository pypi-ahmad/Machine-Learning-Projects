"""Flashcard App — Streamlit app.

Create, study, and review flashcard decks using spaced-repetition
(simple 3-bucket SM2-lite).  Decks are stored as local JSON files.

Usage:
    streamlit run main.py
"""

import json
import random
from datetime import date
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Flashcard App", layout="centered")
st.title("📚 Flashcard App")

DATA_DIR = Path("flashcard_decks")
DATA_DIR.mkdir(exist_ok=True)

SAMPLE_DECK = {
    "name": "Python Basics",
    "cards": [
        {"front": "What is a list comprehension?",
         "back": "A concise way to create lists: [expr for item in iterable if condition]"},
        {"front": "What does len() do?",
         "back": "Returns the number of items in an object (list, string, dict, etc.)"},
        {"front": "How do you open a file in Python?",
         "back": "with open('file.txt', 'r') as f: ..."},
        {"front": "What is a decorator?",
         "back": "A function that wraps another function to add behaviour without modifying it."},
        {"front": "What is the difference between == and is?",
         "back": "== checks value equality; 'is' checks object identity (same memory location)."},
    ],
}


def list_decks() -> list[str]:
    return [p.stem for p in DATA_DIR.glob("*.json")]


def load_deck(name: str) -> dict:
    p = DATA_DIR / f"{name}.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"name": name, "cards": []}


def save_deck(deck: dict) -> None:
    p = DATA_DIR / f"{deck['name']}.json"
    p.write_text(json.dumps(deck, indent=2))


# Ensure sample deck exists
if "Python Basics" not in list_decks():
    save_deck(SAMPLE_DECK)

# ---------------------------------------------------------------------------
# Sidebar — deck management
# ---------------------------------------------------------------------------
st.sidebar.header("Deck Management")
decks = list_decks()

selected_deck = st.sidebar.selectbox("Select deck", decks)
new_deck_name = st.sidebar.text_input("Create new deck")
if st.sidebar.button("Create") and new_deck_name.strip():
    if new_deck_name.strip() not in decks:
        save_deck({"name": new_deck_name.strip(), "cards": []})
        st.sidebar.success(f"Created: {new_deck_name.strip()}")
        st.rerun()

deck = load_deck(selected_deck)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Study", "Add Cards", "Browse Deck"])

with tab1:
    cards = deck.get("cards", [])
    if not cards:
        st.info("This deck has no cards yet. Add some in the 'Add Cards' tab.")
        st.stop()

    # Session state for study
    if "study_idx" not in st.session_state or st.session_state.get("study_deck") != selected_deck:
        order = list(range(len(cards)))
        random.shuffle(order)
        st.session_state.study_order = order
        st.session_state.study_pos   = 0
        st.session_state.study_deck  = selected_deck
        st.session_state.show_back   = False
        st.session_state.correct     = 0
        st.session_state.total       = 0

    pos   = st.session_state.study_pos
    order = st.session_state.study_order

    if pos >= len(order):
        st.balloons()
        st.success(f"🎉 Deck complete!  Score: {st.session_state.correct}/{st.session_state.total}")
        if st.button("🔄 Restart"):
            st.session_state.study_pos  = 0
            st.session_state.show_back  = False
            st.session_state.correct    = 0
            st.session_state.total      = 0
            random.shuffle(st.session_state.study_order)
            st.rerun()
    else:
        card = cards[order[pos]]
        st.write(f"Card **{pos + 1}** of {len(order)}")
        st.progress(pos / len(order))

        with st.container(border=True):
            st.subheader(card["front"])
            if st.session_state.show_back:
                st.divider()
                st.write(card["back"])

        if not st.session_state.show_back:
            if st.button("Show Answer"):
                st.session_state.show_back = True
                st.rerun()
        else:
            col1, col2 = st.columns(2)
            if col1.button("✅ Got it"):
                st.session_state.correct   += 1
                st.session_state.total     += 1
                st.session_state.study_pos += 1
                st.session_state.show_back  = False
                st.rerun()
            if col2.button("❌ Review again"):
                st.session_state.total     += 1
                # Move card to end of queue
                st.session_state.study_order.append(st.session_state.study_order[pos])
                st.session_state.study_pos += 1
                st.session_state.show_back  = False
                st.rerun()

with tab2:
    st.subheader("Add a New Card")
    with st.form("add_card"):
        front = st.text_area("Front (question/term)")
        back  = st.text_area("Back (answer/definition)")
        added = st.form_submit_button("Add Card")
    if added and front.strip() and back.strip():
        deck["cards"].append({"front": front.strip(), "back": back.strip()})
        save_deck(deck)
        st.success("Card added!")
        st.rerun()

with tab3:
    cards = deck.get("cards", [])
    st.write(f"**{len(cards)} card(s)** in deck '{selected_deck}'")
    for i, card in enumerate(cards):
        with st.expander(f"Card {i+1}: {card['front'][:60]}"):
            st.write(f"**Q:** {card['front']}")
            st.write(f"**A:** {card['back']}")
            if st.button(f"Delete card {i+1}", key=f"del_{i}"):
                deck["cards"].pop(i)
                save_deck(deck)
                st.rerun()

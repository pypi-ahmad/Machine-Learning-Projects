"""Meeting Notes App — Streamlit app.

Create, browse, search, and export meeting notes with
attendees, action items, and decisions.  Data stored as JSON.

Usage:
    streamlit run main.py
"""

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Meeting Notes", layout="wide")
st.title("📋 Meeting Notes App")

DATA_FILE = Path("meetings.json")


def load_meetings() -> list[dict]:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text())
        except Exception:
            pass
    return []


def save_meetings(meetings: list[dict]) -> None:
    DATA_FILE.write_text(json.dumps(meetings, indent=2))


if "meetings" not in st.session_state:
    st.session_state.meetings = load_meetings()

meetings = st.session_state.meetings

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["New Meeting", "Browse", "Action Items"])

with tab1:
    st.subheader("Create Meeting Notes")
    col1, col2 = st.columns(2)
    with col1:
        m_title     = st.text_input("Meeting title *")
        m_date      = st.date_input("Date", value=date.today())
        m_time      = st.time_input("Time", value=datetime.now().replace(second=0, microsecond=0).time())
        m_location  = st.text_input("Location / Link", placeholder="Room 101 or zoom.us/j/123")
    with col2:
        m_attendees = st.text_area("Attendees (one per line)", height=120, placeholder="Alice\nBob\nCharlie")
        m_type      = st.selectbox("Meeting type", ["Team", "1:1", "Client", "Brainstorm", "Review", "Other"])

    m_agenda  = st.text_area("Agenda", height=100, placeholder="1. Status update\n2. Blockers\n3. Next steps")
    m_notes   = st.text_area("Notes / Discussion", height=200, placeholder="Key points discussed…")
    m_actions = st.text_area("Action items (one per line: 'owner: task')",
                              height=120, placeholder="Alice: Send report by Friday\nBob: Update dashboard")
    m_decisions = st.text_area("Decisions made", height=80)

    if st.button("💾 Save Meeting Notes", type="primary"):
        if m_title.strip():
            action_items = []
            for line in m_actions.splitlines():
                line = line.strip()
                if ":" in line:
                    owner, _, task = line.partition(":")
                    action_items.append({"owner": owner.strip(), "task": task.strip(), "done": False})
                elif line:
                    action_items.append({"owner": "", "task": line, "done": False})

            new_meeting = {
                "id":         len(meetings),
                "title":      m_title.strip(),
                "date":       str(m_date),
                "time":       m_time.strftime("%H:%M"),
                "location":   m_location.strip(),
                "type":       m_type,
                "attendees":  [a.strip() for a in m_attendees.splitlines() if a.strip()],
                "agenda":     m_agenda.strip(),
                "notes":      m_notes.strip(),
                "actions":    action_items,
                "decisions":  m_decisions.strip(),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            meetings.insert(0, new_meeting)
            st.session_state.meetings = meetings
            save_meetings(meetings)
            st.success(f"Meeting '{m_title.strip()}' saved!")
        else:
            st.warning("Please enter a meeting title.")

with tab2:
    if not meetings:
        st.info("No meetings yet. Create one in the first tab.")
    else:
        search = st.text_input("🔍 Search meetings")
        type_filter = st.multiselect("Meeting type", ["Team","1:1","Client","Brainstorm","Review","Other"])

        shown = [
            m for m in meetings
            if (not search or search.lower() in m["title"].lower()
                or search.lower() in m.get("notes","").lower())
            and (not type_filter or m["type"] in type_filter)
        ]

        for m in shown:
            idx = next(i for i, x in enumerate(meetings) if x["id"] == m["id"])
            with st.expander(f"**{m['date']} {m['time']}** — {m['title']}  ({m['type']})"):
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**Location:** {m.get('location') or '—'}")
                    st.write(f"**Attendees:** {', '.join(m.get('attendees',[]))}")
                with c2:
                    done = sum(1 for a in m.get("actions",[]) if a["done"])
                    total = len(m.get("actions",[]))
                    st.write(f"**Actions:** {done}/{total} complete")

                if m.get("agenda"):
                    st.write("**Agenda:**")
                    st.write(m["agenda"])
                if m.get("notes"):
                    st.write("**Notes:**")
                    st.write(m["notes"])
                if m.get("decisions"):
                    st.info(f"**Decisions:** {m['decisions']}")
                if m.get("actions"):
                    st.write("**Action Items:**")
                    for ai, action in enumerate(m["actions"]):
                        label = f"{'[' + action['owner'] + '] ' if action['owner'] else ''}{action['task']}"
                        checked = st.checkbox(label, value=action["done"],
                                               key=f"act_{m['id']}_{ai}")
                        if checked != action["done"]:
                            meetings[idx]["actions"][ai]["done"] = checked
                            save_meetings(meetings)
                            st.rerun()

                if st.button("🗑️ Delete meeting", key=f"del_{m['id']}"):
                    meetings.pop(idx)
                    save_meetings(meetings)
                    st.rerun()

with tab3:
    st.subheader("All Open Action Items")
    open_actions = []
    for m in meetings:
        for action in m.get("actions", []):
            if not action["done"]:
                open_actions.append({
                    "Meeting": m["title"],
                    "Date":    m["date"],
                    "Owner":   action["owner"] or "—",
                    "Task":    action["task"],
                })
    if not open_actions:
        st.success("No open action items!")
    else:
        df = pd.DataFrame(open_actions)
        st.dataframe(df.sort_values("Date"), use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False).encode()
        st.download_button("📥 Export Action Items CSV", csv, "action_items.csv", "text/csv")

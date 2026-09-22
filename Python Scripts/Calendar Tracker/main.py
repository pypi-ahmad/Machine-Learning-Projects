"""A local Streamlit calendar for creating and reviewing events."""

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st


DATA_FILE = Path(__file__).with_name("events.json")
CATEGORIES = ["Work", "Personal", "Health", "Social", "Other"]
CAT_COLORS = {"Work": "🔵", "Personal": "🟢", "Health": "🔴", "Social": "🟡", "Other": "⚪"}


def is_valid_event(event: object) -> bool:
    """Return whether a stored event has the expected local schema."""
    if not isinstance(event, dict):
        return False
    if not isinstance(event.get("title"), str) or not event["title"].strip():
        return False
    if event.get("category") not in CATEGORIES or not isinstance(event.get("notes"), str):
        return False
    if not isinstance(event.get("duration"), int) or isinstance(event["duration"], bool):
        return False
    if not 15 <= event["duration"] <= 480:
        return False
    if not isinstance(event.get("date"), str) or not isinstance(event.get("time"), str):
        return False
    try:
        date.fromisoformat(event["date"])
        datetime.strptime(event["time"], "%H:%M")
    except ValueError:
        return False
    return True


def load_events() -> list[dict]:
    """Load the local event list, treating missing or invalid files as empty."""
    try:
        events = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return []
    return list(filter(is_valid_event, events)) if isinstance(events, list) else []


def save_events(events: list[dict]) -> None:
    """Persist events in the application directory."""
    DATA_FILE.write_text(json.dumps(events, indent=2), encoding="utf-8")


def month_for_offset(today: date, offset: int) -> date:
    """Return the first day of the month offset from today."""
    year, month = divmod(today.year * 12 + today.month - 1 + offset, 12)
    return date(year, month + 1, 1)


st.set_page_config(
    page_title="Calendar tracker",
    page_icon=":material/calendar_month:",
    layout="wide",
)
st.title("Calendar tracker")

st.session_state.setdefault("events", load_events())
st.session_state.setdefault("cal_offset", 0)
st.session_state.setdefault("confirm_delete", False)

events = st.session_state.events
today = date.today()
add_tab, monthly_tab, all_events_tab = st.tabs(["Add event", "Monthly view", "All events"])

with add_tab:
    with st.form("add_event"):
        st.subheader("New event")
        title = st.text_input("Title", placeholder="Team meeting")
        date_column, time_column = st.columns(2)
        event_date = date_column.date_input("Date", value=today)
        event_time = time_column.time_input(
            "Time", value=datetime.now().replace(minute=0, second=0, microsecond=0).time()
        )
        category = st.selectbox("Category", CATEGORIES)
        duration = st.number_input("Duration (minutes)", 15, 480, 60, step=15)
        notes = st.text_area("Notes", height=80)
        submitted = st.form_submit_button("Add event", type="primary", icon=":material/add:")

    if submitted:
        clean_title = title.strip()
        if not clean_title:
            st.error("Enter an event title.")
        else:
            events.append(
                {
                    "title": clean_title,
                    "date": str(event_date),
                    "time": event_time.strftime("%H:%M"),
                    "category": category,
                    "duration": int(duration),
                    "notes": notes.strip(),
                }
            )
            save_events(events)
            st.success(f"Event '{clean_title}' added for {event_date} at {event_time:%H:%M}.")

with monthly_tab:
    previous_column, current_column, next_column = st.columns([1, 2, 1])
    if previous_column.button("Previous", icon=":material/chevron_left:"):
        st.session_state.cal_offset -= 1
    if next_column.button("Next", icon=":material/chevron_right:"):
        st.session_state.cal_offset += 1
    if current_column.button("Today", icon=":material/today:"):
        st.session_state.cal_offset = 0

    month_date = month_for_offset(today, st.session_state.cal_offset)
    month_str = month_date.strftime("%Y-%m")
    current_column.subheader(month_date.strftime("%B %Y"))

    month_events = [event for event in events if event["date"].startswith(month_str)]
    if not month_events:
        st.info(f"No events in {month_date:%B %Y}.")
    else:
        by_day: dict[str, list[dict]] = {}
        for event in sorted(month_events, key=lambda item: (item["date"], item["time"])):
            by_day.setdefault(event["date"], []).append(event)

        for day_str, day_events in sorted(by_day.items()):
            event_day = datetime.strptime(day_str, "%Y-%m-%d").date()
            label = "Today" if event_day == today else event_day.strftime("%a, %b %d")
            with st.expander(f"{label} - {len(day_events)} event(s)"):
                for event in day_events:
                    icon = CAT_COLORS.get(event["category"], "⚪")
                    st.markdown(
                        f"{icon} **{event['time']}** - {event['title']} "
                        f"*({event['category']}, {event['duration']} min)*"
                    )
                    if event.get("notes"):
                        st.caption(event["notes"])

with all_events_tab:
    if not events:
        st.info("No events yet.")
    else:
        category_filter = st.multiselect("Filter by category", CATEGORIES, default=CATEGORIES)
        filtered = [event for event in events if event["category"] in category_filter]
        filtered = sorted(filtered, key=lambda item: (item["date"], item["time"]))
        rows = [
            {
                "Date": event["date"],
                "Time": event["time"],
                "Title": event["title"],
                "Category": event["category"],
                "Duration": f"{event['duration']} min",
            }
            for event in filtered
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(f"{len(filtered)} event(s) shown")

        st.subheader("Upcoming (next 7 days)")
        cutoff = (today + timedelta(days=7)).isoformat()
        upcoming = [event for event in filtered if today.isoformat() <= event["date"] <= cutoff]
        if upcoming:
            for event in upcoming:
                icon = CAT_COLORS.get(event["category"], "⚪")
                st.markdown(f"{icon} **{event['date']} {event['time']}** - {event['title']}")
        else:
            st.info("No upcoming events in the next 7 days.")

        if not st.session_state.confirm_delete:
            if st.button("Delete all events", type="secondary", icon=":material/delete:"):
                st.session_state.confirm_delete = True
                st.rerun()
        else:
            st.warning("This permanently deletes every local event.")
            cancel_column, confirm_column = st.columns(2)
            if cancel_column.button("Cancel"):
                st.session_state.confirm_delete = False
                st.rerun()
            if confirm_column.button("Confirm deletion", type="primary"):
                events.clear()
                save_events(events)
                st.session_state.confirm_delete = False
                st.rerun()

"""Contact Book — Streamlit app.

Store, search, edit, and export contacts.
Data stored locally as JSON.

Usage:
    streamlit run main.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Contact Book", layout="wide")
st.title("📒 Contact Book")

DATA_FILE = Path("contacts.json")


def load_contacts() -> list[dict]:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text())
        except Exception:
            pass
    return []


def save_contacts(contacts: list[dict]) -> None:
    DATA_FILE.write_text(json.dumps(contacts, indent=2))


if "contacts" not in st.session_state:
    st.session_state.contacts = load_contacts()

contacts = st.session_state.contacts

# ---------------------------------------------------------------------------
# Sidebar — add / edit contact
# ---------------------------------------------------------------------------
st.sidebar.header("Add Contact")
with st.sidebar.form("add_contact"):
    c_name    = st.text_input("Full Name *")
    c_phone   = st.text_input("Phone")
    c_email   = st.text_input("Email")
    c_company = st.text_input("Company")
    c_address = st.text_area("Address", height=68)
    c_notes   = st.text_area("Notes", height=68)
    c_group   = st.selectbox("Group", ["Family", "Friends", "Work", "Other"])
    add_btn   = st.form_submit_button("Add Contact")

if add_btn and c_name.strip():
    new_contact = {
        "name":    c_name.strip(),
        "phone":   c_phone.strip(),
        "email":   c_email.strip(),
        "company": c_company.strip(),
        "address": c_address.strip(),
        "notes":   c_notes.strip(),
        "group":   c_group,
    }
    # Update if name already exists
    existing = next((i for i, c in enumerate(contacts) if c["name"].lower() == c_name.strip().lower()), None)
    if existing is not None:
        contacts[existing] = new_contact
        st.sidebar.success(f"Updated: {c_name.strip()}")
    else:
        contacts.append(new_contact)
        st.sidebar.success(f"Added: {c_name.strip()}")
    st.session_state.contacts = contacts
    save_contacts(contacts)
    st.rerun()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["All Contacts", "View / Edit", "Export"])

with tab1:
    search = st.text_input("🔍 Search by name, phone, email, or company")
    groups = ["All"] + sorted({c.get("group", "Other") for c in contacts})
    grp_filter = st.selectbox("Group", groups)

    filtered = [
        c for c in contacts
        if (grp_filter == "All" or c.get("group") == grp_filter)
        and (
            not search
            or any(
                search.lower() in str(c.get(f, "")).lower()
                for f in ("name", "phone", "email", "company")
            )
        )
    ]

    if not filtered:
        st.info("No contacts found.")
    else:
        st.caption(f"{len(filtered)} contact(s)")
        for c in sorted(filtered, key=lambda x: x["name"].lower()):
            with st.container(border=True):
                col1, col2, col3 = st.columns([3, 3, 1])
                with col1:
                    st.write(f"**{c['name']}**  `{c.get('group','')}`")
                    if c.get("phone"):
                        st.caption(f"📞 {c['phone']}")
                    if c.get("email"):
                        st.caption(f"✉️ {c['email']}")
                with col2:
                    if c.get("company"):
                        st.caption(f"🏢 {c['company']}")
                    if c.get("address"):
                        st.caption(f"📍 {c['address']}")
                with col3:
                    if st.button("🗑️", key=f"del_{c['name']}"):
                        st.session_state.contacts = [x for x in contacts if x["name"] != c["name"]]
                        save_contacts(st.session_state.contacts)
                        st.rerun()

with tab2:
    if not contacts:
        st.info("No contacts yet.")
    else:
        names = [c["name"] for c in contacts]
        sel   = st.selectbox("Select contact", sorted(names))
        rec   = next((c for c in contacts if c["name"] == sel), None)
        if rec:
            st.subheader(rec["name"])
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Phone:** {rec.get('phone') or '—'}")
                st.write(f"**Email:** {rec.get('email') or '—'}")
                st.write(f"**Company:** {rec.get('company') or '—'}")
            with col2:
                st.write(f"**Group:** {rec.get('group','Other')}")
                st.write(f"**Address:** {rec.get('address') or '—'}")
            if rec.get("notes"):
                st.info(f"Notes: {rec['notes']}")

with tab3:
    st.subheader("Export Contacts")
    if not contacts:
        st.info("No contacts to export.")
    else:
        df = pd.DataFrame(contacts)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode()
        st.download_button("📥 Download CSV", csv, "contacts.csv", "text/csv")
        json_str = json.dumps(contacts, indent=2).encode()
        st.download_button("📥 Download JSON", json_str, "contacts.json", "application/json")

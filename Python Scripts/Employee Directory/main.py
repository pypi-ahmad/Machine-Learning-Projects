"""Employee Directory — Streamlit app.

Maintain a searchable employee directory with department,
role, contact, and status information.  Data stored as CSV.

Usage:
    streamlit run main.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Employee Directory", layout="wide")
st.title("🏢 Employee Directory")

DATA_FILE = Path("employees.csv")

DEPARTMENTS = ["Engineering", "Marketing", "Sales", "HR", "Finance",
               "Operations", "Design", "Legal", "Support", "Executive"]
STATUSES    = ["Active", "On Leave", "Remote", "Contractor", "Intern"]


def load_employees() -> pd.DataFrame:
    if DATA_FILE.exists():
        try:
            return pd.read_csv(DATA_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=[
        "ID", "Name", "Email", "Phone", "Department",
        "Role", "Manager", "Location", "Status", "Start Date"
    ])


def save_employees(df: pd.DataFrame) -> None:
    df.to_csv(DATA_FILE, index=False)


if "emp" not in st.session_state:
    st.session_state.emp = load_employees()

emp = st.session_state.emp

# ---------------------------------------------------------------------------
# Sidebar — add employee
# ---------------------------------------------------------------------------
st.sidebar.header("Add / Update Employee")
with st.sidebar.form("add_emp"):
    e_id     = st.text_input("Employee ID *", placeholder="EMP001")
    e_name   = st.text_input("Full Name *")
    e_email  = st.text_input("Email")
    e_phone  = st.text_input("Phone")
    e_dept   = st.selectbox("Department", DEPARTMENTS)
    e_role   = st.text_input("Role / Job Title")
    e_mgr    = st.text_input("Manager")
    e_loc    = st.text_input("Location", "HQ")
    e_status = st.selectbox("Status", STATUSES)
    e_start  = st.date_input("Start Date")
    add_btn  = st.form_submit_button("Add / Update")

if add_btn and e_id.strip() and e_name.strip():
    new_row = {
        "ID": e_id.strip(), "Name": e_name.strip(), "Email": e_email.strip(),
        "Phone": e_phone.strip(), "Department": e_dept, "Role": e_role.strip(),
        "Manager": e_mgr.strip(), "Location": e_loc.strip(),
        "Status": e_status, "Start Date": str(e_start),
    }
    existing = emp[emp["ID"] == e_id.strip()]
    if not existing.empty:
        emp.loc[emp["ID"] == e_id.strip(), list(new_row.keys())] = list(new_row.values())
        st.sidebar.success(f"Updated: {e_name.strip()}")
    else:
        emp = pd.concat([emp, pd.DataFrame([new_row])], ignore_index=True)
        st.sidebar.success(f"Added: {e_name.strip()}")
    st.session_state.emp = emp
    save_employees(emp)
    st.rerun()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Directory", "Department View", "Analytics"])

with tab1:
    if emp.empty:
        st.info("No employees yet. Add one via the sidebar.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            search = st.text_input("🔍 Search by name, role, or email")
        with col2:
            dept_filter = st.multiselect("Department",
                                          emp["Department"].unique().tolist(),
                                          default=emp["Department"].unique().tolist())
        with col3:
            status_filter = st.multiselect("Status",
                                            emp["Status"].unique().tolist(),
                                            default=emp["Status"].unique().tolist())

        mask = (
            emp["Department"].isin(dept_filter) &
            emp["Status"].isin(status_filter)
        )
        if search:
            mask &= (
                emp["Name"].str.contains(search, case=False, na=False) |
                emp["Role"].str.contains(search, case=False, na=False) |
                emp["Email"].str.contains(search, case=False, na=False)
            )

        view = emp[mask].sort_values("Name")
        st.caption(f"{len(view)} employee(s)")
        st.dataframe(view, use_container_width=True, hide_index=True)

        st.divider()
        del_id = st.text_input("Enter Employee ID to delete")
        if st.button("🗑️ Delete Employee") and del_id.strip():
            if del_id.strip() in emp["ID"].values:
                st.session_state.emp = emp[emp["ID"] != del_id.strip()].reset_index(drop=True)
                save_employees(st.session_state.emp)
                st.success(f"Deleted employee {del_id.strip()}")
                st.rerun()
            else:
                st.error("Employee ID not found.")

with tab2:
    if emp.empty:
        st.info("No employees yet.")
    else:
        dept_sel = st.selectbox("Select Department", sorted(emp["Department"].unique()))
        dept_df  = emp[emp["Department"] == dept_sel].sort_values("Name")
        st.subheader(f"{dept_sel} ({len(dept_df)} employees)")

        for _, row in dept_df.iterrows():
            with st.container(border=True):
                c1, c2, c3 = st.columns([2, 2, 1])
                c1.write(f"**{row['Name']}** — {row['Role']}")
                c1.caption(f"📧 {row.get('Email','—')}  ·  📞 {row.get('Phone','—')}")
                c2.write(f"📍 {row.get('Location','—')}  ·  👤 Manager: {row.get('Manager','—')}")
                c2.caption(f"📅 Since: {row.get('Start Date','—')}")
                c3.write(f"`{row['Status']}`")

with tab3:
    if emp.empty:
        st.info("No employees yet.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Total Employees", len(emp))
        c2.metric("Departments",     emp["Department"].nunique())

        st.subheader("Headcount by Department")
        dept_counts = emp["Department"].value_counts()
        st.bar_chart(dept_counts)

        st.subheader("Status Breakdown")
        status_counts = emp["Status"].value_counts()
        st.bar_chart(status_counts)

        csv = emp.to_csv(index=False).encode()
        st.download_button("📥 Export Directory CSV", csv, "employees.csv", "text/csv")

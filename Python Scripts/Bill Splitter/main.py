"""Bill Splitter — Streamlit app.

Split bills among a group of people.
Add expenses, assign payers and participants, see who owes whom.

Usage:
    streamlit run main.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bill Splitter", layout="wide")
st.title("💸 Bill Splitter")

DATA_FILE = Path("bills.json")


def load_data() -> dict:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text())
        except Exception:
            pass
    return {"people": [], "expenses": []}


def save_data(data: dict):
    DATA_FILE.write_text(json.dumps(data, indent=2))


if "data" not in st.session_state:
    st.session_state.data = load_data()

data     = st.session_state.data
people   = data["people"]
expenses = data["expenses"]

# ── Sidebar — manage people ──────────────────────────────────────────────────
st.sidebar.header("People")
new_person = st.sidebar.text_input("Add person")
if st.sidebar.button("Add") and new_person.strip():
    name = new_person.strip()
    if name not in people:
        people.append(name)
        save_data(data)
        st.rerun()

if people:
    rm = st.sidebar.selectbox("Remove", [""] + people)
    if st.sidebar.button("Remove") and rm:
        people.remove(rm)
        save_data(data)
        st.rerun()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Add Expense", "Expenses", "Settlements"])

with tab1:
    if not people:
        st.info("Add people in the sidebar first.")
    else:
        with st.form("add_expense"):
            desc    = st.text_input("Description", placeholder="Dinner at restaurant")
            amount  = st.number_input("Amount ($)", min_value=0.01, step=0.01, format="%.2f")
            payer   = st.selectbox("Paid by", people)
            splits  = st.multiselect("Split among", people, default=people)
            submit  = st.form_submit_button("Add Expense", type="primary")

        if submit and desc and splits:
            share = round(amount / len(splits), 2)
            expenses.append({
                "description": desc,
                "amount":      round(amount, 2),
                "payer":       payer,
                "splits":      splits,
                "share":       share,
            })
            save_data(data)
            st.success(f"Added: {desc} — ${amount:.2f} split {len(splits)} ways (${share:.2f}/person)")

with tab2:
    if not expenses:
        st.info("No expenses yet.")
    else:
        total = sum(e["amount"] for e in expenses)
        st.metric("Total Expenses", f"${total:,.2f}")
        rows = [{"Description": e["description"], "Amount": f"${e['amount']:.2f}",
                 "Paid By": e["payer"], "Split Among": ", ".join(e["splits"]),
                 "Share Each": f"${e['share']:.2f}"} for e in expenses]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if st.button("Clear All Expenses", type="secondary"):
            data["expenses"] = []
            save_data(data)
            st.rerun()

with tab3:
    if not expenses or not people:
        st.info("Add people and expenses first.")
    else:
        # Calculate net balances
        balance: dict[str, float] = {p: 0.0 for p in people}
        for e in expenses:
            payer = e["payer"]
            share = e["share"]
            for person in e["splits"]:
                if person != payer:
                    balance[payer]  += share    # payer is owed
                    balance[person] -= share    # participant owes

        st.subheader("Net Balances")
        bal_rows = [{"Person": p, "Balance": f"${v:+.2f}",
                     "Status": "Gets back" if v > 0.01 else ("Owes" if v < -0.01 else "Settled")}
                    for p, v in balance.items()]
        st.dataframe(pd.DataFrame(bal_rows), use_container_width=True, hide_index=True)

        # Settle up: greedy algorithm
        st.subheader("Who Pays Whom")
        creditors = sorted([(p, v) for p, v in balance.items() if v >  0.005], key=lambda x: -x[1])
        debtors   = sorted([(p, v) for p, v in balance.items() if v < -0.005], key=lambda x:  x[1])

        settlements = []
        c_list = [[p, v] for p, v in creditors]
        d_list = [[p, v] for p, v in debtors]
        ci, di = 0, 0
        while ci < len(c_list) and di < len(d_list):
            creditor, credit = c_list[ci]
            debtor,   debt   = d_list[di]
            amount = min(credit, -debt)
            settlements.append({"From": debtor, "To": creditor, "Amount": f"${amount:.2f}"})
            c_list[ci][1] -= amount
            d_list[di][1] += amount
            if c_list[ci][1] < 0.005: ci += 1
            if d_list[di][1] > -0.005: di += 1

        if settlements:
            st.dataframe(pd.DataFrame(settlements), use_container_width=True, hide_index=True)
        else:
            st.success("Everyone is settled up!")

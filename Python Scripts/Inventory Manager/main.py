"""Inventory Manager — Streamlit app.

Track products, stock levels, reorder points, and transactions.
Alerts on low stock.  Data stored locally as CSV.

Usage:
    streamlit run main.py
"""

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Inventory Manager", layout="wide")
st.title("📦 Inventory Manager")

INV_FILE = Path("inventory.csv")
TXN_FILE = Path("transactions.csv")


def load_inventory() -> pd.DataFrame:
    if INV_FILE.exists():
        try:
            return pd.read_csv(INV_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=[
        "SKU", "Name", "Category", "Quantity", "Unit", "Price", "Cost",
        "Reorder_Point", "Supplier"
    ])


def load_transactions() -> pd.DataFrame:
    if TXN_FILE.exists():
        try:
            return pd.read_csv(TXN_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=["Date", "SKU", "Name", "Type", "Qty", "Notes"])


def save_inventory(df: pd.DataFrame) -> None:
    df.to_csv(INV_FILE, index=False)


def save_transactions(df: pd.DataFrame) -> None:
    df.to_csv(TXN_FILE, index=False)


if "inv" not in st.session_state:
    st.session_state.inv = load_inventory()
if "txn" not in st.session_state:
    st.session_state.txn = load_transactions()

inv = st.session_state.inv
txn = st.session_state.txn

# ---------------------------------------------------------------------------
# Sidebar — add product
# ---------------------------------------------------------------------------
st.sidebar.header("Add / Edit Product")
with st.sidebar.form("add_product"):
    sku      = st.text_input("SKU (unique ID)")
    name     = st.text_input("Product Name")
    category = st.text_input("Category", "General")
    qty      = st.number_input("Initial Quantity", 0, step=1)
    unit     = st.text_input("Unit", "pcs")
    price    = st.number_input("Selling Price ($)", 0.0, step=0.01, format="%.2f")
    cost     = st.number_input("Cost Price ($)", 0.0, step=0.01, format="%.2f")
    reorder  = st.number_input("Reorder Point", 0, step=1, value=10)
    supplier = st.text_input("Supplier", "")
    add_btn  = st.form_submit_button("Add / Update")

if add_btn and sku.strip() and name.strip():
    existing = inv[inv["SKU"] == sku.strip()]
    new_row = {
        "SKU": sku.strip(), "Name": name.strip(), "Category": category,
        "Quantity": qty, "Unit": unit, "Price": price, "Cost": cost,
        "Reorder_Point": reorder, "Supplier": supplier,
    }
    if not existing.empty:
        inv.loc[inv["SKU"] == sku.strip(), list(new_row.keys())] = list(new_row.values())
    else:
        inv = pd.concat([inv, pd.DataFrame([new_row])], ignore_index=True)
    st.session_state.inv = inv
    save_inventory(inv)
    st.sidebar.success(f"Product '{name}' saved.")

if inv.empty:
    st.info("No products yet. Add one via the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Inventory", "Stock Update", "Transactions", "Analytics"])

with tab1:
    st.subheader("Current Inventory")
    # Low stock alerts
    if "Reorder_Point" in inv.columns:
        low = inv[inv["Quantity"] <= inv["Reorder_Point"]]
        if not low.empty:
            st.warning(f"⚠️ {len(low)} product(s) at or below reorder point!")
            st.dataframe(low[["SKU", "Name", "Quantity", "Reorder_Point"]],
                         use_container_width=True)

    cat_filter = st.multiselect("Filter by category",
                                 inv["Category"].unique().tolist(),
                                 default=inv["Category"].unique().tolist())
    view = inv[inv["Category"].isin(cat_filter)]
    st.dataframe(view, use_container_width=True)

    total_value = (inv["Quantity"] * inv.get("Price", 0)).sum() if "Price" in inv else 0
    total_cost  = (inv["Quantity"] * inv.get("Cost",  0)).sum() if "Cost"  in inv else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Products", len(inv))
    c2.metric("Inventory Value", f"${total_value:,.2f}")
    c3.metric("Inventory Cost",  f"${total_cost:,.2f}")

with tab2:
    st.subheader("Update Stock")
    sku_list = inv["SKU"].tolist()
    sel_sku  = st.selectbox("Select product", sku_list)
    sel_row  = inv[inv["SKU"] == sel_sku].iloc[0]
    st.write(f"**{sel_row['Name']}** — Current stock: {sel_row['Quantity']} {sel_row.get('Unit','')}")

    with st.form("update_stock"):
        tx_type = st.radio("Transaction type", ["Restock (in)", "Sale (out)", "Adjustment"], horizontal=True)
        tx_qty  = st.number_input("Quantity", 1, step=1)
        tx_note = st.text_input("Notes")
        update  = st.form_submit_button("Update")

    if update:
        idx = inv[inv["SKU"] == sel_sku].index[0]
        old_qty = inv.at[idx, "Quantity"]
        if "in" in tx_type.lower():
            inv.at[idx, "Quantity"] = old_qty + tx_qty
            direction = "+"
        elif "out" in tx_type.lower():
            if tx_qty > old_qty:
                st.error("Insufficient stock.")
            else:
                inv.at[idx, "Quantity"] = old_qty - tx_qty
                direction = "-"
        else:
            inv.at[idx, "Quantity"] = tx_qty
            direction = "="

        new_txn = pd.DataFrame([{
            "Date": str(date.today()), "SKU": sel_sku,
            "Name": sel_row["Name"], "Type": tx_type,
            "Qty": tx_qty, "Notes": tx_note,
        }])
        st.session_state.txn = pd.concat([txn, new_txn], ignore_index=True)
        save_transactions(st.session_state.txn)
        st.session_state.inv = inv
        save_inventory(inv)
        st.success(f"Stock updated: {old_qty} → {inv.at[idx, 'Quantity']}")
        st.rerun()

with tab3:
    st.subheader("Transaction History")
    if st.session_state.txn.empty:
        st.info("No transactions yet.")
    else:
        st.dataframe(st.session_state.txn.sort_values("Date", ascending=False),
                     use_container_width=True)

with tab4:
    st.subheader("Analytics")
    if "Category" in inv.columns:
        cat_totals = inv.groupby("Category")["Quantity"].sum().sort_values(ascending=False)
        st.write("**Stock by Category**")
        st.bar_chart(cat_totals)

    # Top products by value
    if "Price" in inv.columns:
        inv_copy = inv.copy()
        inv_copy["Value"] = inv_copy["Quantity"] * inv_copy["Price"]
        top = inv_copy.nlargest(10, "Value")[["Name", "Quantity", "Price", "Value"]]
        st.write("**Top 10 by Value**")
        st.dataframe(top, use_container_width=True)

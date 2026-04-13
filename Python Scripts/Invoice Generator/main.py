"""Invoice Generator — Streamlit app.

Create professional invoices, save them as JSON, and export to CSV.
Optional HTML invoice download for printing.

Usage:
    streamlit run main.py
"""

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Invoice Generator", layout="wide")
st.title("🧾 Invoice Generator")

DATA_FILE   = Path("invoices.json")
INV_COUNTER = Path("invoice_counter.txt")


def next_invoice_number() -> str:
    n = int(INV_COUNTER.read_text()) if INV_COUNTER.exists() else 1000
    INV_COUNTER.write_text(str(n + 1))
    return f"INV-{n + 1:04d}"


def load_invoices() -> list[dict]:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text())
        except Exception:
            pass
    return []


def save_invoices(inv: list[dict]):
    DATA_FILE.write_text(json.dumps(inv, indent=2))


def render_html(inv: dict) -> str:
    rows = "".join(
        f"<tr><td>{i['description']}</td><td style='text-align:right'>{i['qty']}</td>"
        f"<td style='text-align:right'>${i['unit_price']:.2f}</td>"
        f"<td style='text-align:right'>${i['total']:.2f}</td></tr>"
        for i in inv["items"]
    )
    return f"""<!DOCTYPE html><html><head><style>
body{{font-family:Arial;padding:40px}} h1{{color:#333}} table{{width:100%;border-collapse:collapse}}
th,td{{border:1px solid #ddd;padding:8px}} th{{background:#f4f4f4}}
.totals{{text-align:right;margin-top:10px}} .total-row{{font-weight:bold;font-size:1.1em}}
</style></head><body>
<h1>INVOICE</h1>
<p><b>Invoice #:</b> {inv['number']}<br>
<b>Date:</b> {inv['date']}<br><b>Due:</b> {inv['due_date']}</p>
<h3>Bill To</h3><p>{inv['client_name']}<br>{inv['client_email']}<br>{inv['client_address']}</p>
<h3>From</h3><p>{inv['company_name']}<br>{inv['company_email']}</p>
<table><thead><tr><th>Description</th><th>Qty</th><th>Unit Price</th><th>Total</th></tr></thead>
<tbody>{rows}</tbody></table>
<div class='totals'>
<p>Subtotal: ${inv['subtotal']:.2f}</p>
<p>Tax ({inv['tax_rate']}%): ${inv['tax_amount']:.2f}</p>
<p class='total-row'>TOTAL: ${inv['total']:.2f}</p>
</div>
<p><b>Notes:</b> {inv.get('notes','')}</p>
</body></html>"""


if "invoices" not in st.session_state:
    st.session_state.invoices = load_invoices()
invoices = st.session_state.invoices

tab1, tab2 = st.tabs(["Create Invoice", "Invoice History"])

with tab1:
    with st.form("invoice_form"):
        st.subheader("Invoice Details")
        c1, c2 = st.columns(2)
        inv_date = c1.date_input("Invoice Date", value=date.today())
        due_date = c2.date_input("Due Date",     value=date.today() + timedelta(days=30))

        st.subheader("Your Business")
        c1, c2 = st.columns(2)
        company_name  = c1.text_input("Company Name", "My Company")
        company_email = c2.text_input("Company Email", "billing@mycompany.com")

        st.subheader("Bill To")
        c1, c2 = st.columns(2)
        client_name  = c1.text_input("Client Name")
        client_email = c2.text_input("Client Email")
        client_addr  = st.text_area("Client Address", height=60)

        st.subheader("Line Items")
        n_items = st.number_input("Number of items", 1, 20, 3)
        items = []
        for i in range(int(n_items)):
            c1, c2, c3 = st.columns([4, 1, 2])
            desc  = c1.text_input(f"Description {i+1}", key=f"desc_{i}")
            qty   = c2.number_input("Qty", 1, 1000, 1, key=f"qty_{i}")
            price = c3.number_input("Unit Price ($)", 0.0, step=0.01, format="%.2f", key=f"price_{i}")
            items.append({"description": desc, "qty": qty,
                           "unit_price": price, "total": qty * price})

        tax_rate = st.number_input("Tax rate (%)", 0.0, 50.0, 10.0, step=0.5)
        notes    = st.text_area("Notes / Payment terms", "Payment due within 30 days.")
        submit   = st.form_submit_button("Generate Invoice", type="primary")

    if submit and client_name:
        subtotal   = sum(i["total"] for i in items)
        tax_amount = subtotal * tax_rate / 100
        inv = {
            "number":       next_invoice_number(),
            "date":         str(inv_date),
            "due_date":     str(due_date),
            "company_name": company_name,
            "company_email": company_email,
            "client_name":  client_name,
            "client_email": client_email,
            "client_address": client_addr,
            "items":        items,
            "subtotal":     round(subtotal, 2),
            "tax_rate":     tax_rate,
            "tax_amount":   round(tax_amount, 2),
            "total":        round(subtotal + tax_amount, 2),
            "notes":        notes,
        }
        invoices.append(inv)
        save_invoices(invoices)

        st.success(f"Invoice {inv['number']} created! Total: ${inv['total']:.2f}")
        html = render_html(inv)
        st.download_button("📥 Download HTML Invoice", html.encode(),
                            f"{inv['number']}.html", "text/html")

with tab2:
    if not invoices:
        st.info("No invoices yet.")
    else:
        rows = [{k: v for k, v in inv.items() if k not in ("items","notes","client_address")}
                for inv in invoices]
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.metric("Total Invoiced", f"${df['total'].sum():,.2f}")
        csv = df.to_csv(index=False).encode()
        st.download_button("📥 Export CSV", csv, "invoices.csv", "text/csv")

        sel = st.selectbox("View invoice", [i["number"] for i in invoices])
        inv = next((i for i in invoices if i["number"] == sel), None)
        if inv:
            st.json(inv)
            html = render_html(inv)
            st.download_button("📥 Download HTML", html.encode(), f"{sel}.html", "text/html")

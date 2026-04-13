"""Sales Dashboard — Streamlit app.

Upload a sales CSV and explore revenue, units, top products,
regional breakdown, and monthly trends.

Expected CSV columns (flexible — app auto-detects):
  date, product, region, sales_amount, units_sold, category

Usage:
    streamlit run main.py
"""

from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sales Dashboard", layout="wide")
st.title("📊 Sales Dashboard")

SAMPLE_CSV = Path("sample_sales.csv")

def make_sample():
    import random, string
    from datetime import date, timedelta
    random.seed(42)
    products  = ["Widget A","Widget B","Gadget X","Gadget Y","Pro Kit","Starter Pack"]
    regions   = ["North","South","East","West"]
    cats      = {"Widget A":"Widgets","Widget B":"Widgets","Gadget X":"Gadgets",
                 "Gadget Y":"Gadgets","Pro Kit":"Kits","Starter Pack":"Kits"}
    rows = []
    d = date(2024, 1, 1)
    for _ in range(300):
        p = random.choice(products)
        u = random.randint(1, 50)
        rows.append({
            "date":         str(d + timedelta(days=random.randint(0, 364))),
            "product":      p,
            "category":     cats[p],
            "region":       random.choice(regions),
            "units_sold":   u,
            "sales_amount": round(u * random.uniform(10, 200), 2),
        })
    return pd.DataFrame(rows)

# ── Data loading ────────────────────────────────────────────────────────────
uploaded = st.sidebar.file_uploader("Upload Sales CSV", type=["csv","tsv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.sidebar.info("No file uploaded — using sample data.")
    if not SAMPLE_CSV.exists():
        make_sample().to_csv(SAMPLE_CSV, index=False)
    df = pd.read_csv(SAMPLE_CSV)

# Auto-detect columns
date_col  = next((c for c in df.columns if "date" in c.lower()), None)
sales_col = next((c for c in df.columns if "sales" in c.lower() or "amount" in c.lower() or "revenue" in c.lower()), None)
units_col = next((c for c in df.columns if "unit" in c.lower() or "qty" in c.lower()), None)
prod_col  = next((c for c in df.columns if "product" in c.lower() or "item" in c.lower()), None)
region_col= next((c for c in df.columns if "region" in c.lower() or "area" in c.lower()), None)
cat_col   = next((c for c in df.columns if "cat" in c.lower()), None)

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# ── Filters ──────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")
if region_col and df[region_col].nunique() > 1:
    regions = st.sidebar.multiselect("Region", df[region_col].unique(), default=df[region_col].unique())
    df = df[df[region_col].isin(regions)]
if cat_col and df[cat_col].nunique() > 1:
    cats = st.sidebar.multiselect("Category", df[cat_col].unique(), default=df[cat_col].unique())
    df = df[df[cat_col].isin(cats)]

# ── Metrics ──────────────────────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
if sales_col:
    c1.metric("Total Revenue", f"${df[sales_col].sum():,.2f}")
if units_col:
    c2.metric("Total Units",   f"{df[units_col].sum():,}")
if sales_col and units_col:
    avg = df[sales_col].sum() / df[units_col].sum() if df[units_col].sum() else 0
    c3.metric("Avg Price/Unit", f"${avg:.2f}")
c4.metric("Transactions", f"{len(df):,}")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Products", "Regional", "Data"])

with tab1:
    if date_col and sales_col:
        st.subheader("Monthly Revenue")
        monthly = df.groupby(df[date_col].dt.to_period("M").astype(str))[sales_col].sum()
        st.line_chart(monthly)
    if date_col and units_col:
        st.subheader("Monthly Units Sold")
        mu = df.groupby(df[date_col].dt.to_period("M").astype(str))[units_col].sum()
        st.bar_chart(mu)

with tab2:
    if prod_col and sales_col:
        st.subheader("Revenue by Product")
        bp = df.groupby(prod_col)[sales_col].sum().sort_values(ascending=False)
        st.bar_chart(bp)
    if prod_col and units_col:
        st.subheader("Units by Product")
        bu = df.groupby(prod_col)[units_col].sum().sort_values(ascending=False)
        st.dataframe(bu.reset_index(), use_container_width=True)

with tab3:
    if region_col and sales_col:
        st.subheader("Revenue by Region")
        br = df.groupby(region_col)[sales_col].sum().sort_values(ascending=False)
        st.bar_chart(br)
    if cat_col and sales_col:
        st.subheader("Revenue by Category")
        bc = df.groupby(cat_col)[sales_col].sum()
        st.bar_chart(bc)

with tab4:
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode()
    st.download_button("📥 Download filtered CSV", csv, "sales_filtered.csv", "text/csv")

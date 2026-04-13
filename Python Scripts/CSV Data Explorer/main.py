"""CSV Data Explorer — Streamlit dashboard.

Upload or enter a CSV file path and explore it interactively:
preview, filter, sort, summary statistics, and basic charts.

Usage:
    streamlit run main.py
"""

import io

import pandas as pd
import streamlit as st

st.set_page_config(page_title="CSV Data Explorer", layout="wide")
st.title("📊 CSV Data Explorer")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
st.sidebar.header("Load Data")
upload = st.sidebar.file_uploader("Upload CSV", type=["csv", "tsv"])
sep_option = st.sidebar.selectbox("Delimiter", [",", ";", "\t", "|"])

df: pd.DataFrame | None = None

if upload:
    try:
        df = pd.read_csv(upload, sep=sep_option)
        st.sidebar.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

if df is None:
    st.info("Upload a CSV file using the sidebar to get started.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filter")

# Column selector
cols = st.sidebar.multiselect("Show columns", df.columns.tolist(), default=df.columns.tolist())
df_view = df[cols] if cols else df

# Row limit
n_rows = st.sidebar.slider("Rows to preview", 5, min(500, len(df)), min(50, len(df)))

# Keyword filter
kw = st.sidebar.text_input("Keyword filter (any column)")
if kw:
    mask = df_view.apply(lambda c: c.astype(str).str.contains(kw, case=False, na=False)).any(axis=1)
    df_view = df_view[mask]

# Sort
sort_col = st.sidebar.selectbox("Sort by", ["(none)"] + df_view.columns.tolist())
sort_asc  = st.sidebar.checkbox("Ascending", True)
if sort_col != "(none)":
    df_view = df_view.sort_values(sort_col, ascending=sort_asc)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Preview", "Statistics", "Charts", "Missing Values"])

with tab1:
    st.subheader(f"Data Preview  ({len(df_view):,} rows shown, {n_rows} displayed)")
    st.dataframe(df_view.head(n_rows), use_container_width=True)

with tab2:
    st.subheader("Summary Statistics")
    st.dataframe(df_view.describe(include="all").T, use_container_width=True)
    st.write(f"**Shape:** {df_view.shape[0]} rows × {df_view.shape[1]} columns")
    st.write(f"**Dtypes:**")
    st.write(df_view.dtypes.rename("dtype").to_frame())

with tab3:
    st.subheader("Quick Charts")
    numeric_cols = df_view.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns available for charting.")
    else:
        chart_type = st.selectbox("Chart type", ["Bar", "Line", "Area", "Histogram"])
        y_col = st.selectbox("Y-axis column", numeric_cols)
        x_col = st.selectbox("X-axis column", ["(index)"] + df_view.columns.tolist())
        plot_df = df_view[[y_col]].head(200) if x_col == "(index)" else df_view[[x_col, y_col]].head(200)
        if x_col != "(index)":
            plot_df = plot_df.set_index(x_col)
        if chart_type == "Bar":
            st.bar_chart(plot_df)
        elif chart_type == "Line":
            st.line_chart(plot_df)
        elif chart_type == "Area":
            st.area_chart(plot_df)
        else:
            import altair as alt
            chart = alt.Chart(df_view.head(500)).mark_bar().encode(
                x=alt.X(y_col, bin=True), y="count()"
            )
            st.altair_chart(chart, use_container_width=True)

with tab4:
    st.subheader("Missing Values")
    missing = df_view.isnull().sum().rename("Missing").to_frame()
    missing["Percent"] = (missing["Missing"] / len(df_view) * 100).round(2)
    st.dataframe(missing[missing["Missing"] > 0], use_container_width=True)
    total_missing = missing["Missing"].sum()
    st.metric("Total missing cells", f"{total_missing:,}")

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
st.sidebar.header("Export")
if st.sidebar.button("Download filtered CSV"):
    csv_bytes = df_view.to_csv(index=False).encode()
    st.sidebar.download_button("💾 Download", csv_bytes, "filtered.csv", "text/csv")

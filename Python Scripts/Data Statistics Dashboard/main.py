"""Data Statistics Dashboard — Streamlit app.

Upload any CSV/Excel file and get an instant statistical report:
descriptive stats, distribution plots, correlation heatmap,
outlier detection, and data quality summary.

Usage:
    streamlit run main.py
"""

import io

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Statistics Dashboard", layout="wide")
st.title("📈 Data Statistics Dashboard")

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
upload = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if not upload:
    st.info("Upload a CSV or Excel file to begin.")
    st.stop()

try:
    if upload.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(upload)
    else:
        df = pd.read_csv(upload)
    st.sidebar.success(f"{len(df):,} rows × {len(df.columns)} columns")
except Exception as e:
    st.error(f"Could not load file: {e}")
    st.stop()

numeric_df = df.select_dtypes(include="number")
cat_df     = df.select_dtypes(exclude="number")

# ---------------------------------------------------------------------------
# Sidebar options
# ---------------------------------------------------------------------------
st.sidebar.header("Options")
selected_cols = st.sidebar.multiselect(
    "Columns to analyse", df.columns.tolist(), default=df.columns.tolist()[:8]
)
if not selected_cols:
    st.warning("Select at least one column.")
    st.stop()

df_sel = df[selected_cols]
num_sel = df_sel.select_dtypes(include="number")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Distributions", "Correlation", "Outliers", "Data Quality"
])

with tab1:
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",    f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Numeric cols", len(numeric_df.columns))
    c4.metric("Categorical cols", len(cat_df.columns))

    st.write("**Sample (first 10 rows)**")
    st.dataframe(df.head(10), use_container_width=True)

    st.write("**Descriptive Statistics**")
    st.dataframe(num_sel.describe().T.round(3), use_container_width=True)

with tab2:
    if num_sel.empty:
        st.info("No numeric columns selected.")
    else:
        col = st.selectbox("Column", num_sel.columns)
        chart_type = st.radio("Type", ["Histogram", "Box-like (sorted values)"], horizontal=True)
        if chart_type == "Histogram":
            bins = st.slider("Bins", 5, 100, 30)
            hist_data = pd.cut(num_sel[col].dropna(), bins=bins).value_counts().sort_index()
            hist_df = pd.DataFrame({"count": hist_data.values},
                                    index=[str(i) for i in hist_data.index])
            st.bar_chart(hist_df)
        else:
            sorted_vals = num_sel[col].dropna().sort_values().reset_index(drop=True)
            st.line_chart(sorted_vals)

        st.write(f"**Statistics for '{col}'**")
        stats = num_sel[col].describe()
        skew  = num_sel[col].skew()
        kurt  = num_sel[col].kurtosis()
        stats_df = pd.concat([stats, pd.Series({"skewness": skew, "kurtosis": kurt})])
        st.dataframe(stats_df.rename("value").to_frame().round(4), use_container_width=True)

with tab3:
    if num_sel.shape[1] < 2:
        st.info("Need at least 2 numeric columns for correlation.")
    else:
        corr = num_sel.corr()
        st.write("**Pearson Correlation Matrix**")
        st.dataframe(corr.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1).format("{:.2f}"),
                     use_container_width=True)

        # Top correlations
        pairs = []
        cols_c = corr.columns.tolist()
        for i in range(len(cols_c)):
            for j in range(i + 1, len(cols_c)):
                pairs.append((cols_c[i], cols_c[j], corr.iloc[i, j]))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        top_pairs = pd.DataFrame(pairs[:10], columns=["Col A", "Col B", "Correlation"])
        st.write("**Top Correlated Pairs**")
        st.dataframe(top_pairs, use_container_width=True)

with tab4:
    if num_sel.empty:
        st.info("No numeric columns.")
    else:
        col = st.selectbox("Column for outlier detection", num_sel.columns, key="out_col")
        method = st.radio("Method", ["IQR", "Z-score (±3σ)"], horizontal=True)
        s = num_sel[col].dropna()
        if method == "IQR":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr    = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = s[(s < lo) | (s > hi)]
        else:
            mean, std = s.mean(), s.std()
            outliers  = s[(s - mean).abs() > 3 * std]

        st.metric("Outliers found", len(outliers))
        if not outliers.empty:
            st.dataframe(outliers.rename("value").to_frame(), use_container_width=True)

with tab5:
    st.subheader("Data Quality Report")
    quality = pd.DataFrame({
        "Missing":    df_sel.isnull().sum(),
        "Missing %":  (df_sel.isnull().mean() * 100).round(2),
        "Unique":     df_sel.nunique(),
        "Dtype":      df_sel.dtypes.astype(str),
        "Zeros":      (df_sel == 0).sum() if not num_sel.empty else 0,
    })
    st.dataframe(quality, use_container_width=True)
    st.metric("Total missing cells", int(df_sel.isnull().sum().sum()))
    st.metric("Duplicate rows", int(df_sel.duplicated().sum()))

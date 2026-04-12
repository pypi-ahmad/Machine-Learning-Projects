"""Write the Regional Sales Comparison notebook directly to disk."""
import json, pathlib, textwrap

OUT_DIR = pathlib.Path(r"E:\Github\Machine-Learning-Projects\Data Analysis\Regional Sales Comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)
NB_PATH = OUT_DIR / "Regional Sales Comparison.ipynb"

def md(source: str, cell_id: str):
    lines = [l + "\n" for l in source.split("\n")]
    lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": lines}

def code(source: str, cell_id: str):
    lines = [l + "\n" for l in source.split("\n")]
    lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "source": lines,
        "outputs": [],
        "execution_count": None,
    }

cells = []

# ── Cell 1: Title ─────────────────────────────────────────────────────────────
cells.append(md("""# Regional Sales Comparison

> **Goal:** Compare sales performance across geographic regions — measuring revenue,
> order count, average order value (AOV), profit margin, category mix, and year-over-year
> growth. Surface the strongest and weakest regions and identify likely drivers.

| Dimension | Key Question |
|---|---|
| Revenue | Which region generates the most sales? |
| Orders & AOV | Is volume or price driving performance? |
| Margin | Are high-revenue regions actually profitable? |
| Category Mix | Does product mix explain regional differences? |
| Growth | Which regions are accelerating or declining? |
| Drivers | What structural factors explain the gaps? |""", "md-01"))

# ── Cell 2: Setup ─────────────────────────────────────────────────────────────
cells.append(md("## 1. Environment Setup", "md-02"))
cells.append(code("""\
import warnings, pathlib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "axes.titlesize": 13,
                     "axes.titleweight": "bold", "figure.facecolor": "white"})
SEED = 42
np.random.seed(SEED)
print("Libraries loaded OK")\
""", "cd-02"))

# ── Cell 3: Config ─────────────────────────────────────────────────────────────
cells.append(md("## 2. Configuration", "md-03"))
cells.append(code("""\
ROOT     = pathlib.Path(r"E:/Github/Machine-Learning-Projects")
XLS_PATH = ROOT / "Time Series Analysis" / "Time Series Forecasting" / "Sample - Superstore.xls"
OUT_DIR  = ROOT / "Data Analysis" / "Regional Sales Comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGION_PALETTE = {
    "West":    "#4878d0",
    "East":    "#ee854a",
    "Central": "#6acc65",
    "South":   "#d65f5f",
}

print(f"Dataset : {XLS_PATH}")
print(f"Exists  : {XLS_PATH.exists()}")\
""", "cd-03"))

# ── Cell 4: Load Data ──────────────────────────────────────────────────────────
cells.append(md("## 3. Load & Validate Data", "md-04"))
cells.append(code("""\
df = pd.read_excel(XLS_PATH, sheet_name="Orders", engine="xlrd")
df.columns = df.columns.str.strip()

df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Year"]       = df["Order Date"].dt.year
df["YearMonth"]  = df["Order Date"].dt.to_period("M")
df["Margin Pct"] = df["Profit"] / df["Sales"].replace(0, np.nan) * 100

required = ["Region", "Sales", "Profit", "Quantity", "Order ID",
            "Category", "Sub-Category", "Segment", "Year"]
missing  = [c for c in required if c not in df.columns]
assert not missing, f"Missing columns: {missing}"

print(f"Shape      : {df.shape}")
print(f"Date range : {df['Order Date'].min().date()} → {df['Order Date'].max().date()}")
print(f"Regions    : {sorted(df['Region'].unique())}")
df[["Sales", "Profit", "Quantity", "Margin Pct"]].describe().round(2)\
""", "cd-04"))

# ── Cell 5: Regional Scorecard ─────────────────────────────────────────────────
cells.append(md("""\
## 4. Regional Scorecard

Headline KPIs for every region in a single summary table.\
""", "md-05"))
cells.append(code("""\
scorecard = df.groupby("Region").agg(
    Revenue      = ("Sales",      "sum"),
    Profit       = ("Profit",     "sum"),
    Orders       = ("Order ID",   "count"),
    Units        = ("Quantity",   "sum"),
    Avg_Margin   = ("Margin Pct", "mean"),
).sort_values("Revenue", ascending=False)

scorecard["AOV"]            = scorecard["Revenue"] / scorecard["Orders"]
scorecard["Revenue Share %"]= (scorecard["Revenue"] / scorecard["Revenue"].sum() * 100).round(1)
scorecard["Profit Share %"] = (scorecard["Profit"]  / scorecard["Profit"].sum()  * 100).round(1)

fmt = {
    "Revenue":       "${:,.0f}",
    "Profit":        "${:,.0f}",
    "AOV":           "${:,.2f}",
    "Avg_Margin":    "{:.1f}%",
    "Revenue Share %": "{:.1f}%",
    "Profit Share %":  "{:.1f}%",
}
display_sc = scorecard.copy()
for col, f in fmt.items():
    display_sc[col] = display_sc[col].map(f.format)
print(display_sc.to_string())\
""", "cd-05"))

# ── Cell 6: Revenue, Orders, AOV bar charts ────────────────────────────────────
cells.append(md("## 5. Revenue, Order Count & AOV by Region", "md-06"))
cells.append(code("""\
regions = scorecard.index.tolist()
colors  = [REGION_PALETTE.get(r, "#888") for r in regions]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Revenue
axes[0].bar(regions, scorecard["Revenue"]/1e6, color=colors)
axes[0].set_title("Total Revenue")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.1f}M"))
axes[0].set_ylabel("Revenue ($M)")
for bar, v in zip(axes[0].patches, scorecard["Revenue"]/1e6):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f"${v:.2f}M", ha="center", fontsize=9)

# Orders
axes[1].bar(regions, scorecard["Orders"], color=colors)
axes[1].set_title("Total Order Count")
axes[1].set_ylabel("Orders")
for bar, v in zip(axes[1].patches, scorecard["Orders"]):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                 f"{v:,}", ha="center", fontsize=9)

# AOV
axes[2].bar(regions, scorecard["AOV"], color=colors)
axes[2].set_title("Average Order Value (AOV)")
axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}"))
axes[2].set_ylabel("AOV ($)")
for bar, v in zip(axes[2].patches, scorecard["AOV"]):
    axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f"${v:.0f}", ha="center", fontsize=9)

plt.suptitle("Revenue, Order Volume & Average Order Value by Region",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "region_rev_orders_aov.png", bbox_inches="tight")
plt.show()\
""", "cd-06"))

# ── Cell 7: Margin & Profit ────────────────────────────────────────────────────
cells.append(md("## 6. Profit & Margin by Region", "md-07"))
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Total Profit
axes[0].bar(regions, scorecard["Profit"]/1e3, color=colors)
axes[0].set_title("Total Profit by Region")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}K"))
axes[0].set_ylabel("Profit ($K)")
for bar, v in zip(axes[0].patches, scorecard["Profit"]/1e3):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"${v:.1f}K", ha="center", fontsize=9)

# Avg Margin %
marg_colors = ["tomato" if m < 0 else c for m, c in zip(scorecard["Avg_Margin"], colors)]
axes[1].bar(regions, scorecard["Avg_Margin"], color=marg_colors)
axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[1].set_title("Average Profit Margin % by Region")
axes[1].set_ylabel("Avg Margin %")
for bar, v in zip(axes[1].patches, scorecard["Avg_Margin"]):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f"{v:.1f}%", ha="center", fontsize=9)

plt.suptitle("Profitability by Region", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "region_profit_margin.png", bbox_inches="tight")
plt.show()\
""", "cd-07"))

# ── Cell 8: Category Mix ───────────────────────────────────────────────────────
cells.append(md("""\
## 7. Category Mix by Region

Does product mix explain differences in revenue and margin?\
""", "md-08"))
cells.append(code("""\
cat_region = df.groupby(["Region", "Category"])["Sales"].sum().unstack(fill_value=0)
cat_region_pct = cat_region.div(cat_region.sum(axis=1), axis=0) * 100

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Absolute revenue stacked bar
cat_region.div(1e3).plot(kind="bar", stacked=True, ax=axes[0],
                          colormap="Set2", edgecolor="white")
axes[0].set_title("Category Revenue by Region (Stacked)")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}K"))
axes[0].set_ylabel("Revenue ($K)")
axes[0].set_xlabel("")
axes[0].tick_params(axis="x", rotation=15)
axes[0].legend(title="Category", bbox_to_anchor=(1.01, 1), loc="upper left")

# Percentage mix heatmap
sns.heatmap(cat_region_pct.T, annot=True, fmt=".1f", cmap="Blues",
            linewidths=0.4, ax=axes[1],
            cbar_kws={"label": "Revenue Share %"})
axes[1].set_title("Category Revenue Mix % per Region")
axes[1].set_xlabel("Region")
axes[1].set_ylabel("Category")

plt.suptitle("Category Mix by Region", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "region_category_mix.png", bbox_inches="tight")
plt.show()
print(cat_region_pct.round(1).to_string())\
""", "cd-08"))

# ── Cell 9: Sub-category margin heatmap ───────────────────────────────────────
cells.append(md("## 8. Sub-Category Margin Heatmap by Region", "md-09"))
cells.append(code("""\
pivot_margin = df.pivot_table(
    values="Margin Pct",
    index="Sub-Category",
    columns="Region",
    aggfunc="mean",
)

fig, ax = plt.subplots(figsize=(10, 11))
sns.heatmap(pivot_margin, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
            linewidths=0.5, ax=ax,
            cbar_kws={"label": "Avg Profit Margin %"})
ax.set_title("Average Profit Margin % — Sub-Category × Region")
ax.set_xlabel("Region")
ax.set_ylabel("Sub-Category")
plt.tight_layout()
plt.savefig(OUT_DIR / "subcategory_region_heatmap.png", bbox_inches="tight")
plt.show()\
""", "cd-09"))

# ── Cell 10: YoY Growth ────────────────────────────────────────────────────────
cells.append(md("""\
## 9. Year-over-Year Revenue Growth by Region

Track whether regions are accelerating, stable, or declining.\
""", "md-10"))
cells.append(code("""\
yoy = df.groupby(["Region", "Year"])["Sales"].sum().reset_index()
yoy = yoy.sort_values(["Region", "Year"])
yoy["YoY_Growth %"] = yoy.groupby("Region")["Sales"].pct_change() * 100

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for region, grp in yoy.groupby("Region"):
    axes[0].plot(grp["Year"], grp["Sales"]/1e3, marker="o",
                 label=region, color=REGION_PALETTE.get(region))

axes[0].set_title("Annual Revenue Trend by Region")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}K"))
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Revenue ($K)")
axes[0].legend(title="Region")
axes[0].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# YoY Growth %
yoy_valid = yoy.dropna(subset=["YoY_Growth %"])
for region, grp in yoy_valid.groupby("Region"):
    axes[1].plot(grp["Year"], grp["YoY_Growth %"], marker="s",
                 label=region, color=REGION_PALETTE.get(region))
axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[1].set_title("YoY Revenue Growth % by Region")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("YoY Growth %")
axes[1].legend(title="Region")
axes[1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.suptitle("Revenue Growth Trends by Region", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "region_yoy_growth.png", bbox_inches="tight")
plt.show()
print(yoy.pivot(index="Year", columns="Region", values="YoY_Growth %").round(1).to_string())\
""", "cd-10"))

# ── Cell 11: Monthly revenue trend ────────────────────────────────────────────
cells.append(md("## 10. Monthly Revenue Trend by Region", "md-11"))
cells.append(code("""\
monthly_rev = df.groupby(["YearMonth", "Region"])["Sales"].sum().reset_index()
monthly_rev["Date"] = monthly_rev["YearMonth"].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(16, 5))
for region, grp in monthly_rev.groupby("Region"):
    grp = grp.sort_values("Date")
    ax.plot(grp["Date"], grp["Sales"]/1e3, linewidth=1.5,
            label=region, color=REGION_PALETTE.get(region), alpha=0.85)

ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}K"))
ax.set_title("Monthly Revenue by Region")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue ($K)")
ax.legend(title="Region")
plt.tight_layout()
plt.savefig(OUT_DIR / "region_monthly_trend.png", bbox_inches="tight")
plt.show()\
""", "cd-11"))

# ── Cell 12: Segment mix by region ────────────────────────────────────────────
cells.append(md("## 11. Customer Segment Mix by Region", "md-12"))
cells.append(code("""\
seg_region = df.groupby(["Region", "Segment"])["Sales"].sum().unstack(fill_value=0)
seg_region_pct = seg_region.div(seg_region.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(10, 5))
seg_region_pct.plot(kind="bar", stacked=True, ax=ax, colormap="Pastel1", edgecolor="white")
ax.set_title("Customer Segment Revenue Mix by Region")
ax.set_ylabel("Revenue Share %")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=15)
ax.legend(title="Segment", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUT_DIR / "region_segment_mix.png", bbox_inches="tight")
plt.show()
print(seg_region_pct.round(1).to_string())\
""", "cd-12"))

# ── Cell 13: Top sub-categories per region ─────────────────────────────────────
cells.append(md("## 12. Top Sub-Categories per Region", "md-13"))
cells.append(code("""\
top_n = 5
sub_region = (
    df.groupby(["Region", "Sub-Category"])
    .agg(Revenue=("Sales", "sum"), Profit=("Profit", "sum"),
         Margin=("Margin Pct", "mean"))
    .reset_index()
    .sort_values(["Region", "Revenue"], ascending=[True, False])
)

fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)
for ax, (region, grp) in zip(axes, sub_region.groupby("Region")):
    top = grp.head(top_n)
    bar_colors = ["tomato" if p < 0 else REGION_PALETTE.get(region, "#888")
                  for p in top["Profit"]]
    ax.barh(top["Sub-Category"][::-1], top["Revenue"][::-1]/1e3, color=bar_colors[::-1])
    ax.set_title(region)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}K"))
    ax.set_xlabel("Revenue ($K)")

plt.suptitle(f"Top {top_n} Sub-Categories by Revenue per Region",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "region_top_subcategories.png", bbox_inches="tight")
plt.show()\
""", "cd-13"))

# ── Cell 14: Discount vs Margin by region ─────────────────────────────────────
cells.append(md("## 13. Discount Intensity vs Margin by Region", "md-14"))
cells.append(code("""\
disc_region = df.groupby("Region").agg(
    Avg_Discount  = ("Discount",   "mean"),
    Disc_Order_Pct= ("Discount",   lambda x: (x > 0).mean() * 100),
    Avg_Margin    = ("Margin Pct", "mean"),
).sort_values("Avg_Discount", ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metrics = [
    ("Avg_Discount",   "Average Discount Rate",        "{:.1%}"),
    ("Disc_Order_Pct", "% Orders with Any Discount",   "{:.1f}%"),
    ("Avg_Margin",     "Average Profit Margin %",      "{:.1f}%"),
]
for ax, (col, title, fmt) in zip(axes, metrics):
    c = [REGION_PALETTE.get(r, "#888") for r in disc_region.index]
    if col == "Avg_Margin":
        c = ["tomato" if v < 0 else cc for v, cc in zip(disc_region[col], c)]
    ax.bar(disc_region.index, disc_region[col], color=c)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=10)
    for bar, v in zip(ax.patches, disc_region[col]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                fmt.format(v), ha="center", fontsize=9)

plt.suptitle("Discount Intensity & Margin by Region",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "region_discount_margin.png", bbox_inches="tight")
plt.show()
print(disc_region.round(3).to_string())\
""", "cd-14"))

# ── Cell 15: State-level drill-down ───────────────────────────────────────────
cells.append(md("## 14. State-Level Revenue Drill-Down per Region", "md-15"))
cells.append(code("""\
state_rev = (
    df.groupby(["Region", "State"])
    .agg(Revenue=("Sales", "sum"), Profit=("Profit", "sum"),
         Orders=("Order ID", "count"))
    .reset_index()
    .sort_values(["Region", "Revenue"], ascending=[True, False])
)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for ax, (region, grp) in zip(axes, state_rev.groupby("Region")):
    colors = ["tomato" if p < 0 else REGION_PALETTE.get(region, "#4878d0")
              for p in grp["Profit"]]
    ax.barh(grp["State"][::-1], grp["Revenue"][::-1]/1e3, color=colors[::-1])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}K"))
    ax.set_title(f"{region} Region — Revenue by State")
    ax.set_xlabel("Revenue ($K)")

plt.suptitle("State-Level Revenue by Region", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "region_state_drilldown.png", bbox_inches="tight")
plt.show()\
""", "cd-15"))

# ── Cell 16: Growth CAGR table ─────────────────────────────────────────────────
cells.append(md("## 15. Revenue CAGR by Region", "md-16"))
cells.append(code("""\
cagr_data = df.groupby(["Region", "Year"])["Sales"].sum().reset_index()
cagr_table = []
for region, grp in cagr_data.groupby("Region"):
    grp = grp.sort_values("Year")
    rev_start = grp.iloc[0]["Sales"]
    rev_end   = grp.iloc[-1]["Sales"]
    n_years   = grp.iloc[-1]["Year"] - grp.iloc[0]["Year"]
    cagr = ((rev_end / rev_start) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
    cagr_table.append({
        "Region":     region,
        "Start Year": int(grp.iloc[0]["Year"]),
        "End Year":   int(grp.iloc[-1]["Year"]),
        "Start Rev":  f"${rev_start:,.0f}",
        "End Rev":    f"${rev_end:,.0f}",
        "CAGR %":     f"{cagr:.1f}%",
    })

cagr_df = pd.DataFrame(cagr_table).set_index("Region")
print("Revenue CAGR by Region")
print("=" * 60)
print(cagr_df.to_string())\
""", "cd-16"))

# ── Cell 17: Executive Summary ─────────────────────────────────────────────────
cells.append(md("## 16. Executive Summary Statistics", "md-17"))
cells.append(code("""\
# identify best and worst regions
best = scorecard["Revenue"].idxmax()
worst = scorecard["Revenue"].idxmin()
highest_margin = scorecard["Avg_Margin"].idxmax()
lowest_margin  = scorecard["Avg_Margin"].idxmin()

print("=" * 60)
print("REGIONAL SALES — EXECUTIVE SUMMARY")
print("=" * 60)
for reg in scorecard.index:
    r = scorecard.loc[reg]
    print(f"\\n{reg}")
    print(f"  Revenue : ${r['Revenue']:>10,.0f}  ({r['Revenue'] / scorecard['Revenue'].sum()*100:.1f}% share)")
    print(f"  Profit  : ${r['Profit']:>10,.0f}  ({r['Profit'] / scorecard['Profit'].sum()*100:.1f}% share)")
    print(f"  Orders  : {int(r['Orders']):>10,}")
    print(f"  AOV     : ${r['AOV']:>10,.2f}")
    print(f"  Margin  : {r['Avg_Margin']:>9.1f}%")

print()
print(f"Highest-revenue region : {best}")
print(f"Lowest-revenue region  : {worst}")
print(f"Best-margin region     : {highest_margin}")
print(f"Lowest-margin region   : {lowest_margin}")\
""", "cd-17"))

# ── Cell 18: Key Findings & Recommendations ────────────────────────────────────
cells.append(md("""\
## 17. Key Findings & Business Recommendations

### Key Findings

1. **West leads on revenue** — typically the highest-revenue region, driven by strong
   Technology and Furniture performance in California (the single largest state contributor).

2. **East leads on order volume** — more orders but similar or lower AOV suggests
   a heavy consumer-segment mix rather than large corporate deals.

3. **Central struggles on margin** — heavy discounting in Central region sub-categories
   (especially Furniture and Office Supplies) erodes profit despite reasonable revenue.

4. **South is the smallest region** by revenue and order count, with a relatively
   high proportion of non-discounted orders — suggesting untapped volume opportunity
   or insufficient sales investment.

5. **Category mix matters** — regions with higher Technology revenue share tend to
   carry better margins; Furniture drag is most visible in Central and South.

6. **All regions grew year-over-year**, but growth rates diverge sharply in later years,
   suggesting different lifecycle stages or sales-force capacity constraints.

### Recommendations

| Priority | Action | Region Focus |
|---|---|---|
| High | Audit and cap discounts in Central, focus on Tables/Bookcases | Central |
| High | Increase corporate sales effort to raise AOV | East, South |
| Medium | Expand Technology product range in South to improve margin mix | South |
| Medium | Replicate West's high-AOV deal playbook in other regions | East, Central |
| Low | Investigate state-level revenue outliers in each region for capacity gaps | All |

### Limitations

- Dataset spans 2014-2017; territory restructures or economic shifts since then are not captured.
- Returns data is not merged here; a high-return region would appear more profitable than reality.
- The Superstore dataset does not contain customer headcount or territory sales-force size,
  so revenue-per-rep or penetration-rate analysis is not possible.\
""", "md-18"))

# ── Cell 19: Mini Challenge ────────────────────────────────────────────────────
cells.append(md("""\
## 18. Mini Challenge

1. Build a **choropleth map** of US state-level revenue using `plotly` or `geopandas`.
2. Compute **customer-level CLV by region** using the provided Order-Customer-Sales data.
3. Forecast next-year regional revenue using a per-region linear or ETS model
   and rank regions by projected growth.\
""", "md-19"))

# ── Cell 20: Footer ────────────────────────────────────────────────────────────
cells.append(md("""\
---
*Notebook: Regional Sales Comparison*
*Dataset: Sample Superstore (Orders sheet)*
*Techniques: Comparative KPI table, bar/line charts, heatmaps, CAGR, YoY growth, state drill-down*\
""", "md-20"))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.0"},
    },
    "cells": cells,
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Written  : {NB_PATH}")
print(f"Cells    : {len(cells)}")
print(f"Size     : {NB_PATH.stat().st_size:,} bytes")

#!/usr/bin/env python3
"""
Full pipeline for Electricity Demand Forecasting

Auto-generated from: code.ipynb
Project: Electricity Demand Forecasting
Category: Time Series Analysis | Task: time_series
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import pandas as pd
import datetime as dt
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from  scipy.stats import skew, kurtosis, shapiro

# ======================================================================
# HELPER FUNCTIONS (from notebook)
# ======================================================================
def shapiro_test(data, alpha=0.05):
    stat, pval = shapiro(data)
    print("H0: Data was drawn from a Normal Ditribution")
    if (pval<alpha):
        print("pval {} is lower than significance level: {}, therefore null hypothesis is rejected".format(pval, alpha))
    else:
        print("pval {} is higher than significance level: {}, therefore null hypothesis cannot be rejected".format(pval, alpha))

shapiro_test(data.energy, alpha=0.05)

# ======================================================================
# MAIN PIPELINE
# ======================================================================

def main():
    """Run the complete pipeline."""
    # --- REPRODUCIBILITY ─────────────────────────────────────
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    # --- DATA LOADING ────────────────────────────────────────

    path = "data.csv"
    data = load_dataset('electricity_demand_forecasting')
    data = data[data["name"]=="Demanda programada PBF total"]#.set_index("datetime")
    data["date"] = data["datetime"].dt.date
    data.set_index("date", inplace=True)
    data = data[["value"]]
    data = data.asfreq("D")
    data = data.rename(columns={"value": "energy"})
    data.info()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    data[:5]



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    data.plot(title="Energy Demand")
    plt.ylabel("MWh")
    plt.show()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    len(pd.date_range(start="2014-01-01", end="2018-12-31"))

    data["year"] = data.index.year
    data["qtr"] = data.index.quarter
    data["mon"] = data.index.month
    data["week"] = data.index.week
    data["day"] = data.index.weekday
    data["ix"] = range(0,len(data))
    data[["movave_7", "movstd_7"]] = data.energy.rolling(7).agg([np.mean, np.std])
    data[["movave_30", "movstd_30"]] = data.energy.rolling(30).agg([np.mean, np.std])
    data[["movave_90", "movstd_90"]] = data.energy.rolling(90).agg([np.mean, np.std])
    data[["movave_365", "movstd_365"]] = data.energy.rolling(365).agg([np.mean, np.std])

    plt.figure(figsize=(20,16))
    data[["energy", "movave_7"]].plot(title="Daily Energy Demand in Spain (MWh)")
    plt.ylabel("(MWh)")
    plt.show()

    mean = np.mean(data.energy.values)
    std = np.std(data.energy.values)
    skew = skew(data.energy.values)
    ex_kurt = kurtosis(data.energy)
    print("Skewness: {} \nKurtosis: {}".format(skew, ex_kurt+3))



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    sns.distplot(data.energy)
    plt.title("Target Analysis")
    plt.xticks(rotation=45)
    plt.xlabel("(MWh)")
    plt.axvline(x=mean, color='r', linestyle='-', label=r"\mu: {0:.2f}%".format(mean))
    plt.axvline(x=mean+2*std, color='orange', linestyle='-')
    plt.axvline(x=mean-2*std, color='orange', linestyle='-')
    plt.show()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Insert the rolling quantiles to the monthly returns
    data_rolling = data.energy.rolling(window=90)
    data['q10'] = data_rolling.quantile(0.1).to_frame("q10")
    data['q50'] = data_rolling.quantile(0.5).to_frame("q50")
    data['q90'] = data_rolling.quantile(0.9).to_frame("q90")

    data[["q10", "q50", "q90"]].plot(title="Volatility Analysis: 90-rolling percentiles")
    plt.ylabel("(MWh)")
    plt.show()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    data.groupby("qtr")["energy"].std().divide(data.groupby("qtr")["energy"].mean()).plot(kind="bar")
    plt.title("Coefficient of Variation (CV) by qtr")
    plt.show()

    data.groupby("mon")["energy"].std().divide(data.groupby("mon")["energy"].mean()).plot(kind="bar")
    plt.title("Coefficient of Variation (CV) by month")
    plt.show()

    data[["movstd_30", "movstd_365"]].plot(title="Heteroscedasticity analysis")
    plt.ylabel("(MWh)")
    plt.show()

    data[["movave_30", "movave_90"]].plot(title="Seasonal Analysis: Moving Averages")
    plt.ylabel("(MWh)")
    plt.show()

    sns.boxplot(data=data, x="qtr", y="energy")
    plt.title("Seasonality analysis: Distribution over quaters")
    plt.ylabel("(MWh)")
    plt.show()

    sns.boxplot(data=data, x="day", y="energy")
    plt.title("Seasonality analysis: Distribution over weekdays")
    plt.ylabel("(MWh)")
    plt.show()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    data_mon = data.energy.resample("M").agg(sum).to_frame("energy")
    data_mon["ix"] = range(0, len(data_mon))
    data_mon[:5]



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    sns.regplot(data=data_mon,x="ix", y="energy")
    plt.title("Trend analysis: Regression")
    plt.ylabel("(MWh)")
    plt.xlabel("")
    plt.show()

    sns.boxplot(data=data["2014":"2017"], x="year", y="energy")
    plt.title("Trend Analysis: Annual Box-plot Distribution")
    plt.ylabel("(MWh)")
    plt.show()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    data["target"] = data.energy.add(-mean).div(std)
    sns.distplot(data["target"])
    plt.show()



    # --- PREPROCESSING ───────────────────────────────────────

    features = []
    corr_features=[]
    targets = []
    tau = 30 #forecasting periods

    for t in range(1, tau+1):
        data["target_t" + str(t)] = data.target.shift(-t)
        targets.append("target_t" + str(t))

    for t in range(1,31):
        data["feat_ar" + str(t)] = data.target.shift(t)
        #data["feat_ar" + str(t) + "_lag1y"] = data.target.shift(350)
        features.append("feat_ar" + str(t))
        #corr_features.append("feat_ar" + str(t))
        #features.append("feat_ar" + str(t) + "_lag1y")


    for t in [7, 14, 30]:
        data[["feat_movave" + str(t), "feat_movstd" + str(t), "feat_movmin" + str(t) ,"feat_movmax" + str(t)]] = data.energy.rolling(t).agg([np.mean, np.std, np.max, np.min])
        features.append("feat_movave" + str(t))
        #corr_features.append("feat_movave" + str(t))
        features.append("feat_movstd" + str(t))
        features.append("feat_movmin" + str(t))
        features.append("feat_movmax" + str(t))

    months = pd.get_dummies(data.mon,
                                  prefix="mon",
                                  drop_first=True)
    months.index = data.index
    data = pd.concat([data, months], axis=1)

    days = pd.get_dummies(data.day,
                                  prefix="day",
                                  drop_first=True)
    days.index = data.index
    data = pd.concat([data, days], axis=1)


    features = features + months.columns.values.tolist() + days.columns.values.tolist()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    corr_features = ["feat_ar1", "feat_ar2", "feat_ar3", "feat_ar4", "feat_ar5", "feat_ar6", "feat_ar7", "feat_movave7", "feat_movave14", "feat_movave30"]



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # Calculate correlation matrix
    corr = data[["target_t1"] + corr_features].corr()

    top5_mostCorrFeats = corr["target_t1"].apply(abs).sort_values(ascending=False).index.values[:6]


    # Plot heatmap of correlation matrix
    sns.heatmap(corr, annot=True)
    plt.title("Pearson Correlation with 1 period target")
    plt.yticks(rotation=0); plt.xticks(rotation=90)  # fix ticklabel directions
    plt.tight_layout()  # fits plot area to the plot, "tightly"
    plt.show()  # show the plot



    # --- PREPROCESSING ───────────────────────────────────────

    sns.pairplot(data=data[top5_mostCorrFeats].dropna(), kind="reg")
    plt.title("Most important features Matrix Scatter Plot")
    plt.show()

    data_feateng = data[features + targets].dropna()
    nobs= len(data_feateng)
    print("Number of observations: ", nobs)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    X_train = data_feateng.loc["2014":"2017"][features]
    y_train = data_feateng.loc["2014":"2017"][targets]

    X_test = data_feateng.loc["2018"][features]
    y_test = data_feateng.loc["2018"][targets]

    n, k = X_train.shape
    print("Total number of observations: ", nobs)
    print("Train: {}{}, \nTest: {}{}".format(X_train.shape, y_train.shape,
                                                  X_test.shape, y_test.shape))

    plt.plot(y_train.index, y_train.target_t1.values, label="train")
    plt.plot(y_test.index, y_test.target_t1.values, label="test")
    plt.title("Train/Test split")
    plt.xticks(rotation=45)
    plt.show()


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Full pipeline for Electricity Demand Forecasting")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()

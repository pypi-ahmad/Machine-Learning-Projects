#!/usr/bin/env python3
"""
Full pipeline for Top Billionaires List Analysis

Auto-generated from: code.ipynb
Project: Top Billionaires List Analysis
Category: Data Analysis | Task: data_analysis
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import os
# Additional imports extracted from mixed cells
import matplotlib.pyplot as plt
import seaborn as sns

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

    df = load_dataset('top_billionaires_list_analysis')
    df.head(5)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df.isna().sum()



    # --- PREPROCESSING ───────────────────────────────────────

    df['Demographics Gender'].fillna(df['Demographics Gender'].mode()[0], inplace=True)
    df['Wealth Type'].fillna(df['Wealth Type'].mode()[0], inplace=True)
    df['Wealth How Category'].fillna(df['Wealth How Category'].mode()[0], inplace=True)
    df['Wealth How Industry'].fillna(df['Wealth How Industry'].mode()[0], inplace=True)
    df['Company Name'].fillna(df['Company Name'].mode()[0], inplace=True)
    df['Company Relationship'].fillna(df['Company Relationship'].mode()[0], inplace=True)
    df['Company Sector'].fillna(df['Company Sector'].mode()[0], inplace=True)
    df['Company Type'].fillna(df['Company Type'].mode()[0], inplace=True)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df.info()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    df[df['Demographics Gender'] == 'male']

    df['Demographics Gender'].unique()

    df['Company Relationship'].unique()

    df[df['Company Relationship'] == 'owner']

    df[(df['Company Relationship'] == 'owner') & (df['Demographics Age'] > 40)]

    df[df['Rank'] < 11]



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df.columns

    # Distribution of billionaire ages
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Demographics Age', bins=30, kde=True)
    plt.title('Distribution of Billionaire Ages')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.show()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Scatter plot showing the relationship between wealth worth and age
    fig = px.scatter(df, x='Demographics Age', y='Wealth Worth In Billions',
                     color='Company Sector', hover_name='Name',
                     title='Wealth Worth vs. Age of Billionaires')
    fig.show()

    # Filter the DataFrame to include only the required columns
    df_filtered = df[['Company Relationship', 'Wealth Worth In Billions']]

    # Group the data by 'Company Relationship' and calculate the average wealth worth
    df_grouped = df_filtered.groupby('Company Relationship').mean()

    # Create the line plot
    plt.plot(df_grouped.index, df_grouped['Wealth Worth In Billions'], marker='o')

    # Set the plot title and labels
    plt.title('Average Wealth Worth by Company Relationship')
    plt.xlabel('Company Relationship')
    plt.ylabel('Wealth Worth In Billions')

    # Rotate the x-axis labels for better readability (optional)
    plt.xticks(rotation=90, fontsize=5.5)

    # Display the plot
    plt.show()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    # Calculate the correlation matrix
    corr_matrix = df[['Demographics Age', 'Wealth Worth In Billions', 'Location GDP']].corr()

    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # Select a subset of columns for the pair plot
    subset_columns = ['Demographics Age', 'Wealth Worth In Billions', 'Company Sector']

    # Create a pair plot
    sns.pairplot(data=df[subset_columns], hue='Company Sector')
    plt.title('Pairwise Relationships')
    plt.show()

    # Violin plot showing the wealth worth by company type
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='Company Type', y='Wealth Worth In Billions')
    plt.xticks(rotation=90)
    plt.title('Wealth Worth by Company Type')
    plt.show()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Summary statistics
    df.describe(include='all')

    # Correlation matrix for numeric columns
    import matplotlib.pyplot as plt
    import seaborn as sns

    numeric_df = df.select_dtypes(include='number')
    if len(numeric_df.columns) > 1:
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Full pipeline for Top Billionaires List Analysis")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()

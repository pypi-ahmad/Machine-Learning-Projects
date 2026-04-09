#!/usr/bin/env python3
"""
Model evaluation for Coffee Quality Analysis

Auto-generated from: code.ipynb
Project: Coffee Quality Analysis
Category: Data Analysis | Task: classification
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors
from dateutil import parser

# Import Scikit-learn for Machine Learning libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.manifold import TSNE
# Additional imports extracted from mixed cells
from lazypredict.Supervised import LazyClassifier
from pycaret.classification import *

# ======================================================================
# EVALUATION PIPELINE
# ======================================================================

def main():
    """Run the evaluation pipeline."""
    USE_AUTOML = True  # Set to False to skip AutoML comparison

    # --- REPRODUCIBILITY ─────────────────────────────────────
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    # --- DATA LOADING ────────────────────────────────────────

    df = load_dataset('coffee_quality_analysis')



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Handle duplicates
    duplicate_rows_data = df[df.duplicated()]
    print("number of duplicate rows: ", duplicate_rows_data.shape)

    # Loop through each column and count the number of distinct values
    for column in df.columns:
        num_distinct_values = len(df[column].unique())
        print(f"{column}: {num_distinct_values} distinct values")



    # --- FEATURE ENGINEERING ─────────────────────────────────

    #check missing ratio
    data_na = (df.isnull().sum() / len(df)) * 100
    data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio' :data_na})
    missing_data.head(20)



    # --- PREPROCESSING ───────────────────────────────────────

    # Mapping the Education
    processing_mapping = {
        "Double Anaerobic Washed": "Washed / Wet",
        "Semi Washed": "Washed / Wet",
        "Honey,Mossto": "Pulped natural / honey",
        "Double Carbonic Maceration / Natural": "Natural / Dry",
        "Wet Hulling": "Washed / Wet",
        "Anaerobico 1000h": "Washed / Wet",
        "SEMI-LAVADO": "Natural / Dry"
    }
    # Fixing the values in the column
    df['Processing Method'] = df['Processing Method'].map(processing_mapping)
    df['Processing Method'].fillna("Washed / Wet", inplace=True)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # Manually impute specific values based on ID (Which we cant use function)
    df.loc[df['ID'] == 99, 'Altitude'] = 5273  # Impute value for ID 99
    df.loc[df['ID'] == 105, 'Altitude'] = 1800  # Impute value for ID 105
    df.loc[df['ID'] == 180, 'Altitude'] = 1400  # Impute value for ID 180


    # Define a function to clean and calculate the mean
    def clean_altitude_range(range_value):
        if isinstance(range_value, str):
            range_value = range_value.replace(" ", "")  # Remove blank spaces
            if '-' in range_value:
                try:
                    start, end = range_value.split('-')
                    start = int(start)
                    end = int(end)
                    return (start + end) / 2
                except ValueError:
                    return np.nan
            else:
                try:
                    return int(range_value)
                except ValueError:
                    return np.nan
        else:
            return range_value

    # Apply the function to clean and calculate the mean for each value in the "Altitude" column
    df['Altitude'] = df['Altitude'].apply(clean_altitude_range)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Extract the prior year from the "Harvest Year" column
    df['Harvest Year'] = df['Harvest Year'].str.split('/').str[0].str.strip()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # Convert "Harvest Year" and "Expiration" columns to datetime objects using dateutil parser
    df['Harvest Year'] = pd.to_datetime(df['Harvest Year'], format='%Y')
    df['Expiration'] = df['Expiration'].apply(parser.parse)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Calculate the difference in days between "Expiration" and "Harvest Year" columns
    df['Coffee Age'] = (df['Expiration'] - df['Harvest Year']).dt.days



    # --- FEATURE ENGINEERING ─────────────────────────────────

    columns_to_drop = ['ID','ICO Number','Owner','Region','Certification Contact','Certification Address','Farm Name',"Lot Number","Mill","ICO Number","Producer",'Company','Expiration', 'Harvest Year',
                       "Unnamed: 0",'Number of Bags','Bag Weight','In-Country Partner','Grading Date','Variety','Status','Defects','Uniformity','Clean Cup','Sweetness','Certification Body']
    df.drop(columns_to_drop, axis=1, inplace=True)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # List of numeric attributes
    numeric_attributes = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 'Overall', 'Total Cup Points', 'Moisture Percentage','Coffee Age']

    # Create a subplot for each numeric attribute
    fig = make_subplots(rows=len(numeric_attributes), cols=1)

    # Add a histogram to the subplot for each numeric attribute
    for i, attribute in enumerate(numeric_attributes):
        fig.add_trace(go.Histogram(x=df[attribute], nbinsx=50, name=attribute), row=i+1, col=1)

    fig.update_layout(height=200*len(numeric_attributes), width=800, title_text="Histograms of Numeric Attributes")
    fig.show()

    # Group the data by country and calculate the mean of Total Cup Points
    df_grouped = df.groupby('Country of Origin')['Total Cup Points'].mean().reset_index()

    # Create a Choropleth map
    fig = px.choropleth(df_grouped,
                        locations='Country of Origin',
                        locationmode='country names',
                        color='Total Cup Points',
                        hover_name='Country of Origin',
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title='Average Total Cup Points by Country')

    fig.show()

    # Create a bar plot with gray color
    plt.figure(figsize=(10, 5))
    sns.barplot(x=df_grouped['Country of Origin'], y=df_grouped['Total Cup Points'], color='gray')
    plt.title('Average Total Cup Points by Country')
    plt.xlabel('Country of Origin')
    plt.ylabel('Average Total Cup Points')
    plt.xticks(rotation=90)
    plt.show()

    # Group the data by country and calculate the mean of Total Cup Points
    df_grouped = df.groupby('Country of Origin')['Coffee Age'].mean().reset_index()

    # Create a Choropleth map
    fig = px.choropleth(df_grouped,
                        locations='Country of Origin',
                        locationmode='country names',
                        color='Coffee Age',
                        hover_name='Country of Origin',
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title='Average Coffee Shelter Life by Country (Days)')

    fig.show()

    # Create a bar plot with gray color
    plt.figure(figsize=(10, 5))
    sns.barplot(x=df_grouped['Country of Origin'], y=df_grouped['Coffee Age'], color='gray')
    plt.title('Average Coffee Shelter Life by Country (Days)')
    plt.xlabel('Country of Origin')
    plt.ylabel('Average Coffee Shelter Life')
    plt.xticks(rotation=90)
    plt.show()

    # Group the data by country and calculate the mean of Altitude
    df_grouped = df.groupby('Country of Origin')['Altitude'].mean().reset_index()

    # Create a Choropleth map
    fig = px.choropleth(df_grouped,
                        locations='Country of Origin',
                        locationmode='country names',
                        color='Altitude',
                        hover_name='Country of Origin',
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title='Average Altitude by Country')

    fig.show()

    # Create a bar plot with gray color
    plt.figure(figsize=(10, 5))
    sns.barplot(x=df_grouped['Country of Origin'], y=df_grouped['Altitude'], color='gray')
    plt.title('Average Altitude by Country')
    plt.xlabel('Country of Origin')
    plt.ylabel('Average Altitude')
    plt.xticks(rotation=90)
    plt.show()

    # Count the unique occurrences of each country
    df_count = df['Country of Origin'].value_counts().reset_index()
    df_count.columns = ['Country of Origin', 'Count']

    # Create a choropleth map
    fig = px.choropleth(df_count,
                        locations='Country of Origin',
                        locationmode='country names',
                        color='Count',
                        hover_name='Country of Origin',
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title='Count of Unique Countries')

    fig.show()

    # Create a bar plot
    plt.figure(figsize=(10, 5))
    sns.barplot(x=df_count['Country of Origin'], y=df_count['Count'], color='gray')
    plt.title('Count of Unique Countries')
    plt.xlabel('Country of Origin')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.show()



    # --- PREPROCESSING ───────────────────────────────────────

    data = df.copy()
    categorical_columns = ['Processing Method']
    numerical_columns = ['Altitude', 'Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 'Overall', 'Total Cup Points', 'Moisture Percentage', 'Category One Defects', 'Quakers', 'Category Two Defects', 'Coffee Age']
    columns_to_drop = ['Country of Origin', 'Color']
    data.drop(columns_to_drop, axis=1, inplace=True)
    dummy_variables = pd.get_dummies(data, columns=categorical_columns, drop_first=False)

    scaler = StandardScaler()

    # Scale the numerical columns
    scaled_numerical = scaler.fit_transform(data[numerical_columns])

    # Convert the scaled numerical columns
    scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_columns)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # Drop the original numerical columns
    dummy_variables = dummy_variables.drop(numerical_columns, axis=1)

    # Concatenate the dummy variables and scaled numerical columns
    processed_df = pd.concat([dummy_variables, scaled_numerical_df], axis=1)

    correlation_matrix = processed_df.corr()

    #Graph I.
    plt.figure(figsize=(15, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
    plt.title("Correlation Matrix Heatmap")
    plt.show()

    corr = processed_df.corr()
    target_corr = corr['Total Cup Points'].drop('Total Cup Points')

    # Sort correlation values in descending order
    target_corr_sorted = target_corr.sort_values(ascending=False)

    #Graph II
    # Create a heatmap of the correlations with the target column
    sns.set(font_scale=0.8)
    sns.set_style("white")
    sns.set_palette("PuBuGn_d")
    sns.heatmap(target_corr_sorted.to_frame(), cmap="coolwarm", annot=True, fmt='.2f')
    plt.title('Correlation with Total Cup Points')
    plt.show()



    # --- PREPROCESSING ───────────────────────────────────────

    # Create a copy of the dataframe to not alter the original
    df_preprocessed = df.copy()

    # Preprocessing: Label encoding for categorical variables
    le = LabelEncoder()
    categorical_features = ['Country of Origin', 'Processing Method', 'Color']
    for feature in categorical_features:
        df_preprocessed[feature] = le.fit_transform(df[feature])

    # Preprocessing: MinMax scaling for numerical/ratio variables
    mm = MinMaxScaler()
    numerical_features = ['Altitude', 'Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 'Overall', 'Total Cup Points', 'Moisture Percentage', 'Category One Defects', 'Quakers', 'Category Two Defects', 'Coffee Age']
    for feature in numerical_features:
        df_preprocessed[feature] = mm.fit_transform(df[feature].values.reshape(-1,1))

    # Apply t-SNE with different perplexity and learning rate
    tsne = TSNE(n_components=2, random_state=42, perplexity=50, learning_rate=200)
    tsne_results = tsne.fit_transform(df_preprocessed)

    # Plotly Interactive plot
    df_tsne = pd.DataFrame(data = tsne_results, columns = ['Dim_1', 'Dim_2'])
    df_tsne['Total Cup Points'] = df['Total Cup Points']
    fig = px.scatter(df_tsne, x='Dim_1', y='Dim_2', color='Total Cup Points', title='t-SNE plot colored by Total Cup Points')
    fig.show()

    categorical_columns = ['Processing Method','Country of Origin', 'Color']
    numerical_columns = ['Altitude', 'Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 'Overall', 'Total Cup Points', 'Moisture Percentage', 'Category One Defects', 'Quakers', 'Category Two Defects', 'Coffee Age']

    # Define preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), [col for col in numerical_columns if col != 'Total Cup Points']),
            ('cat', OneHotEncoder(), categorical_columns)])

    # Append classifier to preprocessing pipeline.
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestRegressor(n_estimators=100, random_state=42))])

    # Split the data into train and test sets
    X = df.drop('Total Cup Points', axis=1)
    y = df['Total Cup Points']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



    # --- AUTOML COMPARISON ────────────────────────────────────

    if USE_AUTOML:

        try:

            # --- LAZYPREDICT BASELINE ────────────────────────

            from lazypredict.Supervised import LazyClassifier

            lazy_clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
            models, predictions = lazy_clf.fit(X_train, X_test, y_train, y_test)

            print(models)



    # --- PYCARET AUTOML ──────────────────────────────────────

            from pycaret.classification import *

            clf_setup = setup(data=df, target='Total.Cup.Points', session_id=42, verbose=False)

            # Compare models and select best
            best_model = compare_models()

            # Display comparison results
            print(best_model)

            # Evaluate the best model
            evaluate_model(best_model)

            # Finalize the model (train on full dataset)
            final_model = finalize_model(best_model)

            print('Final model:', final_model)



        except ImportError:

            print('[AutoML] LazyPredict/PyCaret not installed — skipping AutoML block')

        except Exception as _automl_err:

            print(f'[AutoML] AutoML block failed: {_automl_err}')


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Model evaluation for Coffee Quality Analysis")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()

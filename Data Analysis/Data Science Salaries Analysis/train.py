#!/usr/bin/env python3
"""
Model training for Data Science Salaries Analysis

Auto-generated from: code.ipynb
Project: Data Science Salaries Analysis
Category: Data Analysis | Task: classification
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# Additional imports extracted from mixed cells
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
import pycountry
import plotly.io as pio
from sklearn.preprocessing import StandardScaler
from lazypredict.Supervised import LazyClassifier
from pycaret.classification import *

# ======================================================================
# HELPER FUNCTIONS (from notebook)
# ======================================================================
def assign_broader_category(job_title):
    data_engineering = ["Data Engineer", "Data Analyst", "Analytics Engineer", "BI Data Analyst", "Business Data Analyst", "BI Developer", "BI Analyst", "Business Intelligence Engineer", "BI Data Engineer", "Power BI Developer"]
    data_scientist = ["Data Scientist", "Applied Scientist", "Research Scientist", "3D Computer Vision Researcher", "Deep Learning Researcher", "AI/Computer Vision Engineer"]
    machine_learning = ["Machine Learning Engineer", "ML Engineer", "Lead Machine Learning Engineer", "Principal Machine Learning Engineer"]
    data_architecture = ["Data Architect", "Big Data Architect", "Cloud Data Architect", "Principal Data Architect"]
    management = ["Data Science Manager", "Director of Data Science", "Head of Data Science", "Data Scientist Lead", "Head of Machine Learning", "Manager Data Management", "Data Analytics Manager"]

    if job_title in data_engineering:
        return "Data Engineering"
    elif job_title in data_scientist:
        return "Data Science"
    elif job_title in machine_learning:
        return "Machine Learning"
    elif job_title in data_architecture:
        return "Data Architecture"
    elif job_title in management:
        return "Management"
    else:
        return "Other"

# Apply the function to the 'job_title' column and create a new column 'job_category'
data['job_category'] = data['job_title'].apply(assign_broader_category)
# Function to convert ISO 3166 country code to country name
def country_code_to_name(country_code):
    try:
        return pycountry.countries.get(alpha_2=country_code).name
    except:
        return country_code
    # Function to convert country code to full name
def country_code_to_name(code):
    try:
        country = pycountry.countries.get(alpha_2=code)
        return country.name
    except:
        return None

# ======================================================================
# TRAINING PIPELINE
# ======================================================================

def main():
    """Run the training pipeline."""
    USE_AUTOML = True  # Set to False to skip AutoML comparison

    # --- REPRODUCIBILITY ─────────────────────────────────────
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    # --- DATA LOADING ────────────────────────────────────────

    data = load_dataset('data_science_salaries_analysis')



    # --- FEATURE ENGINEERING ─────────────────────────────────

    #check missing ratio
    data_na = (data.isnull().sum() / len(data)) * 100
    data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio' :data_na})
    missing_data.head(20)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Handle duplicates
    duplicate_rows_data = data[data.duplicated()]
    print("number of duplicate rows: ", duplicate_rows_data.shape)

    # Loop through each column and count the number of distinct values
    for column in data.columns:
        num_distinct_values = len(data[column].unique())
        print(f"{column}: {num_distinct_values} distinct values")



    # --- FEATURE ENGINEERING ─────────────────────────────────

    data['experience_level'] = data['experience_level'].replace({
        'SE': 'Senior',
        'EN': 'Entry level',
        'EX': 'Executive level',
        'MI': 'Mid/Intermediate level',
    })

    data['employment_type'] = data['employment_type'].replace({
        'FL': 'Freelancer',
        'CT': 'Contractor',
        'FT' : 'Full-time',
        'PT' : 'Part-time'
    })
    data['company_size'] = data['company_size'].replace({
        'S': 'SMALL',
        'M': 'MEDIUM',
        'L' : 'LARGE',
    })
    data['remote_ratio'] = data['remote_ratio'].astype(str)
    data['remote_ratio'] = data['remote_ratio'].replace({
        '0': 'On-Site',
        '50': 'Half-Remote',
        '100' : 'Full-Remote',
    })

    # Inflation rates
    us_inflation_rates = {2019: 0.0181, 2020: 0.0123, 2021: 0.0470, 2022: 0.065}
    global_inflation_rates = {2019: 0.0219, 2020: 0.0192, 2021: 0.0350, 2022: 0.088}

    # Function to adjust salary
    def adjust_salary(row):
        year = row['work_year']
        original_salary = row['salary_in_usd']
        currency = row['salary_currency']

        if year == 2023:
            return original_salary

        adjusted_salary = original_salary
        for y in range(year, 2023):
            if currency == 'USD':
                inflation_rate = us_inflation_rates[y]
            else:
                inflation_rate = global_inflation_rates[y]

            adjusted_salary *= (1 + inflation_rate)

        return adjusted_salary

    # Apply the function to the dataset
    data['adjusted_salary'] = data.apply(adjust_salary, axis=1)

    #------------
    #credit : @rrrrrrita
    #------------



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    value_counts = data['job_category'].value_counts(normalize=True) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    top_n = min(17, len(value_counts))
    ax.barh(value_counts.index[:top_n], value_counts.values[:top_n])
    ax.set_xlabel('Percentage')
    ax.set_ylabel('Job Category')
    ax.set_title('Job Titles Percentage')
    plt.show()



    # --- MODEL TRAINING ──────────────────────────────────────

    # Create a list of the columns to analyze
    columns = ['adjusted_salary']

    # Loop over the columns and plot the distribution of each variable
    for col in columns:
        # Plot the distribution of the data
        sns.histplot(data[col], kde=True)

        # Fit a normal distribution to the data
        (mu, sigma) = stats.norm.fit(data[col])
        print('{}: mu = {:.2f}, sigma = {:.2f}'.format(col, mu, sigma))

        # Calculate the skewness and kurtosis of the data
        print('{}: Skewness: {:.2f}'.format(col, data[col].skew()))
        print('{}: Kurtosis: {:.2f}'.format(col, data[col].kurt()))

        # Add the fitted normal distribution to the plot
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        y = stats.norm.pdf(x, mu, sigma)
        plt.plot(x, y, label='Normal fit')

        # Add labels and title to the plot
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.title('Distribution of {}'.format(col))

        # Plot the QQ-plot
        fig = plt.figure()
        stats.probplot(data[col], plot=plt)

        plt.show()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # apply formatting to describe method for 'adjusted_salary' column
    formatted_data = data.loc[:, 'adjusted_salary'].describe().apply(lambda x: f'{x:.2f}')

    # create boxplot and swarmplot for 'adjusted_salary' column
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=data['adjusted_salary'], palette='coolwarm')
    sns.swarmplot(x=data['adjusted_salary'], color='blue', alpha=0.4, size=2.5)
    plt.ylabel('Adjusted Salary')
    plt.title('Boxplot and Swarmplot of Adjusted Salary')
    plt.show()

    # apply styling to formatted data
    styled_data = formatted_data.to_frame().style \
        .background_gradient(cmap='Blues') \
        .set_properties(**{'text-align': 'center', 'border': '1px solid black'})

    # display styled data
    display(styled_data)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    df = data.copy()

    #  Median salary by job title
    pivot_table = df.pivot_table(values='adjusted_salary', index='job_category', columns='work_year', aggfunc='median')
    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title('Median Salary by Year')
    plt.xlabel('Year')
    plt.ylabel('Job Title')
    plt.show()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # Convert country codes to names
    df['company_location'] = df['company_location'].apply(country_code_to_name)
    df['employee_residence'] = df['employee_residence'].apply(country_code_to_name)

    # Average salary by company_location
    avg_salary_by_location = df.groupby('company_location', as_index=False)['adjusted_salary'].mean()

    fig1 = px.choropleth(avg_salary_by_location,
                         locations='company_location',
                         locationmode='country names',
                         color='adjusted_salary',
                         hover_name='company_location',
                         color_continuous_scale=px.colors.sequential.Plasma,
                         title='Average Salary by Company Location',
                         labels={'adjusted_salary': 'Average Adjusted Salary'},
                         projection='natural earth')

    fig1.show()

    # Average salary by company_location
    avg_salary_by_location = df.groupby('company_location')['adjusted_salary'].mean().sort_values(ascending=False)
    plt.figure(figsize=(14, 6))
    sns.barplot(x=avg_salary_by_location.index, y=avg_salary_by_location, color='grey')
    plt.title('Average Salary by Company Location (Yearly)')
    plt.xlabel('Company Location')
    plt.ylabel('Average Adjusted Salary (Yearly)')
    plt.xticks(rotation=90)
    plt.show()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Average salary by employee_residence
    avg_salary_by_residence = df.groupby('employee_residence', as_index=False)['adjusted_salary'].mean()

    fig2 = px.choropleth(avg_salary_by_residence,
                         locations='employee_residence',
                         locationmode='country names',
                         color='adjusted_salary',
                         hover_name='employee_residence',
                         color_continuous_scale=px.colors.sequential.Plasma,
                         title='Average Salary by Employee Residence',
                         labels={'adjusted_salary': 'Average Adjusted Salary'},
                         projection='natural earth')

    fig2.show()

    # Average salary by employee_residence
    avg_salary_by_residence = df.groupby('employee_residence')['adjusted_salary'].mean().sort_values(ascending=False)
    plt.figure(figsize=(14, 6))
    sns.barplot(x=avg_salary_by_residence.index, y=avg_salary_by_residence.values, color='grey')
    plt.title('Average Salary by Employee Residence (Yearly)')
    plt.xlabel('Employee Residence')
    plt.ylabel('Average Adjusted Salary (Yearly)')
    plt.xticks(rotation=90)
    plt.show()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # Filter for remote_ratio of 100
    remote_100 = data[data['remote_ratio'] == 'Full-Remote']

    # Aggregate by country code
    country_counts = remote_100['company_location'].value_counts().reset_index()
    country_counts.columns = ['country_code', 'count']

    # Convert country codes to full names
    country_counts['country_name'] = country_counts['country_code'].apply(country_code_to_name)

    # Create the choropleth map with a logarithmic color scale
    fig = px.choropleth(country_counts,
                        locations='country_name',
                        locationmode='country names',
                        color=np.log10(country_counts['count']),
                        hover_name='country_name',
                        hover_data=['count'],
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title='Choropleth Map of Full-Remote Company Locations',
                        projection='natural earth')

    # Customize the colorbar to show the original count values
    fig.update_coloraxes(colorbar=dict(title='Count (Log Scale)', tickvals=[0, 1, 2, 3], ticktext=['1', '10', '100', '1000']))

    # Show the map
    plt.show()

    # Filter for remote_ratio of 100
    remote_0 = data[data['remote_ratio'] == 'On-Site']

    # Aggregate by country code
    country_counts = remote_0['company_location'].value_counts().reset_index()
    country_counts.columns = ['country_code', 'count']

    # Convert country codes to full names
    country_counts['country_name'] = country_counts['country_code'].apply(country_code_to_name)

    # Create the choropleth map with a logarithmic color scale
    fig = px.choropleth(country_counts,
                        locations='country_name',
                        locationmode='country names',
                        color=np.log10(country_counts['count']),
                        hover_name='country_name',
                        hover_data=['count'],
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title='Choropleth Map of On-Site Company Locations',
                        projection='natural earth')

    # Customize the colorbar to show the original count values
    fig.update_coloraxes(colorbar=dict(title='Count (company)', tickvals=[0, 1, 2, 3], ticktext=['1', '10', '100', '1000']))

    # Show the map
    plt.show()



    # --- PREPROCESSING ───────────────────────────────────────

    dummy_variables = pd.get_dummies(df, columns=categorical_columns, drop_first=False)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    # Scale the numerical columns
    scaled_numerical = scaler.fit_transform(df[numerical_columns])

    # Convert the scaled numerical columns
    scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_columns)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # Drop the original numerical columns
    dummy_variables = dummy_variables.drop(numerical_columns, axis=1)

    # Concatenate the dummy variables and scaled numerical columns
    processed_df = pd.concat([dummy_variables, scaled_numerical_df], axis=1)
    processed_df = processed_df.drop(['work_year', 'salary','salary_in_usd'], axis=1)

    correlation_matrix = processed_df.corr()

    #Graph I.
    plt.figure(figsize=(15, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
    plt.title("Correlation Matrix Heatmap")
    plt.show()

    corr = processed_df.corr()
    target_corr = corr['adjusted_salary'].drop('adjusted_salary')

    # Sort correlation values in descending order
    target_corr_sorted = target_corr.sort_values(ascending=False)

    #Graph II
    # Create a heatmap of the correlations with the target column
    sns.set(font_scale=0.8)
    sns.set_style("white")
    sns.set_palette("PuBuGn_d")
    sns.heatmap(target_corr_sorted.to_frame(), cmap="coolwarm", annot=True, fmt='.2f')
    plt.title('Correlation with Salary')
    plt.show()

    # create dictionary of country code to country name mappings
    country_map = {}
    for country in pycountry.countries:
        country_map[country.alpha_2] = country.name
    # replace values in 'employee_residence' column using dictionary
    data['employee_residence'] = data['employee_residence'].replace(country_map)
    data['company_location'] = data['company_location'].replace(country_map)

    df = data.copy()
    df = df.drop(['work_year','salary','salary_currency','salary_in_usd','salary_in_usd','job_title'], axis=1)



    # --- PREPROCESSING ───────────────────────────────────────

    # Create a copy of the dataframe to not alter the original
    df_preprocessed = df.copy()

    # Preprocessing: Label encoding for categorical variables
    le = LabelEncoder()
    categorical_features = ['experience_level', 'employment_type', 'job_category', 'employee_residence', 'company_location', 'company_size', 'remote_ratio']
    for feature in categorical_features:
        df_preprocessed[feature] = le.fit_transform(df[feature])

    # Preprocessing: MinMax scaling for numerical/ratio variables
    mm = MinMaxScaler()
    numerical_features = ['adjusted_salary']
    for feature in numerical_features:
        df_preprocessed[feature] = mm.fit_transform(df[feature].values.reshape(-1,1))



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Apply t-SNE with different perplexity and learning rate
    tsne = TSNE(n_components=2, random_state=42, perplexity=50, learning_rate=200)
    tsne_results = tsne.fit_transform(df_preprocessed)

    # Plotly Interactive plot
    df_tsne = pd.DataFrame(data = tsne_results, columns = ['Dim_1', 'Dim_2'])
    df_tsne['adjusted_salary'] = df['adjusted_salary']
    fig = px.scatter(df_tsne, x='Dim_1', y='Dim_2', color='adjusted_salary', title='t-SNE plot colored by Salary')
    fig.show()

    # Outlier detection using IQR method
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # Calculate quantiles for salary bin edges
    quantiles = [0, 1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 1]
    bin_edges = [df['adjusted_salary'].quantile(q) for q in quantiles]

    # Convert the continuous salary variable into 7 discrete bins based on quantiles
    salary_labels = ['low', 'low-mid', 'mid', 'mid-high', 'high', 'very-high', 'Top']
    df['salary_range'] = pd.cut(df['adjusted_salary'], bins=bin_edges, labels=salary_labels, include_lowest=True)



    # --- PREPROCESSING ───────────────────────────────────────

    # Label encoding for categorical features
    encoder = LabelEncoder()
    categorical_features = ['employment_type', 'job_category', 'experience_level',
                            'employee_residence', 'remote_ratio', 'company_location', 'company_size']
    for feature in categorical_features:
        data[feature] = encoder.fit_transform(data[feature])
    # Split the dataset into training and testing sets
    X = data.drop(["adjusted_salary", "salary_range"], axis=1)
    y = data["salary_range"]
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

            clf_setup = setup(data=data, target='salary_range', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Model training for Data Science Salaries Analysis")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()

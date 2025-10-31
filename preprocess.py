# File: preprocess.py
# Description: Helper functions for feature engineering and ML preprocessing.

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# -----------------------------------------------------------------
# 1. QUANTITATIVE FEATURE ENGINEERING (Pandas-based)
# -----------------------------------------------------------------

def engineer_financial_features(df, price_col='close', windows=[10, 30]):
    """
    Adds common financial features to a DataFrame.
    
    Assumes a multi-index DataFrame (date, ticker) or a 
    DataFrame with a 'ticker' column for grouping.
    
    Args:
        df (pd.DataFrame): Input data with at least a price column and a 'ticker' column.
        price_col (str): The name of the column to use for price data (e.g., 'close', 'adj_close').
        windows (list): List of integers for moving average and volatility windows.

    Returns:
        pd.DataFrame: DataFrame with new feature columns.
    """
    
    if 'ticker' not in df.columns:
        print("Warning: 'ticker' column not found. Calculating features across all data.")
        # Create a dummy group
        df['ticker'] = 'all'
        
    df_feat = df.copy().sort_index(level='date' if isinstance(df.index, pd.MultiIndex) else 'date')
    asset_groups = df_feat.groupby('ticker')

    # 1. Log Returns
    # We use transform to align the output with the original DataFrame's index
    df_feat[f'{price_col}_log_return'] = asset_groups[price_col].transform(
        lambda x: np.log(x / x.shift(1))
    )

    for w in windows:
        # 2. Moving Averages
        df_feat[f'ma_{w}'] = asset_groups[price_col].transform(
            lambda x: x.rolling(window=w).mean()
        )
        
        # 3. Volatility (Rolling Std Dev of Log Returns)
        df_feat[f'vol_{w}'] = df_feat[f'{price_col}_log_return'].groupby(df_feat['ticker']).transform(
            lambda x: x.rolling(window=w).std()
        )

        # 4. Momentum (Price vs. Moving Average)
        df_feat[f'momentum_{w}'] = df_feat[price_col] / df_feat[f'ma_{w}'] - 1
    
    # Drop NaNs created by shifts and rolls, which models can't handle
    df_feat = df_feat.dropna()
    
    if 'all' in df_feat['ticker'].unique():
        df_feat = df_feat.drop(columns=['ticker'])
        
    return df_feat

# -----------------------------------------------------------------
# 2. SKLEARN PREPROCESSING PIPELINE (Scikit-learn-based)
# -----------------------------------------------------------------

def create_preprocessing_pipeline(numeric_features, categorical_features):
    """
    Creates a ColumnTransformer pipeline for general ML preprocessing.
    This is run *after* feature engineering.

    Args:
        numeric_features (list): List of names of numeric columns to be scaled.
        categorical_features (list): List of names of categorical columns to be encoded.

    Returns:
        sklearn.compose.ColumnTransformer: A preprocessor object to be used in a model pipeline.
    """

    # --- Define individual transformers ---

    # Pipeline for numeric features:
    # 1. Impute missing values (e.g., from volatility) with the median
    # 2. Scale features to have zero mean and unit variance
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical features:
    # 1. Impute missing categories with a constant string 'missing'
    # 2. One-hot encode the categories, ignoring new categories seen at test time
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # --- Combine transformers using ColumnTransformer ---
    
    # This applies the correct transformer to the correct set of columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # Keep any columns not specified (e.g., date or ticker if needed)
    )

    return preprocessor

# -----------------------------------------------------------------
# 3. HOW TO USE IN YOUR MAIN NOTEBOOK (as comments)
# -----------------------------------------------------------------

"""
# === In your main_project.ipynb ===

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# 1. Import your helper functions
import preprocess
import data_sourcing  # (Assuming you have this)

# 2. Load data
# raw_data = data_sourcing.get_stock_data(['AAPL', 'MSFT'], start='2020-01-01', end='2023-12-31')
# 'raw_data' might look like:
# date         ticker   close   volume   sector
# 2020-01-02   AAPL     150.0   1.2M     'Tech'
# 2020-01-02   MSFT     200.0   0.9M     'Tech'
# ...

# 3. Engineer Quant-Specific Features
# engineered_data = preprocess.engineer_financial_features(raw_data, price_col='close', windows=[20, 50])

# 4. Define features (X) and target (y)
# This is a critical step. Your 'target' is what you want to predict (e.g., next day's return)
# engineered_data['target'] = engineered_data.groupby('ticker')['close_log_return'].shift(-1)
# engineered_data = engineered_data.dropna() # Drop NaNs from target shift

# Define which columns are features
numeric_features = ['ma_20', 'vol_20', 'ma_50', 'vol_50', 'momentum_20', 'volume']
categorical_features = ['sector'] # 'ticker' could also be a categorical feature
target = 'target'

features = numeric_features + categorical_features
X = engineered_data[features]
y = engineered_data[target]

# 5. Split data (IMPORTANT: For time series, DO NOT SHUFFLE)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 6. Create the preprocessing pipeline
preprocessor = preprocess.create_preprocessing_pipeline(numeric_features, categorical_features)

# 7. Create the full model pipeline
# This pipeline will:
#    a) Apply the preprocessor (scaling, one-hot-encoding)
#    b) Feed the processed data into the model (LinearRegression)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# 8. Fit and Evaluate
model_pipeline.fit(X_train, y_train)
score = model_pipeline.score(X_test, y_test)
print(f"Model R-squared: {score}")

"""
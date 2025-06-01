import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def remove_univariate_outliers(df_train, df_test, column_configs):
    '''
    Remove outliers based on column-specific rules
    '''
    
    df_train_clean = df_train.copy()
    df_test_clean = df_test.copy()
    
    for config in column_configs:
        col = config['col']
        if col not in df_train_clean.columns or col not in df_test_clean.columns:
            continue

        current_series_train = df_train_clean[col]
        
        lower_b, upper_b = -np.inf, np.inf

        if 'min_val' in config:
            lower_b = max(lower_b, config['min_val'])
        if 'min_quant' in config:
            if not current_series_train.empty and current_series_train.notna().any():
                 lower_b = max(lower_b, current_series_train.quantile(config['min_quant']))


        if 'max_val' in config:
            upper_b = min(upper_b, config['max_val'])
        if 'max_quant' in config:
            if not current_series_train.empty and current_series_train.notna().any():
                upper_b = min(upper_b, current_series_train.quantile(config['max_quant']))

        initial_rows_train = len(df_train_clean)
        df_train_clean = df_train_clean[(df_train_clean[col] >= lower_b) & (df_train_clean[col] <= upper_b)]
        removed_train = initial_rows_train - len(df_train_clean)
        if removed_train > 0:
            print(f'TRAIN \'{col}\': removed {removed_train} rows')

        initial_rows_test = len(df_test_clean)
        df_test_clean = df_test_clean[(df_test_clean[col] >= lower_b) & (df_test_clean[col] <= upper_b)]
        removed_test = initial_rows_test - len(df_test_clean)
        if removed_test > 0:
            print(f'TEST \'{col}\': removed {removed_test} rows using bounds [{lower_b:.2f}, {upper_b:.2f}]')
            
    return df_train_clean, df_test_clean



def remove_price_discrepancy_outliers(df_train, df_test, target_col, area_col, actual_worth_col, iqr_multiplier=3):
    '''
    Remove rows where (target * area) differs significantly from actual_worth.
    Uses classic IQR method with custom multiplier.
    Thresholds calculated on train_df and applied to both datasets.
    '''

    df_train_clean = df_train.copy()
    df_test_clean = df_test.copy()

    train_calculated_worth = df_train_clean[target_col] * df_train_clean[area_col]
    train_price_diff = train_calculated_worth - df_train_clean[actual_worth_col]

    Q1_diff = train_price_diff.quantile(0.25)
    Q3_diff = train_price_diff.quantile(0.75)
    IQR_diff = Q3_diff - Q1_diff

    lower_bound_diff = Q1_diff - iqr_multiplier * IQR_diff
    upper_bound_diff = Q3_diff + iqr_multiplier * IQR_diff

    initial_rows_train = len(df_train_clean)
    train_filter_mask = (train_price_diff >= lower_bound_diff) & (train_price_diff <= upper_bound_diff)
    df_train_clean = df_train_clean[train_filter_mask]
    removed_train = initial_rows_train - len(df_train_clean)
    if removed_train > 0:
        print(f'TRAIN \'price_diff\': removed {removed_train} rows')

    # Calculate for test and apply train boundaries
    test_calculated_worth = df_test_clean[target_col] * df_test_clean[area_col]
    test_price_diff = test_calculated_worth - df_test_clean[actual_worth_col]
    
    initial_rows_test = len(df_test_clean)
    test_filter_mask = (test_price_diff >= lower_bound_diff) & (test_price_diff <= upper_bound_diff)
    df_test_clean = df_test_clean[test_filter_mask]
    removed_test = initial_rows_test - len(df_test_clean)
    if removed_test > 0:
         print(f'TEST \'price_diff\': removed {removed_test} rows using train bounds [{lower_bound_diff:.2f}, {upper_bound_diff:.2f}]')

    return df_train_clean, df_test_clean


def remove_feature_outliers_isoforest(X_train, X_test, y_train, y_test, 
                                     numeric_feature_cols, contamination=0.01, random_state=42):
    '''
    Remove outliers using Isolation Forest
    '''

    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()
    y_train_clean = y_train.copy()
    y_test_clean = y_test.copy()
    
    X_train_numeric = X_train_clean[numeric_feature_cols].copy()
    train_medians = X_train_numeric.median() 
    X_train_numeric_imputed = X_train_numeric.fillna(train_medians)

    X_test_numeric = X_test_clean[numeric_feature_cols].copy()
    X_test_numeric_imputed = X_test_numeric.fillna(train_medians)  # Apply train medians

    # Train model on train data
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    iso_forest.fit(X_train_numeric_imputed)

    # Predict outliers (-1) and normal points (1)
    train_outlier_preds = iso_forest.predict(X_train_numeric_imputed)
    test_outlier_preds = iso_forest.predict(X_test_numeric_imputed)

    # Filter train data
    train_inlier_mask = train_outlier_preds == 1
    initial_rows_train = len(X_train_clean)
    X_train_clean = X_train_clean[train_inlier_mask]
    y_train_clean = y_train_clean[train_inlier_mask]
    removed_train = initial_rows_train - len(X_train_clean)
    if removed_train > 0:
        print(f'TRAIN (Isolation Forest): removed {removed_train} outlier rows')

    # Filter test data
    test_inlier_mask = test_outlier_preds == 1
    initial_rows_test = len(X_test_clean)
    X_test_clean = X_test_clean[test_inlier_mask]
    y_test_clean = y_test_clean[test_inlier_mask]
    removed_test = initial_rows_test - len(X_test_clean)
    if removed_test > 0:
        print(f'TEST (Isolation Forest): removed {removed_test} outlier rows')
        
    return X_train_clean, X_test_clean, y_train_clean, y_test_clean
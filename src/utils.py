import pandas as pd
import matplotlib.pyplot as plt


def detect_outliers_iqr(df: pd.DataFrame,
                        column_name: pd.Series,
                        weight: int) -> list[int]:
    """
    Detecting anomalies using IQR
    """

    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - weight * iqr
    upper_bound = q3 + weight * iqr

    print(f"q1: {q1:.2f}, q3: {q3:.2f}, iqr: {iqr:.2f}")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")

    outliers_indices = df.index[(df[column_name] < lower_bound) | (
        df[column_name] > upper_bound)]
    print(f"Found {len(outliers_indices)} anomalies.")

    return outliers_indices


def calculate_correlations(df: pd.DataFrame,
                           target_col: str,
                           top_n: int = 10):
    """
    Computes Pearson correlation
    """

    numeric_df = df.select_dtypes(include=['number'])
    correlations = numeric_df.corr()[target_col].sort_values(ascending=False)
    correlations = correlations.drop(target_col)

    print("Pearson correlation")
    print('\nMost powerful correlations')
    print(correlations.head(top_n))


def analyze_column(df: pd.DataFrame,
                   col: str):
    """
    Analyze a column in DataFrame
    """

    if col not in df.columns:
        print(f"Column '{col}' not found in DataFrame.")
        return

    print(f"{col}")

    print("\nDescriptive statistics:")
    print(df[col].describe())

    print(f"\n Number of missing values: {df[col].isna().sum()}")

    print("\nUnique values and its frequency:")
    print(df[col].value_counts(dropna=False))

    plt.figure(figsize=(10, 6))
    plt.hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
    plt.title(f"Column distribution: {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.show()

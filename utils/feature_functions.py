"""
Feature engineering functions for technical indicators and aggregations.
"""

import pandas as pd
import numpy as np


def calculate_returns(df: pd.DataFrame, price_col: str = 'close', periods: list = [1]) -> pd.DataFrame:
    """
    Calculate returns for specified periods.

    Args:
        df: DataFrame with price data
        price_col: Name of price column
        periods: List of periods for return calculation

    Returns:
        DataFrame with return columns added
    """
    result = df.copy()

    for period in periods:
        result[f'return_{period}d'] = result[price_col].pct_change(period)

    return result


def calculate_rolling_volatility(df: pd.DataFrame, price_col: str = 'close', windows: list = [5, 10, 20]) -> pd.DataFrame:
    """
    Calculate rolling volatility (std of returns).

    Args:
        df: DataFrame with price data
        price_col: Name of price column
        windows: List of window sizes

    Returns:
        DataFrame with volatility columns added
    """
    result = df.copy()
    returns = result[price_col].pct_change()

    for window in windows:
        result[f'volatility_{window}d'] = returns.rolling(window).std()

    return result


def calculate_rsi(df: pd.DataFrame, price_col: str = 'close', period: int = 14) -> pd.DataFrame:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        df: DataFrame with price data
        price_col: Name of price column
        period: RSI period (default 14)

    Returns:
        DataFrame with RSI column added
    """
    result = df.copy()

    delta = result[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    result[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    return result


def calculate_ma_ratios(df: pd.DataFrame, price_col: str = 'close', periods: list = [10, 20, 50]) -> pd.DataFrame:
    """
    Calculate moving average ratios (price / MA).

    Args:
        df: DataFrame with price data
        price_col: Name of price column
        periods: List of MA periods

    Returns:
        DataFrame with MA ratio columns added
    """
    result = df.copy()

    for period in periods:
        ma = result[price_col].rolling(window=period).mean()
        result[f'ma_ratio_{period}'] = result[price_col] / ma

    return result


def calculate_rolling_correlation(df1: pd.DataFrame, df2: pd.DataFrame,
                                   col1: str, col2: str, windows: list = [20, 60]) -> pd.DataFrame:
    """
    Calculate rolling correlation between two series.

    Args:
        df1: First DataFrame
        df2: Second DataFrame
        col1: Column name from df1
        col2: Column name from df2
        windows: List of window sizes

    Returns:
        DataFrame with correlation columns
    """
    # Merge on date
    merged = df1[['date', col1]].merge(df2[['date', col2]], on='date', how='inner')

    result = merged[['date']].copy()

    for window in windows:
        result[f'corr_{window}d'] = merged[col1].rolling(window).corr(merged[col2])

    return result


def forward_fill_macro(df: pd.DataFrame, value_col: str, date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Forward fill macro data (like CPI) to create daily values.

    Args:
        df: DataFrame with macro data (may have missing dates)
        value_col: Name of value column
        date_range: Complete date range to fill

    Returns:
        DataFrame with forward-filled daily values
    """
    # Create complete date range
    complete_df = pd.DataFrame({'date': date_range})

    # Merge and forward fill
    merged = complete_df.merge(df[['date', value_col]], on='date', how='left')
    merged[value_col] = merged[value_col].fillna(method='ffill')

    return merged


def aggregate_sentiment(sentiment_df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Aggregate article-level sentiment to daily sentiment metrics.

    Args:
        sentiment_df: DataFrame with article-level sentiment scores
        date_col: Name of date column

    Returns:
        DataFrame with daily aggregated sentiment
    """
    daily_sentiment = sentiment_df.groupby(date_col).agg({
        'compound': ['mean', 'std', 'count'],
        'positive': 'mean',
        'negative': 'mean',
        'neutral': 'mean'
    }).reset_index()

    # Flatten column names
    daily_sentiment.columns = [
        date_col,
        'sentiment_mean', 'sentiment_std', 'article_count',
        'positive_mean', 'negative_mean', 'neutral_mean'
    ]

    # Fill NaN std with 0 (when only 1 article)
    daily_sentiment['sentiment_std'] = daily_sentiment['sentiment_std'].fillna(0)

    return daily_sentiment

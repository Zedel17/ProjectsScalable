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


def make_macro_daily_features(
    trading_dates: pd.DatetimeIndex,
    dgs10_series: pd.DataFrame,
    cpi_series: pd.DataFrame,
    method: str = "fixed_release"
) -> pd.DataFrame:
    """
    Create daily macro features with point-in-time correctness to avoid look-ahead bias.

    This function ensures that features for date t only use information available on or before t:
    - DGS10: Daily treasury yield, forward-filled from past values
    - CPI: Monthly series aligned to release dates (NOT reference month dates)

    Args:
        trading_dates: DatetimeIndex of trading days (e.g., from QQQ data)
        dgs10_series: DataFrame with columns [date, dgs10] from FRED
        cpi_series: DataFrame with columns [date, cpiaucsl] from FRED
        method: CPI release date handling method:
            - "fixed_release": Assume CPI for month M is released on 15th of month M+1
            - "alfred": Use ALFRED realtime data (not yet implemented)

    Returns:
        DataFrame with columns:
            - date: Trading date
            - dgs10: 10-year yield (forward-filled)
            - dgs10_chg_1d: 1-day yield change
            - dgs10_chg_5d: 5-day yield change
            - dgs10_chg_20d: 20-day yield change
            - cpi_level_asof: CPI level as-of this date (point-in-time correct)
            - cpi_yoy_asof: CPI year-over-year change (point-in-time correct)

    Point-in-time correctness guarantee:
        For any date t in the output, all feature values reflect only information
        that would have been available to a trader on date t.

    Example:
        If CPI for January 2024 is released on Feb 15, 2024, then:
        - cpi_level_asof on Feb 14, 2024 = December 2023 CPI
        - cpi_level_asof on Feb 15, 2024 = January 2024 CPI
    """
    # Create base DataFrame with trading dates
    result = pd.DataFrame({'date': trading_dates})
    result = result.sort_values('date').reset_index(drop=True)

    # === DGS10: Forward-fill from past values (no leakage) ===
    dgs10_df = dgs10_series.copy()
    dgs10_df['date'] = pd.to_datetime(dgs10_df['date'])

    # Merge and forward-fill
    result = result.merge(dgs10_df[['date', 'dgs10']], on='date', how='left')
    result['dgs10'] = result['dgs10'].fillna(method='ffill')

    # Calculate DGS10 changes
    result['dgs10_chg_1d'] = result['dgs10'].diff(1)
    result['dgs10_chg_5d'] = result['dgs10'].diff(5)
    result['dgs10_chg_20d'] = result['dgs10'].diff(20)

    # === CPI: Align to release dates (point-in-time correct) ===
    if method == "fixed_release":
        # ASSUMPTION: CPI for month M is released on the 15th day of month M+1
        # This is documented approximation (actual release dates vary slightly)
        # See: https://www.bls.gov/schedule/news_release/cpi.htm

        cpi_df = cpi_series.copy()
        cpi_df['date'] = pd.to_datetime(cpi_df['date'])

        # CPI 'date' from FRED is the reference month (e.g., 2024-01-01 for January 2024)
        # Calculate release date: 15th of next month
        cpi_df['release_date'] = cpi_df['date'] + pd.DateOffset(months=1, day=15)

        # Align to next business day if 15th falls on weekend
        # (In practice, BLS releases are always on weekdays)
        cpi_df['release_date'] = cpi_df['release_date'].apply(
            lambda x: x + pd.offsets.BDay(0) if x.weekday() >= 5 else x
        )

        # Create CPI as-of mapping: for each trading date, what CPI value is available?
        # Use merge_asof to find the most recent CPI release as of each date
        cpi_asof = cpi_df[['release_date', 'cpiaucsl']].copy()
        cpi_asof = cpi_asof.rename(columns={'release_date': 'date'})
        cpi_asof = cpi_asof.sort_values('date')

        result = pd.merge_asof(
            result.sort_values('date'),
            cpi_asof,
            on='date',
            direction='backward'
        )
        result = result.rename(columns={'cpiaucsl': 'cpi_level_asof'})

        # Calculate YoY change (12 months = ~252 trading days)
        # We need to use the CPI from 12 months ago AS OF that point in time
        # This requires reconstructing what CPI was known 252 days ago
        result['cpi_level_12m_ago'] = result['cpi_level_asof'].shift(252)
        result['cpi_yoy_asof'] = (
            (result['cpi_level_asof'] / result['cpi_level_12m_ago'] - 1) * 100
        )

        # Drop intermediate column
        result = result.drop(columns=['cpi_level_12m_ago'])

    elif method == "alfred":
        raise NotImplementedError(
            "ALFRED realtime CPI not yet implemented. Use method='fixed_release'."
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'fixed_release' or 'alfred'.")

    return result

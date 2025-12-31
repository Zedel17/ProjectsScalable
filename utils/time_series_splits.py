"""
Time series split utilities for proper backtesting without look-ahead bias.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Iterator


def train_test_split_by_date(
    df: pd.DataFrame,
    split_date: str,
    date_col: str = 'date',
    purge_gap: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/test by a fixed date with a purge gap.

    The purge gap ensures no data leakage between train and test sets.
    For example, with purge_gap=1:
    - If split_date is 2024-01-15
    - Train set: all dates < 2024-01-14 (excluding the purge day)
    - Test set: all dates >= 2024-01-15

    Args:
        df: DataFrame with time series data
        split_date: Date to split on (YYYY-MM-DD format). Test set starts here.
        date_col: Name of date column
        purge_gap: Number of days to exclude before split_date (default 1)

    Returns:
        (train_df, test_df): Tuple of training and test DataFrames

    Example:
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2024-01-01', '2024-01-20'),
        ...     'value': range(20)
        ... })
        >>> train, test = train_test_split_by_date(df, '2024-01-15', purge_gap=1)
        >>> train['date'].max()  # Last train date is Jan 13 (1 day gap before Jan 15)
        >>> test['date'].min()   # First test date is Jan 15
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    split_dt = pd.to_datetime(split_date)

    # Calculate purge date (split_date - purge_gap days)
    purge_dt = split_dt - pd.Timedelta(days=purge_gap)

    # Train: all dates before purge_dt
    train_df = df[df[date_col] < purge_dt].copy()

    # Test: all dates >= split_dt
    test_df = df[df[date_col] >= split_dt].copy()

    return train_df, test_df


def walk_forward_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_size: int = 60,
    purge_gap: int = 1,
    date_col: str = 'date'
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate walk-forward train/test splits for time series cross-validation.

    This creates expanding windows where:
    - Each split has an expanding training set
    - Each split has a fixed-size test set
    - A purge gap separates train and test to prevent leakage

    Args:
        df: DataFrame with time series data
        n_splits: Number of splits to generate
        test_size: Number of days in each test set
        purge_gap: Number of days to exclude between train and test (default 1)
        date_col: Name of date column

    Yields:
        (train_df, test_df): Tuples of training and test DataFrames

    Example:
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2020-01-01', '2024-12-31'),
        ...     'value': range(len(pd.date_range('2020-01-01', '2024-12-31')))
        ... })
        >>> for i, (train, test) in enumerate(walk_forward_split(df, n_splits=3)):
        ...     print(f"Split {i}: Train {train.shape[0]} days, Test {test.shape[0]} days")
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Get unique dates
    dates = df[date_col].unique()
    dates = pd.to_datetime(dates)
    dates = np.sort(dates)

    n_dates = len(dates)

    # Calculate total size needed for all splits
    min_train_size = test_size * 2  # Minimum reasonable train size
    total_needed = min_train_size + (test_size + purge_gap) * n_splits

    if n_dates < total_needed:
        raise ValueError(
            f"Not enough data for {n_splits} splits. "
            f"Need {total_needed} days, have {n_dates} days."
        )

    # Calculate split points
    # We work backwards from the end to ensure we have enough test data
    split_dates = []
    for i in range(n_splits):
        # Start from the end and work backwards
        test_end_idx = n_dates - i * (test_size + purge_gap)
        test_start_idx = test_end_idx - test_size

        if test_start_idx < min_train_size:
            raise ValueError(f"Not enough data for {n_splits} splits")

        split_dates.append({
            'test_start': dates[test_start_idx],
            'test_end': dates[test_end_idx - 1]
        })

    # Reverse to get chronological order
    split_dates = list(reversed(split_dates))

    # Generate splits
    for split_info in split_dates:
        test_start = split_info['test_start']
        purge_dt = test_start - pd.Timedelta(days=purge_gap)

        # Train: all dates before purge_dt
        train_df = df[df[date_col] < purge_dt].copy()

        # Test: dates in test window
        test_df = df[
            (df[date_col] >= test_start) &
            (df[date_col] <= split_info['test_end'])
        ].copy()

        yield train_df, test_df


def get_train_val_test_split(
    df: pd.DataFrame,
    train_end_date: str,
    val_end_date: str,
    date_col: str = 'date',
    purge_gap: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation/test sets by dates with purge gaps.

    Args:
        df: DataFrame with time series data
        train_end_date: Last date of training set (exclusive, with purge gap)
        val_end_date: Last date of validation set (exclusive, with purge gap)
        date_col: Name of date column
        purge_gap: Number of days to exclude between sets

    Returns:
        (train_df, val_df, test_df): Tuple of train, validation, and test DataFrames

    Example:
        >>> train, val, test = get_train_val_test_split(
        ...     df,
        ...     train_end_date='2023-01-01',
        ...     val_end_date='2024-01-01',
        ...     purge_gap=1
        ... )
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    train_end_dt = pd.to_datetime(train_end_date)
    val_end_dt = pd.to_datetime(val_end_date)

    # Train set: everything before train_end_dt - purge_gap
    train_purge_dt = train_end_dt - pd.Timedelta(days=purge_gap)
    train_df = df[df[date_col] < train_purge_dt].copy()

    # Validation set: from train_end_dt to val_end_dt - purge_gap
    val_purge_dt = val_end_dt - pd.Timedelta(days=purge_gap)
    val_df = df[
        (df[date_col] >= train_end_dt) &
        (df[date_col] < val_purge_dt)
    ].copy()

    # Test set: from val_end_dt onwards
    test_df = df[df[date_col] >= val_end_dt].copy()

    return train_df, val_df, test_df

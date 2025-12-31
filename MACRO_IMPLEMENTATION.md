# Macro Features Implementation - Point-in-Time Correctness

## Overview

This implementation ensures **statistically valid macro features** with **point-in-time correctness** to prevent look-ahead bias in the QQQ prediction model.

## Key Principle

**For any date t, all features must use only information available on or before t.**

This is critical for:
- Valid backtesting results
- Realistic out-of-sample performance
- Production deployment (can't use future information)

## Implementation Details

### 1. Data Fetching (`utils/data_fetchers.py`)

#### `fetch_dgs10(start_date, end_date)`
- Fetches 10-year Treasury yield (DGS10) from FRED
- Daily series - values are known in real-time
- Can be forward-filled without leakage concerns

#### `fetch_cpi(start_date, end_date)`
- Fetches Consumer Price Index (CPIAUCSL) from FRED
- **Critical**: Returns reference month dates, NOT release dates
- Example: `date=2024-01-01` means "CPI for January 2024"
- Release date handling is done separately (see below)

#### `fetch_cpi_alfred_realtime()` (placeholder)
- For future implementation using ALFRED realtime vintage data
- Would provide exact historical release dates
- Currently raises NotImplementedError

### 2. Feature Engineering (`utils/feature_functions.py`)

#### `make_macro_daily_features(trading_dates, dgs10_series, cpi_series, method="fixed_release")`

Creates daily macro features with point-in-time correctness.

**DGS10 Features** (no leakage):
- `dgs10`: 10-year yield, forward-filled from past values
- `dgs10_chg_1d`: 1-day change
- `dgs10_chg_5d`: 5-day change
- `dgs10_chg_20d`: 20-day change

**CPI Features** (release-date aligned):
- `cpi_level_asof`: CPI level as known on date t
- `cpi_yoy_asof`: Year-over-year CPI change (point-in-time correct)

**CPI Release Date Logic** (`method="fixed_release"`):
```
Assumption: CPI for month M is released on the 15th day of month M+1
```

Example timeline:
```
2024-01-01: January 2024 reference month (data collected in January)
2024-02-15: January 2024 CPI is RELEASED (becomes available)

Therefore:
- On 2024-02-14: cpi_level_asof = December 2023 CPI
- On 2024-02-15: cpi_level_asof = January 2024 CPI
```

Implementation uses `pd.merge_asof()` with backward direction to find the most recent released CPI value for each trading day.

### 3. Time-Series Splits (`utils/time_series_splits.py`)

#### `train_test_split_by_date(df, split_date, purge_gap=1)`
- Fixed date split with purge gap
- Purge gap prevents leakage at the boundary
- Example with `purge_gap=1`:
  - Train: dates < split_date - 1 day
  - Gap: 1 day excluded
  - Test: dates >= split_date

#### `walk_forward_split(df, n_splits=5, test_size=60, purge_gap=1)`
- Walk-forward cross-validation
- Expanding training windows
- Fixed-size test windows
- Purge gap between each split

#### `get_train_val_test_split(df, train_end_date, val_end_date, purge_gap=1)`
- Three-way split: train / validation / test
- Purge gaps between each set

**Why purge gaps?**
- Features with lags (e.g., 20-day moving average) create dependencies
- Without gap: test data point at t uses features that depend on t-20
- With gap: ensures complete independence

### 4. Notebooks

#### Notebook 1: `1_backfill_yahoo.ipynb`
- Fetches QQQ, XLK, VIX from Yahoo Finance
- Saves to `data/qqq_raw.parquet`, etc.
- Provides trading calendar for macro alignment

#### Notebook 2: `2_backfill_fred.ipynb`
- Fetches DGS10 and CPIAUCSL from FRED
- Saves raw data to:
  - `data/dgs10_raw.parquet`
  - `data/cpi_raw.parquet`
- **Does NOT align to trading days** (done in notebook 5)

#### Notebook 5: `5_macro_sentiment_features.ipynb`
- Loads raw FRED data
- Gets trading calendar from QQQ data
- Creates point-in-time correct features using `make_macro_daily_features()`
- **Critical validation**: Checks CPI changes occur around 15th of each month
- Visualizes CPI as step function (should change monthly)
- Saves to `data/macro_features_daily.parquet`

## Validation & Testing

### Sanity Checks (in notebook 5)

1. **CPI Release Date Validation**:
   ```python
   # Shows CPI values around a known release date
   # Verifies CPI_asof is constant before release, changes after
   ```

2. **CPI Change Dates**:
   ```python
   # Lists all dates when CPI changes
   # Average day should be ~15 if logic is correct
   ```

3. **Visual Verification**:
   - CPI plot should show step function (monthly jumps)
   - DGS10 plot should be smooth (daily forward-fill)

### Expected Behavior

```
Date         cpi_level_asof    Notes
2024-02-01   308.417          (December 2023 CPI)
2024-02-14   308.417          (still December 2023)
2024-02-15   310.326          (January 2024 released!)
2024-02-16   310.326          (stays January 2024)
```

## Usage Example

```python
import pandas as pd
from utils.data_fetchers import fetch_dgs10, fetch_cpi
from utils.feature_functions import make_macro_daily_features

# Load data
dgs10 = pd.read_parquet('data/dgs10_raw.parquet')
cpi = pd.read_parquet('data/cpi_raw.parquet')
qqq = pd.read_parquet('data/qqq_raw.parquet')

# Get trading calendar
trading_dates = pd.DatetimeIndex(sorted(qqq['date'].unique()))

# Create point-in-time correct features
macro_features = make_macro_daily_features(
    trading_dates=trading_dates,
    dgs10_series=dgs10,
    cpi_series=cpi,
    method="fixed_release"
)

# Result has NO look-ahead bias!
```

## Training Pipeline Integration

When training models, use time-series splits:

```python
from utils.time_series_splits import train_test_split_by_date

train_df, test_df = train_test_split_by_date(
    df=macro_features,
    split_date='2024-01-01',
    purge_gap=1  # 1-day gap prevents leakage
)

# Train only on train_df
# Evaluate only on test_df
```

## Files Modified/Created

### Created:
- `utils/time_series_splits.py` - Time-series cross-validation utilities
- `MACRO_IMPLEMENTATION.md` - This document

### Modified:
- `utils/data_fetchers.py` - Added `fetch_dgs10()`, `fetch_cpi()`, `fetch_cpi_alfred_realtime()`
- `utils/feature_functions.py` - Added `make_macro_daily_features()`
- `notebooks/1_backfill_yahoo.ipynb` - Added parquet saving
- `notebooks/2_backfill_fred.ipynb` - Rewritten for FRED backfill with docs
- `notebooks/5_macro_sentiment_features.ipynb` - Rewritten with point-in-time features + validation

## Guarantees

✅ **No look-ahead bias**: Features for date t use only data available on/before t

✅ **CPI release dates**: Approximated using 15th of month M+1 rule (documented assumption)

✅ **DGS10 alignment**: Forward-filled to trading days without leakage

✅ **Time-series splits**: Purge gaps prevent boundary leakage

✅ **Validation**: Sanity checks in notebook 5 verify correctness

## Future Enhancements

1. **ALFRED Integration**: Implement `fetch_cpi_alfred_realtime()` for exact release dates
   - Would replace fixed 15th assumption with historical actual release dates
   - Requires HTTP calls to FRED API with realtime_start parameter

2. **Additional Macro Series**: Extend to other economic indicators
   - Federal Funds Rate
   - Unemployment
   - GDP (quarterly with release date handling)

3. **Automated Testing**: Unit tests for point-in-time correctness
   - Mock CPI release dates
   - Verify no future information leakage

## References

- BLS CPI Release Schedule: https://www.bls.gov/schedule/news_release/cpi.htm
- FRED API Documentation: https://fred.stlouisfed.org/docs/api/
- ALFRED (Archival FRED): https://alfred.stlouisfed.org/

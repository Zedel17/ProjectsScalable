# Hopsworks Setup Guide

## Full Hopsworks Pipeline (Pattern 1)

This project uses **Full Hopsworks** architecture where all data flows through Hopsworks Feature Store.

```
┌─────────────────────────────────────┐
│ External APIs                        │
│  - Yahoo Finance (QQQ, XLK, VIX)    │
│  - FRED (DGS10, CPI)                │
│  - NewsAPI + FinBERT                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Notebooks 1-3: Data Backfill        │
│  → Fetch raw data from APIs         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Hopsworks Feature Groups (Raw)      │
│  - qqq_raw                          │
│  - xlk_raw                          │
│  - vix_raw                          │
│  - dgs10_raw                        │
│  - cpi_raw                          │
│  - news_sentiment_raw               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Notebooks 4-5: Feature Engineering  │
│  → Read from Hopsworks FGs          │
│  → Apply transformations            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Hopsworks Feature Groups (Eng.)    │
│  - qqq_technical_features           │
│  - xlk_sector_features              │
│  - vix_volatility_features          │
│  - macro_features (point-in-time!)  │
│  - sentiment_features               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Notebook 6: Create Feature View     │
│  → Join all engineered FGs          │
│  → Add targets (next-day return)    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Hopsworks Feature View              │
│  → qqq_prediction_fv                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Notebook 7: Model Training          │
│  → Read from Feature View           │
│  → Train XGBoost (regression + clf) │
│  → Upload to Model Registry         │
└─────────────────────────────────────┘
```

## Environment Variables Required

### Minimal Setup (Notebooks 1-2 Only)

Add to your `.env`:

```bash
# FRED API (for notebook 2)
FRED_API_KEY=your_fred_api_key_here

# Hopsworks (for all notebooks)
HOPSWORKS_API_KEY=your_hopsworks_api_key_here
HOPSWORKS_PROJECT_NAME=qqq_prediction
```

### Full Setup (All Features)

```bash
# Hopsworks
HOPSWORKS_API_KEY=your_hopsworks_api_key_here
HOPSWORKS_PROJECT_NAME=qqq_prediction

# FRED API
FRED_API_KEY=your_fred_api_key_here

# NewsAPI (for notebook 3)
NEWS_API_KEY=your_news_api_key_here

# Data range configuration
START_DATE=2023-01-01
END_DATE=2025-12-30
```

## Getting API Keys

### 1. Hopsworks (Required)

1. Go to: https://app.hopsworks.ai/
2. Sign up (free tier)
3. Create new project: `qqq_prediction`
4. Settings → API Keys → Generate New API Key
5. Copy key to `.env`

### 2. FRED API (Required for Notebook 2)

1. Go to: https://fred.stlouisfed.org/docs/api/api_key.html
2. Request API key (free, instant)
3. Copy key to `.env`

### 3. NewsAPI (Optional - for Notebook 3)

1. Go to: https://newsapi.org/register
2. Get free key (100 requests/day)
3. Copy key to `.env`

## No Pre-Downloaded Data Needed

✅ All data is fetched automatically by the notebooks:
- Notebook 1: Fetches from Yahoo Finance (no key needed)
- Notebook 2: Fetches from FRED (needs FRED_API_KEY)
- Notebook 3: Fetches from NewsAPI (needs NEWS_API_KEY)

## Running the Pipeline

### Step 1: Setup Environment

```bash
# Activate virtual environment
source venv/bin/activate

# Ensure all packages installed
pip install -r requirements.txt

# Copy and edit .env file
cp .env.example .env
nano .env  # Add your API keys
```

### Step 2: Run Backfill Notebooks

Run in order (each uploads to Hopsworks):

```bash
jupyter notebook notebooks/1_backfill_yahoo.ipynb
```
- Fetches QQQ, XLK, VIX from Yahoo Finance
- Creates Hopsworks FGs: `qqq_raw`, `xlk_raw`, `vix_raw`

```bash
jupyter notebook notebooks/2_backfill_fred.ipynb
```
- Fetches DGS10, CPI from FRED
- Creates Hopsworks FGs: `dgs10_raw`, `cpi_raw`

(Optional)
```bash
jupyter notebook notebooks/3_backfill_news.ipynb
```
- Fetches news + applies FinBERT sentiment
- Creates Hopsworks FG: `news_sentiment_raw`

### Step 3: Run Feature Engineering

```bash
jupyter notebook notebooks/5_macro_sentiment_features.ipynb
```
- Reads from Hopsworks: `dgs10_raw`, `cpi_raw`, `qqq_raw`
- Creates point-in-time correct features
- Creates Hopsworks FG: `macro_features`

```bash
jupyter notebook notebooks/4_market_features.ipynb
```
- Reads from Hopsworks: `qqq_raw`, `xlk_raw`, `vix_raw`
- Creates technical indicators
- Creates Hopsworks FGs: `qqq_technical_features`, `xlk_sector_features`, `vix_volatility_features`

### Step 4: Training (Later)

```bash
jupyter notebook notebooks/6_create_feature_view.ipynb
jupyter notebook notebooks/7_training.ipynb
```

## Hopsworks Feature Groups Created

### Raw Data (from notebooks 1-3):
- `qqq_raw` - QQQ OHLCV data
- `xlk_raw` - XLK OHLCV data
- `vix_raw` - VIX data
- `dgs10_raw` - Treasury yield (daily)
- `cpi_raw` - CPI (monthly, reference dates)
- `news_sentiment_raw` - Article-level sentiment

### Engineered Features (from notebooks 4-5):
- `qqq_technical_features` - Returns, volatility, RSI, MA ratios
- `xlk_sector_features` - Sector returns, correlation
- `vix_volatility_features` - VIX levels and changes
- `macro_features` - **Point-in-time correct DGS10 and CPI**
- `sentiment_features` - Daily aggregated sentiment

## Verification Checklist

After running notebooks 1-2, verify in Hopsworks UI:

- [ ] Feature Store shows `qqq_raw` with ~500 rows (2 years of trading days)
- [ ] Feature Store shows `xlk_raw` with ~500 rows
- [ ] Feature Store shows `vix_raw` with ~500 rows
- [ ] Feature Store shows `dgs10_raw` with ~500 rows
- [ ] Feature Store shows `cpi_raw` with ~24 rows (monthly data)

After running notebook 5:

- [ ] Feature Store shows `macro_features` with ~500 rows
- [ ] `macro_features` has columns: `date`, `dgs10`, `dgs10_chg_*`, `cpi_level_asof`, `cpi_yoy_asof`
- [ ] Validation passed: CPI changes around 15th of each month

## Troubleshooting

### Error: "HOPSWORKS_API_KEY not found"
- Check `.env` file exists in project root
- Verify key is correctly pasted
- Run `load_dotenv()` in notebook

### Error: "Feature group 'xxx_raw' not found"
- Run backfill notebooks first (1-3)
- Check Hopsworks UI to verify FGs exist
- Ensure you're using correct project name

### Error: "FRED API key invalid"
- Verify FRED_API_KEY in `.env`
- Test key at: https://fred.stlouisfed.org/

### CPI validation shows wrong dates
- This is expected if your data range doesn't include recent months
- Adjust validation window in notebook 5 to match your data range
- Key check: Average day of month should be ~15

## Benefits of Full Hopsworks Pattern

✅ **Single Source of Truth**: All data in Hopsworks Feature Store

✅ **Versioning**: Every feature group is versioned

✅ **Collaboration**: Team members use same features

✅ **Point-in-Time Joins**: Hopsworks handles time-travel queries

✅ **Lineage**: Track which features come from which data sources

✅ **Production Ready**: Same code for dev/staging/prod

## Next Steps

After completing notebooks 1-5:
1. Run notebook 4 for market features
2. Run notebook 6 to create feature view
3. Run notebook 7 to train models
4. Models will use time-series splits (no look-ahead bias!)

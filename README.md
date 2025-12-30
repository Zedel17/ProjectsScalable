# QQQ ETF Prediction using Machine Learning

Multi-factor prediction system for QQQ ETF next-day returns using financial data, macroeconomic indicators, and news sentiment analysis.

## Project Structure

```
ProjectScalable/
├── config/                          # Configuration files
│   ├── config.yaml                  # Main configuration
│   └── feature_config.yaml          # Feature engineering params
├── notebooks/                       # Jupyter notebooks (run in order)
│   ├── 1_backfill_yahoo.ipynb      # Fetch QQQ, XLK, VIX data
│   ├── 2_backfill_fred.ipynb       # Fetch Treasury yield, CPI
│   ├── 3_backfill_news.ipynb       # Fetch news + FinBERT sentiment
│   ├── 4_market_features.ipynb     # Create market features
│   ├── 5_macro_sentiment_features.ipynb  # Create macro/sentiment features
│   ├── 6_create_feature_view.ipynb # Join all features
│   ├── 7_training.ipynb            # Train XGBoost models
│   └── 8_daily_inference.ipynb     # Generate daily predictions
├── utils/                           # Reusable Python utilities
│   ├── data_fetchers.py            # API data fetching functions
│   ├── feature_functions.py        # Feature engineering functions
│   └── hopsworks_helpers.py        # Hopsworks integration
├── dashboard/                       # Gradio dashboard
│   └── app.py                      # Dashboard application
├── requirements.txt                 # Python dependencies
├── .env.example                    # Environment variables template
└── PROJECT_PLAN.md                 # Detailed implementation plan

```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Required API keys:
- **Hopsworks**: Create account at hopsworks.ai
- **NewsAPI**: Get free key at newsapi.org
- **FRED**: Get free key at fred.stlouisfed.org
- **Yahoo Finance**: No key needed (uses yfinance library)

### 3. Run Notebooks in Order

Execute the notebooks sequentially:

1. **Data Backfill** (can be run in parallel):
   - `1_backfill_yahoo.ipynb`
   - `2_backfill_fred.ipynb`
   - `3_backfill_news.ipynb`

2. **Feature Engineering** (can be run in parallel):
   - `4_market_features.ipynb`
   - `5_macro_sentiment_features.ipynb`

3. **Model Pipeline** (run sequentially):
   - `6_create_feature_view.ipynb`
   - `7_training.ipynb`
   - `8_daily_inference.ipynb`

## Features

### Data Sources
- **Yahoo Finance**: QQQ, XLK (tech sector), VIX (volatility)
- **FRED**: 10-year Treasury yield, CPI
- **NewsAPI + FinBERT**: Financial news sentiment

### Feature Categories
1. **QQQ Technical**: Returns, volatility, RSI, MA ratios
2. **XLK Sector**: Returns, correlation with QQQ
3. **VIX Volatility**: Close value, daily changes, rolling stats
4. **Macro Indicators**: Treasury yields, CPI
5. **News Sentiment**: FinBERT sentiment scores aggregated daily

### Models
- **Regression**: XGBoost predicting next-day return
- **Classification**: XGBoost predicting up/down direction

## Usage

### Daily Inference
Run `8_daily_inference.ipynb` to generate predictions for the next trading day.

### Dashboard
Launch the Gradio dashboard:
```bash
python dashboard/app.py
```

## Project Team
- Federico Mercurio
- Margherita Santarossa

## References
- Original project proposal: `Scalable_project_proposal.pdf`
- Detailed implementation plan: `PROJECT_PLAN.md`

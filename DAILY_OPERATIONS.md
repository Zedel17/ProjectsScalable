# Daily Operations Guide - QQQ Prediction Pipeline

This document describes the daily workflow for maintaining and operating the QQQ prediction system.

## Pipeline Overview

```
BACKFILL → FEATURE ENGINEERING → TRAINING → INFERENCE → DASHBOARD
(Daily)    (Daily)                (Weekly)   (Daily)     (Real-time)
```

---

## Daily Tasks (Every Trading Day)

### 1. Data Backfill (Morning - Before Market Opens)

Run these notebooks to fetch latest data:

```bash
# Activate environment
source venv/bin/activate
cd notebooks

# Run backfill notebooks (can be run in parallel)
jupyter nbconvert --to notebook --execute 1_backfill_yahoo.ipynb
jupyter nbconvert --to notebook --execute 2_backfill_fred.ipynb
jupyter nbconvert --to notebook --execute 3_backfill_news.ipynb
```

**What this does:**
- Fetches latest QQQ, XLK, VIX prices from Yahoo Finance
- Updates 10-year Treasury yields and CPI from FRED
- Retrieves and analyzes latest financial news with FinBERT sentiment

**Expected output:**
- New rows added to Hopsworks feature groups
- Date range should include yesterday's trading day

---

### 2. Feature Engineering (After Backfill)

Compute features from raw data:

```bash
# Run feature engineering notebooks (can be run in parallel)
jupyter nbconvert --to notebook --execute 4_market_features.ipynb
jupyter nbconvert --to notebook --execute 5_macro_sentiment_features.ipynb

# Then combine features
jupyter nbconvert --to notebook --execute 6_create_feature_view.ipynb
```

**What this does:**
- Calculates technical indicators (RSI, volatility, MA ratios, etc.)
- Computes rolling correlations and sector features
- Aggregates news sentiment scores
- Combines all features into `qqq_combined_features` feature group

**Expected output:**
- `qqq_combined_features` feature group updated with latest features
- Should have 33 raw features (30 after dropping sentiment aggregates in training/inference)

---

### 3. Daily Inference (After Feature Engineering)

Generate prediction for next trading day:

```bash
# Run inference notebook
jupyter nbconvert --to notebook --execute 8_daily_inference_FIXED.ipynb
```

**What this does:**
- Loads latest features from `qqq_combined_features`
- Validates data freshness (must be < 7 days old)
- Drops same features as training (sentiment_mean, sentiment_std, article_count)
- Generates predictions using trained models
- Saves prediction to `qqq_predictions` feature group

**Expected output:**
- Prediction for next trading day
- Data saved to Hopsworks for dashboard visualization
- Console output showing:
  - Latest feature date
  - Predicted return (%)
  - Predicted direction (UP/DOWN)
  - Probability scores

---

### 4. Dashboard Monitoring (Optional)

View predictions and model performance:

```bash
# Launch dashboard
python dashboard/app.py
```

Then navigate to `http://localhost:7860` in your browser.

**What to check:**
- Latest prediction is displayed
- Historical predictions vs actuals chart
- Feature importance makes sense
- No stale data warnings

---

## Weekly Tasks (Every Monday or After Major Market Events)

### Model Retraining

If model performance degrades or after significant market changes:

```bash
# Retrain models with latest data
jupyter nbconvert --to notebook --execute 7_training.ipynb
```

**When to retrain:**
- Weekly on Mondays (to capture latest patterns)
- After major market events (Fed announcements, crashes, etc.)
- If backtesting shows declining accuracy (< 52% directional accuracy)

**What this does:**
- Loads all historical data from `qqq_combined_features`
- Performs time-series train/val/test split with purge gaps
- Trains XGBoost regressor and classifier
- Saves new models to `models/` directory
- Updates model metadata

**Critical:** After retraining, run inference again to ensure new models work correctly.

---

## Monthly Tasks

### 1. Model Performance Evaluation

```bash
# Run backtest evaluation
jupyter nbconvert --to notebook --execute 9_backtest_evaluation.ipynb
```

**What to check:**
- Directional accuracy (should be > 52%)
- R² score (higher is better, but may be negative for difficult prediction tasks)
- Strategy vs buy & hold performance
- Feature importance stability

### 2. Data Quality Audit

**Check for:**
- Missing data gaps in Hopsworks feature groups
- Sentiment analysis working (article_count > 0)
- No data inconsistencies or outliers
- Feature distributions haven't shifted dramatically

---

## Troubleshooting

### Issue: "Data too old for inference" Warning

**Cause:** Latest data in Hopsworks is > 7 days old
**Solution:** Run backfill notebooks (steps 1-2) to update data

### Issue: Feature mismatch error in inference

**Cause:** Training and inference using different feature sets
**Solution:**
1. Check that `8_daily_inference_FIXED.ipynb` drops same features as training
2. Verify model was trained with current code
3. May need to retrain model

### Issue: Low prediction accuracy

**Cause:** Model drift, market regime change, or poor training data
**Solutions:**
1. Retrain model with latest data
2. Check if features are calculated correctly
3. Verify news sentiment is working (not all zeros)
4. Consider increasing training data window

### Issue: News sentiment all zeros

**Cause:** NewsAPI limit reached or network issues
**Solution:**
1. Check NewsAPI key is valid
2. Verify daily article limit not exceeded
3. Check `3_backfill_news.ipynb` execution logs for errors

---

## Data Freshness Rules

| Component | Freshness Requirement | Action if Stale |
|-----------|----------------------|-----------------|
| Market prices (Yahoo) | < 1 trading day | Run `1_backfill_yahoo.ipynb` |
| FRED data | < 7 days | Run `2_backfill_fred.ipynb` |
| News sentiment | < 1 trading day | Run `3_backfill_news.ipynb` |
| Features | < 1 trading day | Run notebooks 4-6 |
| Model | < 7 days since training | Run `7_training.ipynb` |

---

## Feature Consistency Checklist

Ensure these are identical across backfill → training → inference:

✅ **Features used:**
- 30 features total (after dropping 3 sentiment aggregates)
- Dropped features: `sentiment_mean`, `sentiment_std`, `article_count`
- Kept features: `positive_mean`, `negative_mean`, `neutral_mean`

✅ **Data source:**
- All read from `qqq_combined_features` feature group version 1

✅ **Feature engineering:**
- Same calculation methods (RSI, volatility windows, etc.)
- Same lag periods (1, 2, 3, 5 days)
- Same rolling windows (5, 10, 20, 60 days)

---

## Quick Start: Complete Daily Workflow

```bash
#!/bin/bash
# daily_update.sh - Run this every trading day

# 1. Activate environment
source venv/bin/activate
cd notebooks

# 2. Backfill data
echo "Step 1/3: Backfilling data..."
jupyter nbconvert --to notebook --execute 1_backfill_yahoo.ipynb
jupyter nbconvert --to notebook --execute 2_backfill_fred.ipynb
jupyter nbconvert --to notebook --execute 3_backfill_news.ipynb

# 3. Feature engineering
echo "Step 2/3: Engineering features..."
jupyter nbconvert --to notebook --execute 4_market_features.ipynb
jupyter nbconvert --to notebook --execute 5_macro_sentiment_features.ipynb
jupyter nbconvert --to notebook --execute 6_create_feature_view.ipynb

# 4. Generate prediction
echo "Step 3/3: Generating prediction..."
jupyter nbconvert --to notebook --execute 8_daily_inference_FIXED.ipynb

echo "Daily update complete!"
```

**Usage:**
```bash
chmod +x daily_update.sh
./daily_update.sh
```

---

## API Keys Required

Ensure these are set in `.env`:

```bash
HOPSWORKS_API_KEY=your_key_here
HOPSWORKS_PROJECT_NAME=your_project_name
NEWS_API_KEY=your_newsapi_key
FRED_API_KEY=your_fred_key
```

---

## Next Steps

After mastering daily operations:
1. Set up automated scheduling (cron jobs or cloud scheduler)
2. Add email/Slack notifications for predictions
3. Implement automated model retraining triggers
4. Deploy dashboard to production (Hugging Face Spaces)

---

## Support & References

- **README.md**: Project overview and setup
- **HOPSWORKS_SETUP.md**: Hopsworks configuration guide
- **HUGGINGFACE_DEPLOYMENT.md**: Dashboard deployment guide
- **Scalable_project_proposal.pdf**: Original project specification

# Multi-Factor QQQ ETF Return Prediction System

**Authors:** Federico Mercurio & Margherita Santarossa
**Course:** Scalable Machine Learning and Deep Learning
**Date:** December 2024

## Overview

This project implements an end-to-end machine learning system for predicting next-day returns of the QQQ ETF (Invesco QQQ Trust), which tracks the NASDAQ-100 index. The system combines traditional market indicators with macroeconomic data and real-time news sentiment analysis to generate daily predictions.

Financial markets present unique challenges for machine learning applications. The non-stationary nature of financial time series, the risk of look-ahead bias, and the need for real-time predictions require careful architectural decisions. Our implementation addresses these challenges through a serverless feature store architecture, time-series aware training procedures, and automated daily pipelines.

## Motivation

The QQQ ETF is one of the most actively traded securities, representing the performance of major technology companies. Predicting its movements has practical applications for portfolio management, risk assessment, and trading strategies. However, traditional approaches often rely on limited feature sets or fail to properly handle temporal dependencies in financial data.

Our system integrates multiple data sources to capture different aspects of market behavior. Technical indicators reflect price momentum and volatility patterns. Macroeconomic variables like Treasury yields and inflation provide context about the broader economic environment. News sentiment captures market psychology and unexpected events that traditional indicators might miss.

## Architecture

The system follows a feature store architecture using Hopsworks as the central repository for all features and predictions. This approach provides several advantages over traditional batch processing pipelines: features are versioned and reproducible, point-in-time correctness is enforced to prevent look-ahead bias, and feature reuse across different models becomes straightforward.

### Data Pipeline

Data flows through a multi-stage pipeline from raw sources to predictions:

**Stage 1: Data Ingestion**
Three parallel backfill processes collect data from external APIs. Yahoo Finance provides OHLCV data for QQQ, XLK (technology sector ETF), and VIX (volatility index). The FRED API supplies macroeconomic indicators including 10-year Treasury yields and Consumer Price Index. NewsAPI retrieves financial news articles that are then processed through FinBERT for sentiment analysis.

**Stage 2: Feature Engineering**
Raw data transforms into features through two parallel processes. Market features include returns over multiple time horizons, volatility measures, technical indicators like RSI, moving average ratios, and correlation metrics between QQQ and sector performance. Macro and sentiment features aggregate economic indicators and news sentiment scores on a daily basis.

**Stage 3: Feature Store**
All features merge into a combined feature group in Hopsworks. This serves as the single source of truth for both training and inference, ensuring consistency across the pipeline.

**Stage 4: Model Training**
XGBoost models train on historical features with a time-series aware train/validation/test split. We implement purge gaps between splits to prevent information leakage from overlapping time windows. The system produces two models: a regression model for predicting exact return values and a classification model for predicting directional movement.

**Stage 5: Daily Inference**
The inference pipeline runs after market close each trading day. It loads the latest features, generates predictions using both models, and saves results back to Hopsworks for dashboard visualization.

## Technical Implementation

### Feature Engineering

The system generates 30 features across five categories:

QQQ technical indicators capture price momentum through returns calculated over 1, 2, 3, and 5-day windows. Volatility measurements use rolling standard deviations over 5, 10, and 20-day periods. We also compute the Relative Strength Index and ratios between current prices and moving averages.

Sector features from XLK help identify whether QQQ movements reflect broader technology sector trends or are specific to the index composition. These include XLK returns and rolling correlations between QQQ and XLK.

VIX features measure market volatility expectations through the VIX closing level, daily changes, and moving average ratios that indicate whether current volatility is elevated relative to recent history.

Macroeconomic features include the 10-year Treasury yield level and its 20-day change, plus year-over-year CPI inflation. Treasury yields affect discount rates for equity valuations, while inflation influences Federal Reserve policy decisions.

News sentiment features aggregate FinBERT predictions across articles published each day. FinBERT is a BERT model fine-tuned on financial text that classifies sentiment into positive, neutral, and negative categories. We compute mean sentiment scores rather than simple counts to focus on the prevailing sentiment tone.

### Model Training

Training follows a rigorous time-series protocol to ensure the models learn genuine predictive patterns rather than artifacts of data leakage. The data splits into 70% training, 15% validation, and 15% test sets, maintaining chronological order. Crucially, we implement purge gaps - removing several days of data between each split to account for feature lag periods and prevent information leakage.

Both XGBoost models use the same features but optimize different objectives. The regressor minimizes mean squared error to predict continuous return values. The classifier maximizes log loss to predict binary up/down direction and provides probability estimates.

Hyperparameter tuning occurs on the validation set, with early stopping to prevent overfitting. The final models save to both local storage and the Hopsworks Model Registry, enabling reproducibility and easy deployment.

### Point-in-Time Correctness

A critical aspect of the implementation is ensuring that features available for prediction genuinely would have been available at prediction time. Several design decisions support this:

Market data from Yahoo Finance becomes available after market close and can be used for next-day predictions without concern for look-ahead bias. Treasury yield data from FRED updates daily and is similarly safe to use with appropriate lags.

CPI data requires more careful handling. The Bureau of Labor Statistics releases CPI figures mid-month for the previous month. We implement a fixed-day-of-month release schedule approximation to avoid using data that would not yet have been released at prediction time.

News sentiment poses the trickiest challenge. Articles published during trading hours might discuss intraday price movements, creating potential look-ahead bias. We address this by aggregating sentiment from the previous day only, ensuring that sentiment features reflect only information available before the prediction target period.

### Automation

The system automates daily operations through GitHub Actions workflows. A daily pipeline runs at 9 PM UTC (after U.S. market close) on weekdays, executing the backfill notebooks, feature engineering, inference, and dashboard image generation. A separate weekly workflow handles model retraining every Monday morning to incorporate new data and adapt to changing market conditions.

This automation ensures predictions remain current without manual intervention, though the system includes data freshness validation that warns if features are more than 7 days old.

## Technologies

**Machine Learning**
- XGBoost 2.0 for gradient boosting models
- scikit-learn for metrics and preprocessing
- Hugging Face Transformers for FinBERT sentiment analysis
- PyTorch as the backend for transformer models

**Feature Store**
- Hopsworks 4.2 for feature storage and model registry
- PyArrow for efficient data serialization

**Data Sources**
- yfinance for market data from Yahoo Finance
- fredapi for macroeconomic data from FRED
- newsapi-python for news article retrieval
- FinBERT (ProsusAI/finbert) for sentiment analysis

**Development Environment**
- Jupyter for notebook-based development
- pandas and numpy for data manipulation
- matplotlib, seaborn, and plotly for visualization

**Dashboard**
- Gradio 6.2 for the interactive web interface
- Plotly for interactive charts

**Deployment**
- GitHub Actions for workflow automation
- Hugging Face Spaces for dashboard hosting

**Configuration Management**
- python-dotenv for environment variables
- PyYAML for configuration files

## Project Structure

```
ProjectScalable/
├── config/
│   ├── config.yaml              # Pipeline configuration
│   └── feature_config.yaml      # Feature engineering parameters
├── notebooks/
│   ├── 1_backfill_yahoo.ipynb   # Fetch market data
│   ├── 2_backfill_fred.ipynb    # Fetch macro data
│   ├── 3_backfill_news.ipynb    # Fetch news and analyze sentiment
│   ├── 4_market_features.ipynb  # Engineer market features
│   ├── 5_macro_sentiment_features.ipynb  # Engineer macro/sentiment features
│   ├── 6_create_feature_view.ipynb       # Combine all features
│   ├── 7_training.ipynb                  # Train models
│   ├── 8_daily_inference_FIXED.ipynb     # Generate predictions
│   ├── 9_backtest_evaluation.ipynb       # Evaluate performance
│   └── generate_dashboard_images.py      # Create visualizations
├── utils/
│   ├── data_fetchers.py         # API wrapper functions
│   ├── feature_functions.py     # Feature engineering utilities
│   ├── hopsworks_helpers.py     # Hopsworks integration
│   └── time_series_splits.py    # Train/val/test splitting
├── dashboard/
│   ├── app.py                   # Main dashboard application
│   └── static/images/           # Pre-generated performance charts
├── .github/workflows/
│   ├── daily_pipeline.yml       # Automated daily predictions
│   └── weekly_retraining.yml    # Weekly model updates
├── requirements.txt
└── README.md
```

## Setup and Installation

### Prerequisites

You need Python 3.10 or later and accounts for the following services:
- Hopsworks (free tier available at hopsworks.ai)
- NewsAPI (free tier: 100 requests/day)
- FRED API (free, no request limits)

### Installation Steps

1. Clone the repository and install dependencies:
```bash
git clone <repository-url>
cd ProjectScalable
pip install -r requirements.txt
```

2. Create a `.env` file with your API credentials:
```bash
HOPSWORKS_API_KEY=your_key_here
HOPSWORKS_PROJECT_NAME=your_project_name
NEWS_API_KEY=your_newsapi_key
FRED_API_KEY=your_fred_key
```

3. Run the initial backfill to populate historical data:
```bash
# These can run in parallel
jupyter nbconvert --execute 1_backfill_yahoo.ipynb
jupyter nbconvert --execute 2_backfill_fred.ipynb
jupyter nbconvert --execute 3_backfill_news.ipynb
```

4. Generate features:
```bash
jupyter nbconvert --execute 4_market_features.ipynb
jupyter nbconvert --execute 5_macro_sentiment_features.ipynb
jupyter nbconvert --execute 6_create_feature_view.ipynb
```

5. Train the models:
```bash
jupyter nbconvert --execute 7_training.ipynb
```

6. Generate initial predictions:
```bash
jupyter nbconvert --execute 8_daily_inference_FIXED.ipynb
```

### Dashboard

Launch the interactive dashboard to view predictions:
```bash
cd dashboard
python app.py
```

The dashboard provides four main views: live predictions showing the latest forecast with probability scores, historical performance comparing predictions against actual returns, feature importance analysis highlighting which factors drive predictions, and detailed metrics including accuracy, error measures, and backtested trading strategy performance.

## Results and Performance

The models demonstrate varying performance across different evaluation metrics, reflecting the inherent difficulty of financial prediction.

For directional accuracy, the classification model correctly predicts the sign of next-day returns approximately 52-55% of the time on held-out test data. While this may seem modest, it represents a meaningful edge in financial markets where a 51% accuracy can be profitable after accounting for transaction costs.

Regression performance shows higher variability. The R-squared metric often appears negative on test data, which is not uncommon for financial prediction tasks. This indicates that predicting exact return magnitudes remains challenging. However, the mean absolute error provides a more interpretable measure, typically ranging from 0.5% to 1.0%, meaning predictions are usually within one percentage point of actual returns.

The backtesting framework simulates a simple trading strategy: hold QQQ when the model predicts positive returns, stay in cash otherwise. Over recent 90-day windows, this strategy shows mixed results compared to a buy-and-hold baseline. The strategy sometimes outperforms during volatile periods when the model successfully predicts down days, but may underperform during strong trending markets where staying fully invested would be optimal.

These results align with the efficient market hypothesis and the known difficulty of consistently beating market benchmarks. The value of the system lies not in guaranteeing outperformance but in providing a systematic, explainable framework for generating daily forecasts that integrate multiple information sources.

## Challenges and Solutions

Several technical challenges arose during implementation.

The most significant involved ensuring proper temporal ordering and preventing look-ahead bias. Early versions inadvertently used future information by not implementing purge gaps between data splits. This was resolved by creating dedicated time-series splitting utilities that enforce strict chronological ordering and remove buffer periods between splits.

Data quality issues appeared with news sentiment analysis. The NewsAPI free tier limits daily requests, requiring careful management to avoid hitting rate limits. Additionally, FinBERT processing can be slow for large article batches. We addressed this through batch processing with progress tracking and caching of sentiment results in Hopsworks.

Model deployment in GitHub Actions initially failed due to missing system dependencies. The transformers library and PyTorch have specific requirements that weren't captured in the original requirements file. Adding explicit dependency management and using CPU-only PyTorch for inference resolved these issues.

Feature alignment between training and inference required careful attention. Models expect features in a specific order with specific names. We implemented dynamic feature loading that queries the model for expected features and reorders input data accordingly, preventing silent failures from misaligned features.

## Future Enhancements

Several extensions could improve the system's capabilities.

Additional feature sources would enrich the prediction model. Options pricing data would provide market-implied volatility and sentiment. Social media sentiment from Twitter or Reddit could capture retail investor behavior. Earnings announcement schedules and analyst estimates would help predict company-specific events affecting QQQ constituents.

More sophisticated models could better capture temporal dependencies. LSTM or Transformer architectures might learn longer-term patterns in price sequences. Ensemble methods combining multiple model types could improve robustness. Attention mechanisms could dynamically weight different feature categories based on market regime.

Risk management features would make the system more practical for actual trading. Position sizing based on prediction confidence would vary exposure according to signal strength. Stop-loss and take-profit rules would limit downside and lock in gains. Portfolio-level risk monitoring would ensure diversification across multiple prediction systems.

Real-time inference would reduce the gap between data availability and prediction generation. Currently the system runs once daily after market close. Intraday predictions updating as news breaks or technical patterns emerge would provide more timely signals.

These enhancements remain future work, as the current implementation successfully demonstrates the core concepts of feature store architecture, time-series aware machine learning, and automated prediction pipelines.

## Documentation

Additional documentation files provide detailed guidance:

- `DAILY_OPERATIONS.md` - Instructions for running the daily pipeline manually
- `GITHUB_ACTIONS_SETUP.md` - Guide for configuring automated workflows
- `HOPSWORKS_SETUP.md` - Hopsworks project configuration instructions
- `PIPELINE_FIXES_SUMMARY.md` - Record of pipeline coherence improvements
- `Scalable_project_proposal.pdf` - Original project proposal

## Conclusion

This project demonstrates a production-ready machine learning system for financial prediction. The architecture emphasizes reproducibility, correctness, and maintainability through feature store patterns, automated pipelines, and careful temporal handling.

While financial prediction remains inherently difficult and results are mixed, the system provides a solid foundation for further research and development. The modular design allows easy experimentation with new features, models, and prediction strategies.

Most importantly, the implementation addresses the real-world challenges of deploying machine learning in financial contexts: handling time-series data properly, avoiding look-ahead bias, integrating multiple data sources, and maintaining predictions through automated pipelines.

## License

This project was developed for educational purposes as part of the Scalable Machine Learning and Deep Learning course.

## Acknowledgments

We thank the course instructors for guidance on feature store architectures and serverless machine learning systems. The Hopsworks platform provided robust feature storage and model registry capabilities. The open-source community maintains the excellent libraries that power this system.

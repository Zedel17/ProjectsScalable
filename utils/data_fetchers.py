"""
Data fetching utilities for Yahoo Finance, FRED, and NewsAPI.
"""

import yfinance as yf
import pandas as pd
from fredapi import Fred
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()


def fetch_yahoo_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance.

    Args:
        ticker: Stock/ETF ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    data = data.reset_index()

    # yfinance returns MultiIndex columns like ('Close', 'QQQ')
    # Flatten by taking just the first level (the price type)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    # Now rename to lowercase
    column_mapping = {
        'Date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Adj Close': 'adj_close',
        'Volume': 'volume'
    }

    data = data.rename(columns=column_mapping)

    # Select only the columns we have (adj_close might not always be present)
    available_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    data = data[available_cols]

    return data


def validate_ohlcv_data(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame contains required OHLCV columns.

    Args:
        df: DataFrame to validate

    Returns:
        True if valid, raises ValueError otherwise
    """
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {df.columns.tolist()}")

    # Check for data
    if len(df) == 0:
        raise ValueError("DataFrame is empty - no data fetched")

    return True


def fetch_fred_series(series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch economic data from FRED API.

    Args:
        series_id: FRED series ID (e.g., 'DGS10', 'CPIAUCSL')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        DataFrame with date index and series values
    """
    fred_api_key = os.getenv('FRED_API_KEY')
    if not fred_api_key:
        raise ValueError("FRED_API_KEY not found in environment variables")

    fred = Fred(api_key=fred_api_key)
    data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)

    df = pd.DataFrame({
        'date': data.index,
        series_id.lower(): data.values
    })

    return df


def fetch_dgs10(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch 10-year Treasury yield (DGS10) from FRED.

    DGS10 is a daily series. Missing values (weekends/holidays) can be
    forward-filled from the last available value without leakage concerns
    since yields are known in real-time.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        DataFrame with columns: date, dgs10
    """
    df = fetch_fred_series('DGS10', start_date, end_date)
    df = df.rename(columns={'dgs10': 'dgs10'})
    df['date'] = pd.to_datetime(df['date'])
    return df


def fetch_cpi(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch CPI (CPIAUCSL) from FRED.

    CPIAUCSL is a monthly series. The raw data shows the CPI value for each month,
    but DOES NOT include release date information. This function fetches the raw
    monthly observations.

    IMPORTANT: CPI values must be aligned to their release dates, not their
    reference month, to avoid look-ahead bias. Use make_macro_daily_features()
    to properly align CPI to trading days with release date handling.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        DataFrame with columns: date (month start), cpiaucsl
    """
    df = fetch_fred_series('CPIAUCSL', start_date, end_date)
    df = df.rename(columns={'cpiaucsl': 'cpiaucsl'})
    df['date'] = pd.to_datetime(df['date'])
    return df


def fetch_cpi_alfred_realtime(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch CPI with ALFRED realtime vintage data to get exact release dates.

    ALFRED (Archival Federal Reserve Economic Data) provides "as-of" snapshots
    showing when data became available. This is the gold standard for avoiding
    look-ahead bias.

    Note: This requires making HTTP requests to FRED's API for each vintage.
    The fredapi library doesn't directly support realtime_start, so we use
    direct API calls.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        DataFrame with columns: date (release date), reference_date (month), cpiaucsl

    Raises:
        NotImplementedError: If ALFRED access is not available
    """
    # This is a placeholder for ALFRED realtime implementation
    # To implement: use requests library to call FRED API with realtime_start parameter
    # Example endpoint: https://api.stlouisfed.org/fred/series/observations
    #   ?series_id=CPIAUCSL&realtime_start=YYYY-MM-DD&api_key=...

    raise NotImplementedError(
        "ALFRED realtime fetching not yet implemented. "
        "Use fetch_cpi() with make_macro_daily_features(method='fixed_release') instead."
    )


def fetch_news_articles(query: str, date: str, max_articles: int = 100) -> list:
    """
    Fetch news articles for a specific date from NewsAPI.

    Args:
        query: Search query string
        date: Date in YYYY-MM-DD format
        max_articles: Maximum number of articles to fetch

    Returns:
        List of article dictionaries
    """
    news_api_key = os.getenv('NEWS_API_KEY')
    if not news_api_key:
        raise ValueError("NEWS_API_KEY not found in environment variables")

    newsapi = NewsApiClient(api_key=news_api_key)

    # NewsAPI allows searching within a date range
    articles = newsapi.get_everything(
        q=query,
        from_param=date,
        to=date,
        language='en',
        sort_by='relevancy',
        page_size=min(max_articles, 100)
    )

    return articles.get('articles', [])


def apply_finbert_sentiment(text: str, model, tokenizer) -> dict:
    """
    Apply FinBERT sentiment analysis to text.

    Args:
        text: Text to analyze
        model: FinBERT model
        tokenizer: FinBERT tokenizer

    Returns:
        Dictionary with sentiment scores
    """
    import torch

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # FinBERT outputs: negative, neutral, positive
    sentiment_scores = predictions[0].tolist()

    return {
        'negative': sentiment_scores[0],
        'neutral': sentiment_scores[1],
        'positive': sentiment_scores[2],
        'compound': sentiment_scores[2] - sentiment_scores[0]  # positive - negative
    }

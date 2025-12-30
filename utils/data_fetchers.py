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
        DataFrame with OHLCV data
    """
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    data = data.reset_index()
    data.columns = [col.lower() if isinstance(col, str) else col for col in data.columns]
    return data


def validate_ohlcv_data(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame contains required OHLCV columns.

    Args:
        df: DataFrame to validate

    Returns:
        True if valid, raises ValueError otherwise
    """
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

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

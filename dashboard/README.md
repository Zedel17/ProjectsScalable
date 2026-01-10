---
title: QQQ ETF Prediction Dashboard
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: false
---

# QQQ ETF Prediction Dashboard

Real-time predictions for next-day QQQ ETF returns using machine learning.

## Features

- **Next-Day Predictions**: Predicted return and up/down direction
- **Historical Performance**: Compare predictions vs actual returns
- **Feature Importance**: See which factors drive predictions
- **Live Data**: Powered by Hopsworks feature store

## Models

- **XGBoost Regressor**: Predicts next-day return percentage
- **XGBoost Classifier**: Predicts probability of up/down movement

## Data Sources

- Yahoo Finance (QQQ, XLK, VIX)
- FRED (Treasury yields, CPI)
- NewsAPI + FinBERT sentiment analysis

## Setup

This dashboard requires Hopsworks credentials stored as Hugging Face Secrets:
- `HOPSWORKS_API_KEY`
- `HOPSWORKS_PROJECT_NAME`

## Authors

Federico Mercurio & Margherita Santarossa

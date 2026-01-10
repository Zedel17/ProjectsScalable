# Deploy QQQ Dashboard to Hugging Face Spaces

## Step-by-Step Deployment Guide

### 1. Create a New Hugging Face Space

1. Go to https://huggingface.co/new-space
2. Fill in the details:
   - **Space name**: `qqq-prediction-dashboard` (or your preferred name)
   - **License**: Choose appropriate license (e.g., MIT)
   - **SDK**: Select **Gradio**
   - **Space hardware**: CPU basic (free tier is sufficient)
3. Click **Create Space**

### 2. Prepare Files for Upload

You need to upload these files to the Hugging Face Space:

```
dashboard/
├── app.py                  # Main dashboard application
├── requirements.txt        # Dependencies
├── README.md              # Space description
└── utils/
    └── hopsworks_helpers.py   # Helper functions
```

### 3. Upload Files to Hugging Face Space

**Option A: Using the Web Interface**

1. In your new Space, click **Files** → **Add file** → **Upload files**
2. Upload these files:
   - `dashboard/app.py` → upload as `app.py`
   - `dashboard/requirements.txt` → upload as `requirements.txt`
   - `dashboard/README.md` → upload as `README.md`
3. Create a `utils` folder:
   - Click **Add file** → **Create a new file**
   - Name it `utils/hopsworks_helpers.py`
   - Copy and paste the content from `utils/hopsworks_helpers.py`
   - Commit the file

**Option B: Using Git (Recommended)**

```bash
# Clone your Hugging Face Space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/qqq-prediction-dashboard
cd qqq-prediction-dashboard

# Copy files from your project
cp ../ProjectScalable/dashboard/app.py ./
cp ../ProjectScalable/dashboard/requirements.txt ./
cp ../ProjectScalable/dashboard/README.md ./

# Create utils directory and copy helper
mkdir -p utils
cp ../ProjectScalable/utils/hopsworks_helpers.py ./utils/

# Commit and push
git add .
git commit -m "Initial dashboard deployment"
git push
```

### 4. Configure Environment Variables (Secrets)

1. In your Space, go to **Settings** → **Repository secrets**
2. Add the following secrets:
   - **Name**: `HOPSWORKS_API_KEY`
     **Value**: Your Hopsworks API key
   - **Name**: `HOPSWORKS_PROJECT_NAME`
     **Value**: Your Hopsworks project name (e.g., `scalable_lab1_featurestore`)

### 5. Verify Deployment

1. Wait for the Space to build (usually 1-2 minutes)
2. The dashboard should automatically launch at: `https://huggingface.co/spaces/YOUR_USERNAME/qqq-prediction-dashboard`
3. Check the logs if there are any errors

## Files Breakdown

### `app.py`
Already configured in `dashboard/app.py` - no changes needed.

### `requirements.txt`
```
gradio==6.2.0
hopsworks==4.2.0
pandas
plotly
python-dotenv
```

### `README.md`
```markdown
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
- Next-Day Predictions: Predicted return and up/down direction
- Historical Performance: Compare predictions vs actual returns
- Feature Importance: See which factors drive predictions
- Live Data: Powered by Hopsworks feature store

## Models
- XGBoost Regressor: Predicts next-day return percentage
- XGBoost Classifier: Predicts probability of up/down movement

## Authors
Federico Mercurio & Margherita Santarossa
```

## Troubleshooting

**Issue**: Dashboard shows "No predictions available"
- **Solution**: Make sure you've run notebook `8_daily_inference.ipynb` to create predictions in Hopsworks

**Issue**: Authentication error
- **Solution**: Double-check your Hugging Face Secrets are correctly set with valid Hopsworks credentials

**Issue**: Import errors
- **Solution**: Verify `utils/hopsworks_helpers.py` is in the correct directory structure

**Issue**: Build fails
- **Solution**: Check the build logs in Hugging Face Space settings. Most common issue is missing dependencies in requirements.txt

## Current Dashboard URL
Once deployed, your dashboard will be live at:
```
https://huggingface.co/spaces/YOUR_USERNAME/qqq-prediction-dashboard
```

## Notes
- The dashboard fetches live data from Hopsworks on each refresh
- Free Hugging Face Spaces may have slower performance than local deployment
- The Space will automatically rebuild when you push changes to the repository

# GitHub Actions Setup Guide for Daily Pipeline

This guide explains how to set up automated daily execution of the QQQ prediction pipeline using GitHub Actions.

---

## Overview

The GitHub Actions workflow will automatically:
1. **Backfill data** from Yahoo Finance, FRED, and NewsAPI
2. **Engineer features** from raw data
3. **Generate predictions** for the next trading day
4. **Update visualizations** for the dashboard
5. **Save results** to Hopsworks

**Schedule**: Runs daily at 9 PM UTC (5 PM EST / 2 PM PST) on weekdays

---

## Step 1: Set Up GitHub Secrets

You need to configure API keys as GitHub secrets so they're not exposed in your code.

### How to Add Secrets:

1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add each of the following secrets:

| Secret Name | Value | Where to Get It |
|------------|-------|-----------------|
| `HOPSWORKS_API_KEY` | Your Hopsworks API key | [Hopsworks Settings](https://c.app.hopsworks.ai/account/api/generated) |
| `HOPSWORKS_PROJECT_NAME` | Your project name | From your Hopsworks project |
| `NEWS_API_KEY` | Your NewsAPI key | [NewsAPI.org](https://newsapi.org/account) |
| `FRED_API_KEY` | Your FRED API key | [FRED API Keys](https://fred.stlouisfed.org/docs/api/api_key.html) |

---

## Step 2: Commit and Push the Workflow File

The workflow file has been created at `.github/workflows/daily_pipeline.yml`.

**Push it to GitHub:**

```bash
cd /Users/federicomercurio/ProjectScalable

# Add the workflow file
git add .github/workflows/daily_pipeline.yml

# Commit
git commit -m "Add GitHub Actions workflow for daily pipeline automation"

# Push to main branch
git push origin main
```

---

## Step 3: Verify Workflow Setup

1. Go to your GitHub repository
2. Click on the **Actions** tab
3. You should see "Daily QQQ Prediction Pipeline" in the workflows list

---

## Step 4: Test the Workflow Manually

Before waiting for the scheduled run, test it manually:

1. Go to **Actions** tab
2. Click on "Daily QQQ Prediction Pipeline"
3. Click **Run workflow** button (top right)
4. Select branch: `main`
5. Click **Run workflow**

The workflow will execute immediately. Monitor the logs to ensure everything works.

---

## Step 5: Monitor Execution

### View Logs:

1. Go to **Actions** tab
2. Click on a workflow run to see details
3. Click on "run-pipeline" job
4. Expand each step to view logs

### Check Results:

After successful execution:
- **Hopsworks**: Check that new data was added to feature groups
- **Predictions**: Verify `qqq_predictions` feature group has latest entry
- **Dashboard**: Refresh dashboard to see updated predictions

---

## Workflow Schedule

The workflow runs automatically:
- **Time**: 9 PM UTC (5 PM EST / 2 PM PST)
- **Days**: Monday to Friday (weekdays only)
- **Timezone**: UTC

### Why This Time?

- US stock market closes at 4 PM EST
- By 5 PM EST, all end-of-day data is available
- Prediction generated ready for next trading day

---

## Troubleshooting

### Issue 1: Workflow Fails on Notebook Execution

**Possible causes:**
- API rate limits (NewsAPI: 100 requests/day on free tier)
- Hopsworks connection timeout
- Missing data from APIs

**Solution:**
- Check execution logs in GitHub Actions
- Verify API keys are correct in secrets
- Check if APIs are operational

### Issue 2: "No predictions available yet"

**Cause:** Feature groups may be empty or prediction notebook failed

**Solution:**
1. Check Hopsworks feature groups have data
2. Run backfill notebooks manually to populate data
3. Re-run the workflow

### Issue 3: Notebook Timeout

**Cause:** Notebook takes longer than 10 minutes (600 seconds)

**Solution:** Increase timeout in workflow file:
```yaml
--ExecutePreprocessor.timeout=1200  # 20 minutes
```

---

## Optional: Add Notifications

### Email Notifications

GitHub Actions sends email notifications by default for failed workflows to repository owner.

### Slack Notifications (Optional)

Add this step at the end of the workflow to notify on Slack:

```yaml
      - name: Notify Slack on Success
        if: success()
        uses: slackapi/slack-github-action@v1.24.0
        with:
          webhook-url: ${{ secrets.SLACK_WEBHOOK_URL }}
          payload: |
            {
              "text": "✅ Daily QQQ Pipeline completed successfully!"
            }
```

**Setup:**
1. Create a Slack webhook: https://api.slack.com/messaging/webhooks
2. Add `SLACK_WEBHOOK_URL` as GitHub secret
3. Uncomment the notification steps in workflow

---

## Advanced: Weekly Retraining

To add weekly model retraining, create another workflow file:

`.github/workflows/weekly_retraining.yml`:

```yaml
name: Weekly Model Retraining

on:
  schedule:
    # Run every Monday at 6 AM UTC
    - cron: '0 6 * * 1'
  workflow_dispatch:

jobs:
  retrain-models:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install jupyter nbconvert

      - name: Create .env file
        run: |
          echo "HOPSWORKS_API_KEY=${{ secrets.HOPSWORKS_API_KEY }}" >> .env
          echo "HOPSWORKS_PROJECT_NAME=${{ secrets.HOPSWORKS_PROJECT_NAME }}" >> .env

      - name: Retrain Models
        run: |
          cd notebooks
          jupyter nbconvert --to notebook --execute 7_training.ipynb --ExecutePreprocessor.timeout=1200

      - name: Regenerate Dashboard Images
        run: |
          cd notebooks
          python generate_dashboard_images.py
```

---

## Cost & Usage Considerations

### GitHub Actions

- **Free tier**: 2,000 minutes/month for public repos
- **Usage**: ~5-10 minutes per run
- **Monthly usage**: ~110-220 minutes (22 trading days)
- **Status**: Well within free tier ✅

### API Limits

| API | Free Tier Limit | Daily Usage | Status |
|-----|----------------|-------------|--------|
| NewsAPI | 100 requests/day | ~20-50 | ✅ OK |
| FRED | 1,000 requests/day | ~2-5 | ✅ OK |
| Yahoo Finance | Unlimited (rate limited) | ~10-20 | ✅ OK |
| Hopsworks | Free tier: 25 GB storage | Minimal | ✅ OK |

---

## Workflow File Structure

```
.github/
└── workflows/
    ├── daily_pipeline.yml       # Daily data + inference
    └── weekly_retraining.yml    # Weekly model retraining (optional)
```

---

## Monitoring Checklist

**Daily:**
- [ ] Check workflow completed successfully
- [ ] Verify prediction was saved to Hopsworks
- [ ] Check dashboard shows latest prediction

**Weekly:**
- [ ] Review prediction accuracy
- [ ] Monitor API usage limits
- [ ] Check for any failed workflow runs

**Monthly:**
- [ ] Evaluate model performance
- [ ] Review feature importance changes
- [ ] Check data quality

---

## Disabling Automatic Runs

If you need to temporarily disable automatic runs:

1. Go to **Actions** tab
2. Click on "Daily QQQ Prediction Pipeline"
3. Click **⋮** (three dots) → **Disable workflow**

To re-enable, click **Enable workflow**.

---

## Next Steps

After setting up GitHub Actions:

1. ✅ **Push workflow file to GitHub**
2. ✅ **Add all secrets in repository settings**
3. ✅ **Manually trigger first run to test**
4. ✅ **Verify prediction appears in Hopsworks**
5. ✅ **Check dashboard displays new prediction**
6. 🔄 **Wait for first scheduled run next trading day**

---

## Support

If you encounter issues:

1. Check GitHub Actions logs for error messages
2. Review `DAILY_OPERATIONS.md` for manual execution steps
3. Verify all API keys are valid and not expired
4. Check Hopsworks project is accessible

---

**Status**: Ready to deploy! 🚀

**Estimated setup time**: 10-15 minutes

"""
Generate visualization images for dashboard display
Run this after training and inference to create charts
"""

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from utils.hopsworks_helpers import get_feature_store, get_model_registry
import joblib
import warnings
import os
import glob
import shutil
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Output directory
OUTPUT_DIR = '../dashboard/static/images'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("GENERATING DASHBOARD IMAGES")
print("="*60)

# Load models
print("\n1. Loading models...")

# Create models directory if it doesn't exist
os.makedirs('../models', exist_ok=True)

regressor_path = '../models/qqq_regressor.pkl'
classifier_path = '../models/qqq_classifier.pkl'

# Download from Hopsworks Model Registry if not present locally
if not os.path.exists(regressor_path) or not os.path.exists(classifier_path):
    print("   Connecting to Hopsworks Model Registry...")
    mr = get_model_registry()

    if not os.path.exists(regressor_path):
        print("   Downloading regressor...")
        reg_model = mr.get_model("qqq_return_regressor", version=1)
        model_dir = reg_model.download()
        pkl_files = glob.glob(f"{model_dir}/*.pkl")
        if pkl_files:
            shutil.copy(pkl_files[0], regressor_path)
        else:
            raise FileNotFoundError(f"No .pkl file found in {model_dir}")

    if not os.path.exists(classifier_path):
        print("   Downloading classifier...")
        cls_model = mr.get_model("qqq_direction_classifier", version=1)
        model_dir = cls_model.download()
        pkl_files = glob.glob(f"{model_dir}/*.pkl")
        if pkl_files:
            shutil.copy(pkl_files[0], classifier_path)
        else:
            raise FileNotFoundError(f"No .pkl file found in {model_dir}")

# Load models
regressor = joblib.load(regressor_path)
classifier = joblib.load(classifier_path)
print("   ✓ Models loaded")

# Connect to Hopsworks and load data
print("\n2. Loading data from Hopsworks...")
fs = get_feature_store()
combined_fg = fs.get_feature_group('qqq_combined_features', version=1)
df = combined_fg.read()

# Sort by date
df['date'] = pd.to_datetime(df['date'])
if hasattr(df['date'].dtype, 'tz') and df['date'].dtype.tz is not None:
    df['date'] = df['date'].dt.tz_localize(None)
df = df.sort_values('date').reset_index(drop=True)
print(f"   ✓ Data loaded: {len(df)} rows")

# Prepare features (same as training)
feature_cols = [col for col in df.columns
                if col not in ['date', 'qqq_close', 'target_return', 'target_direction']]
X_all = df[feature_cols].copy()

# Drop same features as training
cols_to_drop = ['sentiment_mean', 'sentiment_std', 'article_count']
X_all = X_all.drop(columns=cols_to_drop, errors='ignore')

# Get targets
y_return = df['target_return']
y_direction = df['target_direction']

# Generate predictions for last 90 days
print("\n3. Generating predictions for last 90 days...")
eval_df = df.tail(90).copy()
X_eval = X_all.tail(90)
y_eval_return = y_return.tail(90)
y_eval_direction = y_direction.tail(90)

pred_returns = regressor.predict(X_eval)
pred_directions = classifier.predict(X_eval)
pred_probas = classifier.predict_proba(X_eval)[:, 1]
print("   ✓ Predictions generated")

# =============================================================================
# CHART 1: Predicted vs Actual Returns (Time Series)
# =============================================================================
print("\n4. Generating Chart 1: Predicted vs Actual Returns...")
fig, ax = plt.subplots(figsize=(12, 6))
dates = eval_df['date'].values

ax.plot(dates, y_eval_return.values, label='Actual Return',
        marker='o', alpha=0.9, linewidth=2.5, markersize=5, color='#2E86AB', markeredgecolor='white', markeredgewidth=0.5)
ax.plot(dates, pred_returns, label='Predicted Return',
        marker='s', alpha=0.9, linewidth=2.5, markersize=4, color='#FF6B35', markeredgecolor='white', markeredgewidth=0.5)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Return', fontsize=12, fontweight='bold')
ax.set_title('QQQ Predicted vs Actual Returns (Last 90 Days)',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper left', framealpha=0.9, shadow=True)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)

# Add metrics text box
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
mae = mean_absolute_error(y_eval_return, pred_returns)
rmse = np.sqrt(mean_squared_error(y_eval_return, pred_returns))
dir_acc = accuracy_score((y_eval_return > 0).astype(int), (pred_returns > 0).astype(int))

textstr = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nDir Acc: {dir_acc:.2%}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/1_predicted_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {OUTPUT_DIR}/1_predicted_vs_actual.png")

# =============================================================================
# CHART 2: Prediction Probabilities (Direction)
# =============================================================================
print("\n5. Generating Chart 2: Direction Probabilities...")
fig, ax = plt.subplots(figsize=(12, 6))

# Color bars by actual direction
colors = ['green' if d == 1 else 'red' for d in y_eval_direction]
ax.bar(dates, pred_probas, color=colors, alpha=0.6, width=0.8, edgecolor='black', linewidth=0.5)
ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')

ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Probability of UP Movement', fontsize=12, fontweight='bold')
ax.set_title('Predicted Probability of UP Movement (Green=Actual UP, Red=Actual DOWN)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, 1)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.tick_params(axis='x', rotation=45)

# Add accuracy
clf_acc = accuracy_score(y_eval_direction, pred_directions)
textstr = f'Classification Accuracy: {clf_acc:.2%}'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/2_direction_probabilities.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {OUTPUT_DIR}/2_direction_probabilities.png")

# =============================================================================
# CHART 3: Feature Importance (Regression Model)
# =============================================================================
print("\n6. Generating Chart 3: Feature Importance...")
fig, ax = plt.subplots(figsize=(10, 8))

feature_importance = pd.DataFrame({
    'feature': X_eval.columns,
    'importance': regressor.feature_importances_
}).sort_values('importance', ascending=False).head(15)

sns.barplot(data=feature_importance, y='feature', x='importance',
            ax=ax, palette='viridis')
ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Most Important Features (Regression Model)',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/3_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {OUTPUT_DIR}/3_feature_importance.png")

# =============================================================================
# CHART 4: Cumulative Returns (Strategy vs Buy & Hold)
# =============================================================================
print("\n7. Generating Chart 4: Trading Strategy Performance...")
fig, ax = plt.subplots(figsize=(12, 6))

# Simulate strategy
strategy_returns = []
buy_hold_returns = []

for i in range(len(eval_df)):
    actual_return = y_eval_return.iloc[i]
    predicted_direction = pred_directions[i]

    # Strategy: Long if predict UP, flat if predict DOWN
    if predicted_direction == 1:
        strategy_returns.append(actual_return)
    else:
        strategy_returns.append(0)

    buy_hold_returns.append(actual_return)

# Calculate cumulative returns
strategy_cumulative = np.cumprod(1 + np.array(strategy_returns)) - 1
buy_hold_cumulative = np.cumprod(1 + np.array(buy_hold_returns)) - 1

ax.plot(dates, strategy_cumulative * 100, label='Model Strategy',
        linewidth=3, color='#28A745', marker='o', markersize=3, markevery=10)
ax.plot(dates, buy_hold_cumulative * 100, label='Buy & Hold',
        linewidth=3, alpha=0.8, color='#6C757D', linestyle='--')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
ax.set_title('Model Strategy vs Buy & Hold (Last 90 Days)',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)

# Add performance summary
final_strategy = strategy_cumulative[-1] * 100
final_buy_hold = buy_hold_cumulative[-1] * 100
outperformance = final_strategy - final_buy_hold

textstr = f'Strategy: {final_strategy:+.2f}%\nBuy & Hold: {final_buy_hold:+.2f}%\nOutperformance: {outperformance:+.2f}%'
props = dict(boxstyle='round', facecolor='lightgreen' if outperformance > 0 else 'lightcoral', alpha=0.8)
ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='bottom', horizontalalignment='right', bbox=props, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/4_strategy_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {OUTPUT_DIR}/4_strategy_performance.png")

# =============================================================================
# CHART 5: Confusion Matrix (Classification)
# =============================================================================
print("\n8. Generating Chart 5: Confusion Matrix...")
from sklearn.metrics import confusion_matrix

fig, ax = plt.subplots(figsize=(8, 6))

cm = confusion_matrix(y_eval_direction, pred_directions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'],
            cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 14, 'fontweight': 'bold'})

ax.set_xlabel('Predicted Direction', fontsize=12, fontweight='bold')
ax.set_ylabel('Actual Direction', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix (Last 90 Days)', fontsize=14, fontweight='bold', pad=20)

# Add accuracy
textstr = f'Accuracy: {clf_acc:.2%}'
props = dict(boxstyle='round', facecolor='white', alpha=0.9)
ax.text(0.5, 1.15, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', horizontalalignment='center', bbox=props, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/5_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {OUTPUT_DIR}/5_confusion_matrix.png")

# =============================================================================
# CHART 6: Model Performance Summary (Metrics Dashboard)
# =============================================================================
print("\n9. Generating Chart 6: Performance Metrics Summary...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Model Performance Summary (Last 90 Days)', fontsize=16, fontweight='bold', y=0.995)

# Subplot 1: Scatter plot (Predicted vs Actual)
ax = axes[0, 0]
ax.scatter(y_eval_return, pred_returns, alpha=0.7, s=60, c='#2E86AB', edgecolors='black', linewidth=0.8)
ax.plot([y_eval_return.min(), y_eval_return.max()],
        [y_eval_return.min(), y_eval_return.max()],
        'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel('Actual Return', fontsize=11, fontweight='bold')
ax.set_ylabel('Predicted Return', fontsize=11, fontweight='bold')
ax.set_title('Predicted vs Actual Scatter', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Subplot 2: Residuals histogram
ax = axes[0, 1]
residuals = y_eval_return - pred_returns
ax.hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Prediction Error', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title(f'Residuals Distribution (Mean={residuals.mean():.4f})',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Subplot 3: ROC-like visualization (Probability distribution)
ax = axes[1, 0]
ax.hist(pred_probas[y_eval_direction == 0], bins=15, alpha=0.6,
        label='Actual Down', edgecolor='black', color='red')
ax.hist(pred_probas[y_eval_direction == 1], bins=15, alpha=0.6,
        label='Actual Up', edgecolor='black', color='green')
ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
ax.set_xlabel('Predicted Probability (UP)', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('Probability Distribution by Actual Direction', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Subplot 4: Metrics table
ax = axes[1, 1]
ax.axis('off')

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_eval_direction, pred_probas)

metrics_data = [
    ['Metric', 'Value'],
    ['', ''],
    ['Regression MAE', f'{mae:.4f}'],
    ['Regression RMSE', f'{rmse:.4f}'],
    ['Directional Accuracy', f'{dir_acc:.2%}'],
    ['', ''],
    ['Classification Accuracy', f'{clf_acc:.2%}'],
    ['AUC-ROC', f'{auc:.4f}'],
    ['', ''],
    ['Strategy Return', f'{final_strategy:+.2f}%'],
    ['Buy & Hold Return', f'{final_buy_hold:+.2f}%'],
    ['Outperformance', f'{outperformance:+.2f}%']
]

table = ax.table(cellText=metrics_data, cellLoc='left', loc='center',
                colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header row
for i in range(2):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style data rows
for i in range(2, len(metrics_data)):
    if metrics_data[i][0] == '':
        continue
    for j in range(2):
        if j == 1 and i in [2, 3]:  # Regression metrics
            table[(i, j)].set_facecolor('#E3F2FD')
        elif j == 1 and i in [6, 7]:  # Classification metrics
            table[(i, j)].set_facecolor('#FFF3E0')
        elif j == 1 and i in [9, 10, 11]:  # Strategy metrics
            table[(i, j)].set_facecolor('#F1F8E9')

ax.set_title('Performance Metrics', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/6_metrics_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {OUTPUT_DIR}/6_metrics_summary.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("IMAGE GENERATION COMPLETE")
print("="*60)
print(f"\nGenerated 6 images in: {OUTPUT_DIR}/")
print("\nImages created:")
print("  1. 1_predicted_vs_actual.png - Time series comparison")
print("  2. 2_direction_probabilities.png - Classification probabilities")
print("  3. 3_feature_importance.png - Top 15 features")
print("  4. 4_strategy_performance.png - Trading strategy vs buy & hold")
print("  5. 5_confusion_matrix.png - Classification confusion matrix")
print("  6. 6_metrics_summary.png - Complete performance dashboard")
print("\nThese images can now be displayed in your dashboard!")
print("="*60)

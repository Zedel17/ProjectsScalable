"""
Gradio Dashboard for QQQ Prediction Visualization
"""

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
sys.path.append('..')

from utils.hopsworks_helpers import get_feature_store, get_model_registry
from dotenv import load_dotenv

load_dotenv()


def load_predictions_data():
    """Load predictions and feature data from Hopsworks."""
    try:
        fs = get_feature_store()

        # Get predictions feature group
        predictions_fg = fs.get_feature_group('qqq_predictions', version=1)
        predictions_df = predictions_fg.read()

        # Get combined features for actuals
        combined_fg = fs.get_feature_group('qqq_combined_features', version=1)
        features_df = combined_fg.read()

        return predictions_df, features_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()


def create_prediction_chart(predictions_df, features_df):
    """Create interactive chart of predictions vs actuals."""
    if predictions_df.empty:
        return None

    # Merge predictions with actuals
    merged = predictions_df.merge(
        features_df[['date', 'target_return', 'target_direction']],
        left_on='prediction_date',
        right_on='date',
        how='left'
    )

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Predicted vs Actual Returns', 'Up/Down Probability'),
        vertical_spacing=0.15
    )

    # Returns plot
    fig.add_trace(
        go.Scatter(x=merged['prediction_date'], y=merged['target_return'],
                   name='Actual Return', mode='lines+markers'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=merged['prediction_date'], y=merged['predicted_return'],
                   name='Predicted Return', mode='lines+markers'),
        row=1, col=1
    )

    # Direction probability plot
    colors = ['red' if d == 0 else 'green' for d in merged['target_direction']]
    fig.add_trace(
        go.Bar(x=merged['prediction_date'], y=merged['predicted_proba_up'],
               name='Prob(Up)', marker_color=colors),
        row=2, col=1
    )

    fig.update_layout(height=700, showlegend=True)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Return", row=1, col=1)
    fig.update_yaxes(title_text="Probability", row=2, col=1)

    return fig


def create_feature_importance_chart(predictions_df):
    """Create feature importance visualization."""
    if predictions_df.empty:
        return None

    # Get latest prediction
    latest = predictions_df.iloc[-1]

    # Extract feature columns
    feature_cols = [col for col in latest.index if col.startswith('feature_')]
    feature_names = [col.replace('feature_', '') for col in feature_cols]
    feature_values = [latest[col] for col in feature_cols]

    fig = go.Figure(go.Bar(
        x=feature_values,
        y=feature_names,
        orientation='h'
    ))

    fig.update_layout(
        title="Top Features Driving Latest Prediction",
        xaxis_title="Feature Value",
        yaxis_title="Feature",
        height=400
    )

    return fig


def get_latest_prediction():
    """Get the latest prediction summary."""
    predictions_df, _ = load_predictions_data()

    if predictions_df.empty:
        return "No predictions available yet."

    latest = predictions_df.iloc[-1]

    direction = "UP ⬆️" if latest['predicted_direction'] == 1 else "DOWN ⬇️"

    summary = f"""
    ### Latest Prediction ({latest['prediction_date']})

    **Predicted Direction:** {direction}

    **Predicted Return:** {latest['predicted_return']:.4f} ({latest['predicted_return']*100:.2f}%)

    **Probability of UP:** {latest['predicted_proba_up']:.2%}

    **Probability of DOWN:** {(1-latest['predicted_proba_up']):.2%}

    ---
    *Models: Regressor v{int(latest['model_version_regressor'])}, Classifier v{int(latest['model_version_classifier'])}*
    """

    return summary


def refresh_dashboard():
    """Refresh all dashboard components."""
    predictions_df, features_df = load_predictions_data()

    prediction_summary = get_latest_prediction()
    prediction_chart = create_prediction_chart(predictions_df, features_df)
    feature_chart = create_feature_importance_chart(predictions_df)

    return prediction_summary, prediction_chart, feature_chart


# Image paths
IMAGE_DIR = "static/images"

# Create Gradio interface
with gr.Blocks(title="QQQ Prediction Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📈 QQQ ETF Prediction Dashboard")
    gr.Markdown("**Real-time predictions for next-day QQQ returns** | XGBoost + FinBERT Sentiment Analysis")

    with gr.Tabs():
        # Tab 1: Latest Prediction (Live Data)
        with gr.Tab("🔴 Live Prediction"):
            gr.Markdown("## Latest Prediction from Hopsworks")

            with gr.Row():
                with gr.Column(scale=1):
                    prediction_text = gr.Markdown(get_latest_prediction())
                    refresh_btn = gr.Button("🔄 Refresh Data", variant="primary", size="lg")

                with gr.Column(scale=2):
                    feature_importance = gr.Plot(label="Top Features Driving Latest Prediction")

            with gr.Row():
                prediction_chart = gr.Plot(label="Prediction History (From Hopsworks)")

            # Refresh button action
            refresh_btn.click(
                fn=refresh_dashboard,
                inputs=[],
                outputs=[prediction_text, prediction_chart, feature_importance]
            )

            # Load initial data
            demo.load(
                fn=refresh_dashboard,
                inputs=[],
                outputs=[prediction_text, prediction_chart, feature_importance]
            )

        # Tab 2: Performance Analysis
        with gr.Tab("📊 Performance Analysis"):
            gr.Markdown("## Model Performance on Last 90 Days")
            gr.Markdown("*These visualizations show backtesting results on recent data*")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Predicted vs Actual Returns")
                    gr.Markdown("**Blue** = Actual | **Orange** = Predicted")
                    gr.Image(f"{IMAGE_DIR}/1_predicted_vs_actual.png",
                            show_label=False)

                with gr.Column():
                    gr.Markdown("### Direction Probabilities")
                    gr.Markdown("**Green** = Actual UP | **Red** = Actual DOWN")
                    gr.Image(f"{IMAGE_DIR}/2_direction_probabilities.png",
                            show_label=False)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Trading Strategy Performance")
                    gr.Markdown("**Green** = Model Strategy | **Gray Dashed** = Buy & Hold")
                    gr.Image(f"{IMAGE_DIR}/4_strategy_performance.png",
                            show_label=False)

                with gr.Column():
                    gr.Markdown("### Confusion Matrix")
                    gr.Markdown("Classification accuracy on direction predictions")
                    gr.Image(f"{IMAGE_DIR}/5_confusion_matrix.png",
                            show_label=False)

        # Tab 3: Feature Analysis
        with gr.Tab("🔍 Feature Importance"):
            gr.Markdown("## Top Features Driving Model Predictions")
            gr.Markdown("These are the most important features for the regression model")

            gr.Image(f"{IMAGE_DIR}/3_feature_importance.png",
                    show_label=False)

            gr.Markdown("""
            ### Top Feature Categories:
            - **Technical Indicators**: volatility, RSI, moving average ratios
            - **Macro Factors**: Treasury yields (DGS10), CPI inflation
            - **Volatility Metrics**: VIX levels and changes
            - **Sector Performance**: XLK (Tech sector) returns and correlation
            - **News Sentiment**: FinBERT sentiment scores (positive/negative/neutral)
            """)

        # Tab 4: Complete Metrics
        with gr.Tab("📋 Metrics Summary"):
            gr.Markdown("## Complete Performance Dashboard")

            gr.Image(f"{IMAGE_DIR}/6_metrics_summary.png",
                    show_label=False)

            gr.Markdown("""
            ### Metric Definitions:

            **Regression Metrics:**
            - **MAE**: Mean Absolute Error - average prediction error
            - **RMSE**: Root Mean Squared Error - penalizes large errors more
            - **Directional Accuracy**: % of times we predict the correct sign (up/down)

            **Classification Metrics:**
            - **Accuracy**: % of correct direction predictions
            - **AUC-ROC**: Model's ability to distinguish UP from DOWN (0.5=random, 1.0=perfect)

            **Trading Strategy:**
            - **Strategy Return**: Performance if we only buy when model predicts UP
            - **Buy & Hold**: Performance if we always stay invested
            - **Outperformance**: Strategy minus buy & hold baseline
            """)


    gr.Markdown("""
    ---
    **Project**: Multi-Factor QQQ Prediction Using Financial and News Sentiment Data

    **Authors**: Federico Mercurio & Margherita Santarossa | **Date**: December 2025

    **Data Sources**: Yahoo Finance (QQQ, XLK, VIX) | FRED (Treasury, CPI) | NewsAPI + FinBERT

    **Models**: XGBoost Regressor + Classifier | **Features**: 30 engineered features
    """)


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)

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


# Create Gradio interface
with gr.Blocks(title="QQQ Prediction Dashboard") as demo:
    gr.Markdown("# QQQ ETF Prediction Dashboard")
    gr.Markdown("Real-time predictions for next-day QQQ returns using XGBoost and FinBERT sentiment")

    with gr.Row():
        with gr.Column(scale=1):
            prediction_text = gr.Markdown(get_latest_prediction())
            refresh_btn = gr.Button("🔄 Refresh Data")

        with gr.Column(scale=2):
            feature_importance = gr.Plot(label="Feature Importance")

    with gr.Row():
        prediction_chart = gr.Plot(label="Prediction History")

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


if __name__ == "__main__":
    demo.launch(share=False)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_predictions(
    y_test: pd.Series,
    test_lr: np.ndarray,
    test_rf: np.ndarray,
    test_arima: np.ndarray,
    pred_stack: np.ndarray,
    title: str = "Actual vs Predicted — Test Set"
):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_test.index,
        y=y_test.values,
        mode="lines",
        name="Actual",
        line=dict(width=3)
    ))

    fig.add_trace(go.Scatter(
        x=y_test.index,
        y=test_lr,
        mode="lines",
        name="Linear Regression"
    ))

    fig.add_trace(go.Scatter(
        x=y_test.index,
        y=test_rf,
        mode="lines",
        name="Random Forest"
    ))

    fig.add_trace(go.Scatter(
        x=y_test.index,
        y=test_arima,
        mode="lines",
        name="ARIMA"
    ))

    fig.add_trace(go.Scatter(
        x=y_test.index,
        y=pred_stack,
        mode="lines",
        name="STACKING (Ridge)",
        line=dict(width=4, dash="dash")
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99)
    )

    return fig


def plot_error_metrics(
    df_metrics: pd.DataFrame,
    metrics=("RMSE", "MAE"),
    title="Model Performance (Error Metrics)"
):
    df_plot = df_metrics[list(metrics)]

    fig = go.Figure()

    for model in df_plot.index:
        fig.add_trace(go.Bar(
            name=model,
            x=metrics,
            y=df_plot.loc[model].values
        ))

    fig.update_layout(
        title=title,
        barmode="group",
        yaxis_title="Metric value",
        template="plotly_white"
    )

    return fig


def plot_score_metrics(
    df_metrics: pd.DataFrame,
    metrics=("R2", "DA"),
    title="Model Performance (Score Metrics)"
):
    df_plot = df_metrics[list(metrics)]

    fig = go.Figure()

    for model in df_plot.index:
        fig.add_trace(go.Bar(
            name=model,
            x=metrics,
            y=df_plot.loc[model].values
        ))

    fig.update_layout(
        title=title,
        barmode="group",
        yaxis_title="Score value",
        template="plotly_white"
    )

    return fig

def plot_stack_pred(
    y_test,
    pred_stack
):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Actual vs Predicted",
            "RMSE & MAE",
            "R2 & DA",
            ""
        )
    )

    # Line chart
    fig.add_trace(go.Scatter(
        x=y_test.index,
        y=y_test.values,
        name="Actual",
        line=dict(width=3)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=y_test.index,
        y=pred_stack,
        name="STACKING",
        line=dict(width=3, dash="dash")
    ), row=1, col=1)

    return fig


def plot_full_dashboard(
    y_test,
    pred_stack,
    df_metrics
):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Actual vs Predicted",
            "RMSE & MAE",
            "R2 & DA",
            ""
        )
    )

    # Line chart
    fig.add_trace(go.Scatter(
        x=y_test.index,
        y=y_test.values,
        name="Actual",
        line=dict(width=3)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=y_test.index,
        y=pred_stack,
        name="STACKING",
        line=dict(width=3, dash="dash")
    ), row=1, col=1)

    # Error metrics
    for model in df_metrics.index:
        fig.add_trace(go.Bar(
            x=["RMSE", "MAE"],
            y=df_metrics.loc[model, ["RMSE", "MAE"]],
            name=model
        ), row=1, col=2)

    # Score metrics
    for model in df_metrics.index:
        fig.add_trace(go.Bar(
            x=["R2", "DA"],
            y=df_metrics.loc[model, ["R2", "DA"]],
            name=model,
            showlegend=False
        ), row=2, col=1)

    fig.update_layout(
        height=800,
        title="Stock Price Prediction – Model Comparison",
        template="plotly_white"
    )

    return fig

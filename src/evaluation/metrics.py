import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

def directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:

    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))

    return np.mean(true_direction == pred_direction)


def compute_regression_metrics(
    y_true,
    y_pred
) -> pd.DataFrame:

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    da = directional_accuracy(y_true, y_pred)

    metrics_df = pd.DataFrame(
        {
            "RMSE": [rmse],
            "MAE": [mae],
            "DA": [da],
            "R2": [r2]
        }
    )

    return metrics_df

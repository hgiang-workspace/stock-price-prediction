import numpy as np
import pandas as pd
from typing import Iterator, Tuple


def walk_forward_split(
    data: pd.DataFrame,
    train_size: int,
    test_size: int,
    step_size: int = None
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:

    if step_size is None:
        step_size = test_size

    n_samples = len(data)

    start = 0
    while True:
        train_start = start
        train_end = train_start + train_size
        test_end = train_end + test_size

        if test_end > n_samples:
            break

        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(train_end, test_end)

        yield train_idx, test_idx

        start += step_size


def walk_forward_validate(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    train_size: int,
    test_size: int,
    step_size: int = None,
    fit_kwargs: dict = None
):

    if fit_kwargs is None:
        fit_kwargs = {}

    predictions = []
    actuals = []

    for train_idx, test_idx in walk_forward_split(
        X, train_size, test_size, step_size
    ):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train, **fit_kwargs)

        y_pred = model.predict(X_test)

        predictions.extend(y_pred)
        actuals.extend(y_test.values)

    return np.array(predictions), np.array(actuals)

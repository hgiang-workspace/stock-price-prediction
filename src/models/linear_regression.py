import numpy as np
import pandas as pd
from typing import Tuple


class LinearRegression:

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 2000,
        random_state: int = 42
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.w = None
        self.b = None
        self.loss_history = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True
    ):

        np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        self.w = np.random.randn(n_features, 1) * 0.01
        self.b = 0.0

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if verbose:
            print(f"Bắt đầu huấn luyện với {n_samples} mẫu...")

        for epoch in range(self.epochs):

            y_pred = np.dot(X, self.w) + self.b
            error = y_pred - y

            loss = np.mean(error ** 2) / 2
            self.loss_history.append(loss)

            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.6f}")

        if verbose:
            print("\nHUẤN LUYỆN HOÀN TẤT")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w) + self.b

    def get_params(self) -> Tuple[np.ndarray, float]:
        return self.w, self.b

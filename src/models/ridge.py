import numpy as np


class RidgeRegression:

    def __init__(self, alpha: float = 0.001, fit_b: bool = True):
        self.alpha = alpha
        self.fit_b = fit_b
        self.W = None  
        self.b = None

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    def fit(self, X: np.ndarray, y: np.ndarray):

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if self.fit_b:
            Z = self._add_bias(X)
        else:
            Z = X

        n_features = Z.shape[1]

        I = np.eye(n_features)
        if self.fit_b:
            I[0, 0] = 0

        A = Z.T @ Z + self.alpha * I
        B = Z.T @ y

        W = np.linalg.solve(A, B)

        if self.fit_b:
            self.b = W[0, 0]
            self.W = W[1:].flatten()
        else:
            self.b = 0.0
            self.W = W.flatten()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:

        if self.fit_b:
            return X @ self.W + self.b
        return X @ self.W
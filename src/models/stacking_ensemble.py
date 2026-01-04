import numpy as np
import pandas as pd
import os

from src.models.linear_regression import LinearRegression
from src.models.random_forest import RandomForestRegressor
from src.models.arima import ARIMA
from src.models.ridge import RidgeRegression

from src.data_processing import walk_forward_validate
from src.data_processing.train_test_split import split_and_scale
from src.evaluation.metrics import compute_regression_metrics


class StackingEnsemble:

    def __init__(
        self,
        ridge_alpha=0.001,
        start_year=2000,
        first_train_end_year=2005,
        meta_train_year=2019
    ):
        self.ridge_alpha = ridge_alpha
        self.start_year = start_year
        self.first_train_end_year = first_train_end_year
        self.meta_train_year = meta_train_year

        self.lr_final = None
        self.rf_final = None
        self.arima_series = None

        self.meta_model = RidgeRegression(alpha=ridge_alpha)

        self.base_predictions = []
        self.base_targets = []

    def prepare_target(self, df, target_col="Adj Close"):
        df = df.copy()
        df["y_target"] = df[target_col].shift(-1)
        df.dropna(inplace=True)
        return df

    def train_base_models_walk_forward(self, df, feature_cols):

        years = sorted(df["Date"].dt.year.unique())

        for year in years:
            if year < self.first_train_end_year:
                continue
            if year >= self.meta_train_year:
                break

            train_df = df[df["Date"].dt.year <= year]
            test_df = df[df["Date"].dt.year == year + 1]

            if len(test_df) == 0:
                continue

            X_train = train_df[feature_cols].values
            y_train = train_df["y_target"].values

            X_test = test_df[feature_cols].values
            y_test = test_df["y_target"].values

            # Linear Regression 
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            lr_pred = lr.predict(X_test)
            print(f"Trained LR for year {year}")

            # Random Forest 
            rf = RandomForestRegressor()
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            print(f"Trained RF for year {year}")

            # ARIMA 
            arima_series = train_df["Adj Close"].values
            arima_pred = ARIMA(
                arima_series,
                p=2, d=1, q=1,
                steps=len(test_df)
            )
            print(f"Trained ARIMA for year {year}")

            # Tạo data cho meta-model
            for i in range(len(test_df)):
                self.base_predictions.append([
                float(lr_pred[i]),
                float(rf_pred[i]),
                float(arima_pred[i]),
                float(train_df["Adj Close"].iloc[-1])
            ])

                self.base_targets.append(float(y_test[i]))

        base_df = pd.DataFrame(
            self.base_predictions,
            columns=[
                "y_pred_LR",
                "y_pred_RF",
                "y_pred_ARIMA",
                "AdjClose_last_train"
            ]
        )
        output_dir = "data/processed"
        os.makedirs(output_dir, exist_ok=True)

        csv_path = f"{output_dir}/stacking_base_predictions.csv"
        base_df.to_csv(csv_path, index=False)

        print(f"Saved base model predictions to {csv_path}")
        print(base_df.head())


    def train_meta_model(self):

        Z = np.array(self.base_predictions, dtype=np.float64)
        y = np.array(self.base_targets, dtype=np.float64)

        self.meta_model.fit(Z, y)

    def retrain_base_models_full(self, df, feature_cols):

        X = df[feature_cols].values
        y = df["y_target"].values

        self.lr_final = LinearRegression()
        self.lr_final.fit(X, y)
        self.rf_final = RandomForestRegressor()
        self.rf_final.fit(X, y)
        self.arima_series = df["Adj Close"].values

    def predict_test(self, df_test, feature_cols):
        X_test = df_test[feature_cols].values

        lr_pred = self.lr_final.predict(X_test)
        rf_pred = self.rf_final.predict(X_test)

        arima_pred = ARIMA(
            self.arima_series,
            p=5, d=1, q=1,
            steps=len(df_test)
        )

        last_adj = self.arima_series[-1]

        Z_test = np.column_stack([
            lr_pred,
            rf_pred,
            arima_pred,
            np.repeat(last_adj, len(lr_pred))
        ])

        return self.meta_model.predict(Z_test)

    def evaluate(self, y_true, y_pred):
        return compute_regression_metrics(y_true, y_pred)

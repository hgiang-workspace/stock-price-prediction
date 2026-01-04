import pandas as pd
import numpy as np
from pathlib import Path

from src.data_collection.download_stock_data import download_stock_data
from src.data_collection.download_market_indices import download_market_index

from src.data_processing.preprocessing import preprocess_and_merge
from src.data_processing.feature_engineering import build_feature_set
from src.data_processing.train_test_split import split_and_scale

from src.data_processing.walk_forward import walk_forward_validate
from src.models.stacking_ensemble import StackingEnsemble

from src.evaluation.metrics import compute_regression_metrics


TICKER = "AAPL"
START_DATE = "2000-01-01"
END_DATE = "2026-01-01"

MARKET_INDICES = {
    "^VIX": "VIX",
    "^IXIC": "NASDAQ",
    "^TNX": "TNX"
}

FEATURE_COLS = [
    "return", "lag_1", "lag_5",
    "ma_20",
    "volatility_10",
    "VIX", "NASDAQ", "TNX"
]

print("Downloading stock data...")
download_stock_data(
    ticker=TICKER,
    start_date=START_DATE,
    end_date=END_DATE
)

print("Downloading market indices...")
for symbol, name in MARKET_INDICES.items():
    download_market_index(
        index_symbol=symbol,
        index_name=name,
        start_date=START_DATE,
        end_date=END_DATE
    )


print("Tiền xử lý & merge dữ liệu...")
df = preprocess_and_merge(
    ticker=TICKER,
    start_date=START_DATE,
    end_date=END_DATE,
    index_names=list(MARKET_INDICES.values())
)

print("Tạo các chỉ số kỹ thuật...")
df = build_feature_set(df)
df.dropna(inplace=True)

print("Training stacking ensemble...")



stacking = StackingEnsemble(
    ridge_alpha=0.001,
    start_year=2000,
    first_train_end_year=2010,
    meta_train_year=2019
)

df = stacking.prepare_target(df)
train_val_df, test_df = split_and_scale(df_fe=df, train_ratio=0.8, target="y_target")
print(train_val_df.head())
stacking.train_base_models_walk_forward(
    train_val_df,
    feature_cols=FEATURE_COLS
)

stacking.train_meta_model()

stacking.retrain_base_models_full(
    train_val_df,
    feature_cols=FEATURE_COLS
)

print("Predicting on test set...")
y_pred = stacking.predict_test(test_df, FEATURE_COLS)
y_true = test_df["y_target"].values

metrics = stacking.evaluate(y_true, y_pred)
print(metrics)

results = pd.DataFrame({
    "Date": test_df["Date"],
    "Actual": y_true,
    "Predicted": y_pred
})

print("PIPELINE COMPLETED SUCCESSFULLY")

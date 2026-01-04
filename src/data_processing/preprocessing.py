import pandas as pd
import os
from typing import List


def preprocess_and_merge(
    ticker: str,
    index_names: List[str],
    start_date: str,
    end_date: str,
    stock_dir: str = "data/raw/stocks",
    macro_dir: str = "data/raw/market_indices",
    output_dir: str = "data/processed"
) -> pd.DataFrame:

    os.makedirs(output_dir, exist_ok=True)

    # === Load stock data ===
    stock_path = f"{stock_dir}/{ticker}_historical_{start_date[:4]}_{end_date[:4]}.csv"
    stock_df = pd.read_csv(stock_path)

    stock_df["Date"] = pd.to_datetime(stock_df["Date"])

    data = stock_df.copy()

    # === Load & merge macro indices ===
    for index_name in index_names:
        index_file = f"{macro_dir}/{index_name}_{start_date[:4]}_{end_date[:4]}.csv"

        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Market index file not found: {index_file}")

        index_df = pd.read_csv(index_file)
        index_df["Date"] = pd.to_datetime(index_df["Date"])

        data = data.merge(index_df, on="Date", how="left")

    # === Sort & handle missing values ===
    data = data.sort_values("Date").reset_index(drop=True)

    # Forward-fill macro indicators (financial standard)
    macro_cols = index_names
    data[macro_cols] = data[macro_cols].ffill()

    output_path = f"{output_dir}/{ticker}_cleaned_{start_date[:4]}_{end_date[:4]}.csv"
    data.to_csv(output_path, index=False)

    print(f"[INFO] Saved merged dataset to {output_path}")
    print(data.head())
    print(data.tail())

    return data, output_path

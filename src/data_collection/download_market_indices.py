import yfinance as yf
import os

def download_market_index(
    index_symbol: str,
    index_name: str,
    start_date: str,
    end_date: str,
    output_dir: str = "data/raw/market_indices"
):

    os.makedirs(output_dir, exist_ok=True)

    df = yf.download(
        index_symbol,
        start=start_date,
        end=end_date,
        interval="1d"
    )

    df.reset_index(inplace=True)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    df = df[['Date', 'Close']]
    df.rename(columns={'Close': index_name}, inplace=True)

    file_path = f"{output_dir}/{index_name}_{start_date[:4]}_{end_date[:4]}.csv"
    df.to_csv(file_path, index=False)

    print(f"Saved market index data to {file_path}")
    print(df.head())

    return df

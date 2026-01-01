import yfinance as yf
import os

def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    output_dir: str = "data/raw/stocks"
):
    """
    Download historical stock price data from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'MSFT')
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    output_dir : str
        Directory to save CSV file
    """

    os.makedirs(output_dir, exist_ok=True)

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False
    )

    df.reset_index(inplace=True)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

    file_path = f"{output_dir}/{ticker}_historical_{start_date[:4]}_{end_date[:4]}.csv"
    df.to_csv(file_path, index=False)

    print(f"Saved stock data to {file_path}")
    print(df.head())

    return df

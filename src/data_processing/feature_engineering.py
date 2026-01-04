import pandas as pd
from typing import List

def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "Adj Close",
    lags: List[int] = [1, 5, 10]
) -> pd.DataFrame:

    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    return df


def add_return_and_volatility(
    df: pd.DataFrame,
    target_col: str = "Adj Close",
    vol_windows: List[int] = [5, 10]
) -> pd.DataFrame:

    df["return"] = df[target_col].pct_change()

    for window in vol_windows:
        df[f"volatility_{window}"] = df["return"].rolling(window).std()

    return df


def add_moving_averages(
    df: pd.DataFrame,
    target_col: str = "Adj Close",
    windows: List[int] = [5, 20]
) -> pd.DataFrame:

    for window in windows:
        df[f"ma_{window}"] = df[target_col].rolling(window).mean()

    return df


def compute_rsi(
    series: pd.Series,
    window: int = 14
) -> pd.Series:

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
):

    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal

    return macd, macd_signal, macd_hist


def add_technical_indicators(
    df: pd.DataFrame,
    target_col: str = "Adj Close"
) -> pd.DataFrame:

    df["RSI_14"] = compute_rsi(df[target_col], window=14)

    (
        df["MACD"],
        df["MACD_signal"],
        df["MACD_hist"]
    ) = compute_macd(df[target_col])

    return df


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:

    df["Date"] = pd.to_datetime(df["Date"])
    df = add_lag_features(df)
    df = add_return_and_volatility(df)
    df = add_moving_averages(df)
    df = add_technical_indicators(df)

    return df

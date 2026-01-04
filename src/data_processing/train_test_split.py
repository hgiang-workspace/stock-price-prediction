import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


TARGET = "Adj Close"

FEATURES = [
    "return", "lag_1", "lag_5",
    "ma_20",
    "volatility_10",
    "VIX", "NASDAQ", "TNX","Adj Close"
]


import pandas as pd
from sklearn.preprocessing import StandardScaler

def split_and_scale(
    df_fe: pd.DataFrame,
    train_ratio: float = 0.8,
    features = FEATURES,
    target = TARGET,
    date_col = "Date"
):

    # Tách cột
    X = df_fe[features]
    y = df_fe[target]
    dates = df_fe[date_col]

    n = len(df_fe)
    train_size = int(n * train_ratio)

    # Split theo thời gian
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    date_train = dates.iloc[:train_size]

    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]
    date_test = dates.iloc[train_size:]

    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Rebuild DataFrame
    train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=features,
        index=X_train.index
    )
    train_scaled[target] = y_train.values
    train_scaled[date_col] = date_train.values

    test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=features,
        index=X_test.index
    )
    test_scaled[target] = y_test.values
    test_scaled[date_col] = date_test.values

    # Đưa Date lên đầu cho dễ nhìn
    cols_order = [date_col] + features + [target]
    train_scaled = train_scaled[cols_order]
    test_scaled = test_scaled[cols_order]

    return train_scaled, test_scaled


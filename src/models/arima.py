import numpy as np

def difference(series, d):
    series = np.asarray(series)
    diff = series.copy()

    for _ in range(d):
        diff = np.diff(diff)

    return diff


def inverse_difference(last_real_value, diff_preds):
    restored = []
    prev = last_real_value

    for dp in diff_preds:
        value = prev + dp
        restored.append(value)
        prev = value

    return np.asarray(restored)

def fit_ar(y, p):
    Y = y[p:]
    X = []

    for i in range(p, len(y)):
        row = y[i - p:i][::-1]
        X.append(np.concatenate(([1], row)))  

    X = np.asarray(X)

    coef = np.linalg.inv(X.T @ X) @ X.T @ Y
    return coef


def predict_ar_step(history, coef):
    p = len(coef) - 1
    x = np.concatenate(([1], history[-p:][::-1]))
    return np.dot(coef, x)

def fit_ma(residuals, q, lr=0.001, epochs=300):
    theta = np.zeros(q)
    n = len(residuals)

    for _ in range(epochs):
        grad = np.zeros(q)

        for t in range(q, n):
            ma_term = np.dot(theta, residuals[t - q:t][::-1])
            err = residuals[t] - ma_term
            grad -= 2 * err * residuals[t - q:t][::-1]

        grad /= max(1, n - q)
        grad = np.clip(grad, -1, 1)
        theta -= lr * grad

    return theta


def predict_ma_step(errors, theta):
    q = len(theta)
    if q == 0:
        return 0.0
    return np.dot(theta, errors[-q:][::-1])

def predict_ma_step(errors, theta):
    q = len(theta)
    if q == 0:
        return 0
    return np.dot(theta, errors[-q:][::-1])


# ARIMA 
def ARIMA(series, p, d, q, steps):

    series = np.array(series)

    # DIFFERENCE
    diff = difference(series, d)

    # FIT AR
    ar_coef = fit_ar(diff, p)

    # fitted to get residuals 
    fitted = []
    for i in range(p, len(diff)):
        x = np.concatenate(([1], diff[i-p:i][::-1]))
        fitted.append(np.dot(ar_coef, x))

    residuals = diff[p:] - np.array(fitted)

    #FIT MA
    if q > 0:
        theta = fit_ma(residuals, q)
    else:
        theta = []

    # ROLLING FORECAST 
    history = list(diff)
    errors = list(residuals)
    diff_predictions = []

    for _ in range(steps):
        ar_part = predict_ar_step(history, ar_coef)
        ma_part = predict_ma_step(errors, theta)
        yhat = ar_part + ma_part

        diff_predictions.append(yhat)

        # update rolling memory
        history.append(yhat)

        # error UNKNOWN trong tương lai
        # giả định error = 0
        errors.append(0)

    # INVERSE DIFFERENCE 
    last_real_value = series[-1]
    forecast = inverse_difference(last_real_value, diff_predictions)

    return forecast

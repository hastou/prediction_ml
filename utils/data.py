import pandas as pd
from pandas import Series
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def interpolate_na(data_frame, col_x, cols_y):
    """
    Interpolate data in all cols in cols_y
    :param data_frame: DataFrame containing data
    :param col_x: name of the col to use as X
    :param cols_y: list of names to use as y, each col will be interpolated independently
    :return: modified DataFrame
    """
    df = data_frame.copy()
    X = data_frame[col_x]

    for col_y in cols_y:
        y = data_frame[col_y]  # Column to interpolate
        y_not_na = y.notna()  # Get null values

        X_train = X[y_not_na].values.reshape(-1, 1)
        y_train = y[y_not_na].values.reshape(-1, 1)
        X_pred = X[y.isna()].values.reshape(-1, 1)

        model = make_pipeline(PolynomialFeatures(4), Lasso())
        model.fit(X_train, y_train)
        y_pred = model.predict(X_pred)

        # todo: transform into series before concat
        X = pd.concat([Series(X_train[:, 0]), Series(y_train[:, 0])], axis=1)
        y = pd.concat([Series(X_pred[:, 0]), Series(y_pred[:, 0])], axis=1)

        # y = pd.concat([y_train, y_pred])
        # y.sort_index(inplace=True)  # Sort to have all values in the same order in the resulting DataFrame
        # df[col_y] = y
        df = pd.concat([X, y])

    return df







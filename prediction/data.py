import pandas as pd
import numpy as np


def get_data(
        establishment_number=None,
        columns_to_drop=[],
        year_to_separate=2017,
        path_to_file="data/data_all.gz",
        threshold_visitors=None,
        drop_na=False,
):
    """
    Get the data to train and test DataFrame from sqlite database
    :param threshold_visitors: if None, do nothing, if not, remove all values below this threshold
    :param path_to_file: path to the file to use
    :param establishment_number: id of the establishment in the database
    :param columns_to_drop: optional columns to drop when getting data
    :param year_to_separate: year to separate train and test, the year is part of test
    :return: X_train, y_train, X_test, y_test
    """
    date_col = "Date"
    data = pd.read_csv(path_to_file, sep=";", decimal=",", index_col="id")
    data[date_col] = pd.to_datetime(data[date_col])

    if drop_na:
        data.dropna(inplace=True)

    # Remove columns below threshold
    if threshold_visitors is not None:
        data = data[data["visitors"] >= threshold_visitors]

    if establishment_number is not None:
        data = data[data["library"] == establishment_number]

    # Separate data in train and test DataFrame
    train = data[data.loc[:, date_col].dt.year < year_to_separate]
    test = data[data.loc[:, date_col].dt.year >= year_to_separate]

    # Drop columns
    to_drop = [date_col, "Date:1"] + columns_to_drop
    train.drop(to_drop, axis=1, inplace=True)
    test.drop(to_drop, axis=1, inplace=True)

    visitors = "visitors"
    X_train = train.drop(visitors, axis=1)
    y_train = train[[visitors]]
    X_test = test.drop(visitors, axis=1)
    y_test = test[[visitors]]

    return X_train, y_train, X_test, y_test



def normalize_columns(df, cols=None):
    df = df.copy()

    if cols is None:
        cols = list(df)

    for col in cols:
        c = df[col]
        m = np.mean(c)
        if m == 0:
            continue
        s = np.std(c)
        c = (c - m) / s
        if np.sum(c.isna()) > 0:
            continue
        df[col] = c

    return df




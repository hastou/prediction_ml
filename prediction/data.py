import pandas as pd
import sqlite3
from sklearn.preprocessing import normalize


def get_request():
    pass


def get_data(
        establishment_number=None,
        columns_to_drop=[],
        year_to_separate=2017,
        database_name="all.db",
        normalize_data=False,
):
    """
    Get the data to train and test DataFrame from sqlite database
    :param establishment_number: id of the establishment in the database
    :param columns_to_drop: optional columns to drop when getting data
    :param database_name: name of the sqlite file to use as database
    :param year_to_separate: year to separate train and test, the year is part of test
    :param normalize_data: Normalize or not data
    :return: X_train, y_train, X_test, y_test
    """
    conn = sqlite3.connect(database_name)

    # Create temporary view
    sql_query_create_view = """CREATE TEMPORARY VIEW data_library AS SELECT * FROM data"""
    if establishment_number is not None:
        sql_query_create_view += " WHERE library = {}".format(establishment_number)
    cursor = conn.cursor()
    cursor.execute(sql_query_create_view)
    conn.commit()

    to_drop = ["date", "library"] + columns_to_drop
    request = "SELECT * from data_library d where Date(strftime('%Y', d.Date))"
    train = pd.read_sql_query(
        "{} < Date('{}');".format(request, year_to_separate),
        conn,
        index_col="id",
    )
    train.drop(to_drop, axis=1, inplace=True)
    test = pd.read_sql_query(
        "{} >= Date('{}');".format(request, year_to_separate),
        conn,
        index_col="id",
    )
    test.drop(to_drop, axis=1, inplace=True)
    conn.close()

    visitors = "visitors"

    # Extract X and y for train and test
    X_train = train.drop(visitors, axis=1)
    y_train = train[[visitors]]

    X_test = test.drop(visitors, axis=1)
    y_test = test[[visitors]]

    # Normalize data
    if normalize_data:
        axis = 0
        X_train = normalize(X_train, axis=axis)
        X_test = normalize(X_test, axis=axis)

    return X_train, y_train, X_test, y_test

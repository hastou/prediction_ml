import pandas as pd
import sqlite3


def get_data(library_number=None):
    conn = sqlite3.connect("all.db")

    sql_query_create_view = """CREATE TEMPORARY VIEW data_library AS SELECT * FROM data"""
    if library_number is not None:
        sql_query_create_view += " WHERE library = {}".format(library_number)
    cursor = conn.cursor()
    cursor.execute(sql_query_create_view)
    conn.commit()

    to_drop = ["date", "library"]
    train = pd.read_sql_query(
        """
        SELECT * from data_library d where Date(strftime('%Y', d.Date)) < Date('2017')
        """,
        conn,
        index_col="id",
    )
    train.drop(to_drop, axis=1, inplace=True)
    test = pd.read_sql_query(
        """
        select * from data_library d where Date(strftime('%Y', d.Date)) >= Date('2017')
        """,
        conn,
        index_col="id",
    )
    test.drop(to_drop, axis=1, inplace=True)
    conn.close()

    visitors = "visitors"

    X_train = train.drop(visitors, axis=1)
    y_train = train[[visitors]]

    X_test = test.drop(visitors, axis=1)
    y_test = test[[visitors]]

    return X_train, y_train, X_test, y_test

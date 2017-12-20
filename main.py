import numpy as np
from prediction.data import get_data, normalize_columns
from sklearn.preprocessing import normalize
from prediction.test_classes import test_models
import time
from warnings import filterwarnings


def normalize_tuple(data):
    X_train, y_train, X_test, y_test = data
    X_train_norm = normalize_columns(X_train)
    X_test_norm = normalize_columns(X_test)
    y_train_norm = normalize_columns(y_train)
    y_test_norm = normalize_columns(y_test)
    return X_train_norm, y_train_norm, X_test_norm, y_test_norm


def remove_cols(ll, to_drop):
    def d(data):
        if len(list(data)) < 2:
            return data
        return data.drop(to_drop, axis=1)
    return [d(l) for l in ll]


def main():

    for lib in range(3):
        print("===== Lib : " + str(lib) + " ======")
        cols = [
            "date_timestamp",
            # "day_of_year",
            "Vacances_A",
            "Vacances_B",
            # "Vacances_C",
            # "Férié",
            "library",
        ]
        data = get_data(
            columns_to_drop=cols,
            threshold_visitors=1, drop_na=True,
            establishment_number=lib
        )

        cols_without_weather = [
            'rainfall',
            'temperature',
            'humidity',
            'pressure',
            'pressure_variation',
            'pressure_variation_3h',
        ]

        l = len(cols_without_weather)
        for i in range(l):
            cols_d = cols_without_weather[i:]
            d = remove_cols(data, cols_d)
            test_models("lib " + str(i), d, print_results=True)
            break




if __name__ == "__main__":
    filterwarnings("ignore")
    main()



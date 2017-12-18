import numpy as np
from prediction.data import get_data, normalize_columns
from sklearn.preprocessing import normalize
from prediction.test_classes import test_models
import time


def normalize_tuple(data):
    X_train, y_train, X_test, y_test = data
    X_train_norm = normalize_columns(X_train)
    X_test_norm = normalize_columns(X_test)
    y_train_norm = normalize_columns(y_train)
    y_test_norm = normalize_columns(y_test)
    return X_train_norm, y_train_norm, X_test_norm, y_test_norm


def main():
    cols = [
        "date_timestamp",
        # "day_of_year",
        "Vacances_A",
        "Vacances_B",
        # "Vacances_C",
        # "Férié",
        "library",
    ]
    begin = time.time()
    data_lib_1 = get_data(establishment_number=1, columns_to_drop=cols, threshold_visitors=1)
    # data_lib_1 = normalize_tuple(data_lib_1)

    mid = time.time()
    print("Data loaded :", mid - begin)
    results = test_models("library_1", data_lib_1, print_results=True)
    print("End :", time.time() - mid)

    for r in results:
        print("{} : {} / {} ||| mean error : {} | mean : {} / {}".format(
            r["model_name"],
            round_float(r["score"]),
            round_float(r["score_adjusted"]),
            round_float(r["mean_error"]),

            round_float(r["mean_pred"]),
            round_float(r["mean"]),
        ))
        # data.to_csv("test.csv", sep=";", decimal=",")





if __name__ == "__main__":
    main()



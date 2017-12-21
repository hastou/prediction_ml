import numpy as np
from warnings import filterwarnings

from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.regression import r2_score, mean_absolute_error

from prediction.data import get_data, normalize_columns
from prediction.test_classes import test_models



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


def get_data_without_cols(lib=0):
    cols = [
        "date_timestamp",
        # "day_of_year",
        "Vacances_A",
        "Vacances_B",
        # "Vacances_C",
        # "Férié",
        "library",
    ]
    cols_without_weather = [
        'rainfall',
        'temperature',
        'humidity',
        'pressure',
        'pressure_variation',
        'pressure_variation_3h',
    ]
    data = get_data(
        columns_to_drop=cols + cols_without_weather, drop_na=True,
        establishment_number=lib,
        drop_value_below_threshold=None,
    )
    return data


def main_test_with_weather():

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
        cols_without_weather = [
            'rainfall',
            'temperature',
            'humidity',
            'pressure',
            'pressure_variation',
            'pressure_variation_3h',
        ]
        data = get_data(
            columns_to_drop=cols + cols_without_weather, drop_na=True,
            establishment_number=lib,
            drop_value_below_threshold=None,
        )
        test_models("Lib : " + str(lib), data, print_results=True)
        # length = len(cols_without_weather)
        # for i in range(length):
        #     cols_d = cols_without_weather[i:]
        #     d = remove_cols(data, cols_d)
        #     test_models("lib " + str(i), d, print_results=True)
        #     break


def main_grid_search():
    data = get_data_without_cols(2)
    X_train, y_train, X_test, y_test = data
    # test_models("Library 2", data, print_results=True)
    parameters = {
        # "criterion": ("mse", "mae"),
        # "max_features": ("auto", "sqrt", "log2"),
        # "max_depth": list(range(5, 16)),
        # "min_samples_split": list(range(2, 6)),
        # "min_samples_leaf": list(range(1, 5)),
        # "min_weight_fraction_leaf": np.arange(0, 0.5, 0.1),
        # "min_impurity_decrease": list(range(0, 4)),
    }
    model = RandomForestRegressor(
        n_estimators=10, n_jobs=-1, random_state=1,

        criterion="mse", max_features="sqrt",
        min_samples_split=3, max_depth=9,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0,
        min_impurity_decrease=0,
    )
    clf = GridSearchCV(model, parameters, n_jobs=-1)
    clf.fit(X_train, y_train.iloc[:, 0])
    results = clf.cv_results_
    best_params = clf.best_params_
    print()


def main():
    data = get_data_without_cols(2)
    X_train, y_train, X_test, y_test = data
    model = RandomForestRegressor(
        n_estimators=100, n_jobs=-1, random_state=1,

        criterion="mse", max_features="sqrt",
        # min_samples_split=3, max_depth=9,
        # min_samples_leaf=1,
        # min_weight_fraction_leaf=0,
        # min_impurity_decrease=0,
    )
    model.fit(X_train, y_train.iloc[:, 0])
    y_pred = model.predict(X_test).reshape(-1, 1)

    r2 = r2_score(y_test, y_pred)
    m_e = mean_absolute_error(y_test, y_pred)
    mean = y_test.mean(axis=0).mean()
    mean_pred = y_pred.mean(axis=0).mean()

    print("{} | mean error : {} | mean {} / {}".format(
        round(r2, 3),
        round(m_e, 3),
        round(mean_pred, 3),
        round(mean, 3),
    ))


if __name__ == "__main__":
    filterwarnings("ignore")
    main_test_with_weather()



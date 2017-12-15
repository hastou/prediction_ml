from sklearn import linear_model
from prediction.prediction_classes import PolynomialRegression
from prediction.score import calculate_r2_adjusted_score, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import pandas as pd
import numpy as np


def generate_range_of_polynomial_model(_class, name, begin=0, end=10):
    polynomial_models = []
    for i in range(begin, end):
        to_append = (
            _class(i),
            "{} {}".format(name, i)
        )
        polynomial_models.append(to_append)

    return polynomial_models


classes_to_test = [
    (
        linear_model.LinearRegression(),
        "Regression Linéaire",
    ),
    *generate_range_of_polynomial_model(PolynomialRegression, "Polynomial", 2, 9),

    # Les deux classes suivantes peuvent être bien des fois, donc ça vaut le coup de tester

    # /!\ This class add 330s of test /!\
    # Tu peux tenter de réduire grandement le nombre d'estimateurs
    # (
    #     RandomForestRegressor(n_estimators=100, criterion="mae", min_samples_split=2, random_state=1),
    #     "Random Forest : estimators:100, criterion=mae, min_samples_split=2, random_state=1"
    # ),

    # /!\ This class add 230s of test /!\
    # (
    #     SVR(),
    #     "SVR",
    # ),
]


def test_models(library_name, xy_train_test, models=classes_to_test):
    X_train, y_train, X_test, y_test = xy_train_test

    results = []

    for model, name in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        n, p = X_test.shape
        r_adj = calculate_r2_adjusted_score(r2, n, p)

        out = np.concatenate([X_test, y_test, y_pred], axis=1)
        # out = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test), pd.DataFrame(y_pred)], axis=1)
        df = pd.DataFrame(out, columns=[* list(X_test), * list(y_test), "pred"])
        results += [{
            "score": r2,
            "score_adjusted": r_adj,
            "parameters": list(X_test),
            "library_name": library_name,
            "model_name": name,
            "data": df,
        }]

    return results




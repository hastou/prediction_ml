from sklearn import linear_model
from prediction.prediction_classes import PolynomialRegression
from prediction.score import calculate_r2_adjusted_score, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR



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
    # *generate_range_of_polynomial_model(PolynomialRegression, "Polynomial", 2, 6),
    (
        PolynomialRegression(3),
        "Polynomial 3",
    ),
    (
        RandomForestRegressor(n_estimators=100, criterion="mae", min_samples_split=2, random_state=1),
        "Random Forest : estimators:100, criterion=mae, min_samples_split=2, random_state=1"
    ),
    (
        SVR(),
        "SVR",
    ),
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

        results += [{
            "score": r2,
            "score_adjusted": r_adj,
            "parameters": list(X_test),
            "library_name": library_name,
            "model_name": name,
        }]

    return results




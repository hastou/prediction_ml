from sklearn import linear_model
from models.polynomial_model import PolynomialRegression


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
    # todo: add classes here
    (
        linear_model.LinearRegression(),
        "Regression Linéaire",
    ),

]


classes_to_test.append(* generate_range_of_polynomial_model(PolynomialRegression, "Polynomial", 2, 5))


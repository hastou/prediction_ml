from sklearn import linear_model
from prediction.prediction_classes import PolynomialClass


def add_range(_class, name, begin=0, end=10):
    for i in range(begin, end):
        to_append = (
            _class(i),
            "{} {}".format(name, i)
        )
        classes_to_test.append(to_append)


classes_to_test = [
    # todo: add classes here
    (
        linear_model.LinearRegression(),
        "Regression Linéaire",
    ),

]


add_range(PolynomialClass, "Polynomial", 2)


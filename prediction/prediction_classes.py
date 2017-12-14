from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures


class PolynomialRegression(linear_model.LinearRegression):

    polynomial = None

    def __init__(self, degree=2):
        self.polynomial = PolynomialFeatures(degree=degree)
        super().__init__()

    def fit(self, x_train, y_train):
        x = self.polynomial.fit_transform(x_train)
        return super().fit(x, y_train)

    def predict(self, x_test):
        x = self.polynomial.fit_transform(x_test)
        return super().predict(x)

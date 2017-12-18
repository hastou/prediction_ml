from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class PolynomialRegression(LinearRegression):

    polynomial = None

    def __init__(self, degree=2):
        self.polynomial = PolynomialFeatures(degree=degree)
        self.degree = degree
        super().__init__(normalize=True)

    def fit(self, x_train, y_train):
        x = self.polynomial.fit_transform(x_train)
        return super().fit(x, y_train)

    def predict(self, x_test):
        x = self.polynomial.fit_transform(x_test)
        return super().predict(x)

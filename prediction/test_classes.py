from .base_class import BaseClass
from sklearn.linear_model import LinearRegression


class LinearRegression(BaseClass):

    regressor = LinearRegression()

    def fit(self, x_train, y_train):
        self.regressor.fit(x_train, y_train)

    def predict(self, x_test):
        return self.regressor.predict(x_test)



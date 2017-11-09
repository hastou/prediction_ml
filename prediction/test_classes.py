from .base_class import BaseClass
from sklearn.linear_model import LinearRegression
import copy

class LinearRegression(BaseClass):

    regressor = LinearRegression()

    def fit(self, X_train, y_train):
        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        return self.regressor.predict(X_test)



all_classes = []
_locals = copy.copy(locals())
for l in _locals:
    print(l, type(l))
    if isinstance(l, BaseClass) and l is not BaseClass:
        all_classes.append(l)


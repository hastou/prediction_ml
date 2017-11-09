from prediction.base_class import DataClass
import prediction.test_classes as test_classes
from prediction.base_class import BaseClass
from sklearn.metrics import r2_score


all_classes = []
for l in test_classes.__dict__.values():
    if not hasattr(l, "base_class"):
        continue
    if l.base_class and l is not BaseClass:
        all_classes.append(l)


data_path = "data/biblio_3_per_half_hour.csv"
data = DataClass(file_path=data_path)
x_train, y_train, x_test, y_test = data.separated_data


def calculate_score(test_class):
    t = test_class()
    t.fit(x_train, y_train)
    y_pred = t.predict(x_test)
    score = r2_score(y_test, y_pred)
    return {
        "name": test_class.__name__,
        "score": score,
    }


def print_score(_results):
    for r in _results:
        print(r["name"], ":", r["score"])

results = [calculate_score(test_class) for test_class in all_classes]
print_score(results)

from prediction.base_class import DataClass
from prediction.test_classes import classes_to_test
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
import copy


# data_path = "data/biblio_3_per_half_hour.csv"
# data = DataClass(file_path=data_path)
# x_train, y_train, x_test, y_test = data.separated_data
# x_size = data.x_parameters_size


def test_file(path_file, filename=None, path_json="results.json", **kwargs):
    if filename is None:
        filename = path_file
    data = DataClass(file_path=path_file, **kwargs)
    x_train, y_train, x_test, y_test = data.separated_data
    x_size = data.x_parameters_size

    _results = {}
    for test_class in classes_to_test:
        out = calculate_score(test_class)
    _results[out[0]] = out[1]
    results = {filename: _results}
    # write_results(results)
    old_results = read_old(path_json)
    r = merge_results(old_results, results)
    write_new_results(r, path_json)


# def best_results(all_results):
#     best_of_all = ()
#     bests = ()
#     for key in all_results:
#         results = all_results[key]
#         m = -1000
#         for k in results:
#
#
#     return best_of_all, bests


def calculate_r2_adjusted_score(r2, n, p):
    """
    Calculate R² adjusted score
    :param r2: R²_score
    :param n: size of the test sample | len(y_test)
    :param p: Number of parameters of x | len(x)
    :return: R² adjusted score
    """
    return 1 - (
        (1 - r2)*(
            (n - 1) /
            (n - p - 1)
        )
    )


def calculate_score(_test_class, x_train, y_train, x_test, y_test, x_size):
    t = _test_class[0]
    t.fit(x_train, y_train)
    y_pred = t.predict(x_test)
    score = r2_score(y_test, y_pred)
    r2_adj = calculate_r2_adjusted_score(score, len(y_test), x_size)
    return (
        _test_class[1],  # Name
        {
            "r2 score": score,
            "r2 adjusted": r2_adj,
        }
    )


def print_score(_results):
    for r in _results:
        print(r, ":", _results[r])


def read_old(path):
    content = ""
    with open(path, "r", encoding="utf-8") as file:
        content = file.read()
    j = {}
    try:
        j = json.loads(content)
    except:
        pass
    return j


def write_new_results(_results, path):
    with open(path, "w", encoding="utf-8") as file:
        j = json.dumps(_results, indent=4)
        file.write(j)


def merge_results(old_results, new_results):
    out = copy.copy(old_results)
    for key in new_results:
        if key not in out:
            out[key] = new_results[key]
        else:
            out[key] = merge_results_same_file(out[key], new_results[key])
    return out


def merge_results_same_file(old_results, new_results):
    for key in new_results:
        old_results[key] = new_results[key]
    return old_results


def write_results(_results, path="results.json"):
    """
    Deprecated
    """
    with open(path, "r+", encoding="UTF-8") as file:
        old_content = file.read()
        to_write = {}
        try:
            to_write = json.loads(old_content)
        except:
            pass
        file.seek(0)
        file.truncate()
        for r in _results:
            to_write[r] = _results[r]
        j = json.dumps(to_write, indent=4)
        file.write(j)

#
# results = {}
# for test_class in classes_to_test:
#     out = calculate_score(test_class)
#     results[out[0]] = out[1]
#
# print_score(results)
# write_results(results)
#




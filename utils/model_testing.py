import numpy as np
from datetime import datetime
import json

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

from utils.score import r2_score_adjusted


def test_model(model, X, y, cv=5, n_jobs=-1):
    """
    Not used function, just in case you need to test 1 model
    for multiple models, use test_models
    Test a model
    :param model: model to test
    :param X:
    :param y:
    :param cv: number of cross validation to do
    :param n_jobs: number of cpu to use, -1 means "all"
    :return: list of score, each score is one cross validation
    """
    n = len(y)
    p = len(X)
    scorer = make_scorer(r2_score_adjusted, n=n, p=p)
    results = cross_val_score(model, X, y, cv=cv, n_jobs=n_jobs, scoring=scorer)

    return results


def test_models(list_of_models, X, y, cv=5, n_jobs=-1):
    """
    Test a list of models
    :param list_of_models:
    :param X:
    :param y:
    :param cv: number of cross validation to do
    :param n_jobs: number of cpu to use, -1 means "all"
    :return:
    """
    results = []
    n = len(y)
    p = len(X)
    scorer = make_scorer(r2_score_adjusted, n=n, p=p)
    for model in list_of_models:
        r = cross_val_score(model, X, y, cv=cv, n_jobs=n_jobs, scoring=scorer)
        mean = np.mean(r)
        name = type(model).__name__
        results.append({"name": name, "score": mean})
    results.sort(key=lambda _r: _r["score"])
    return results


def write_new_results(results, path):
    with open(path, "a+", encoding="utf-8") as file:
        j = json.dumps(results, indent=4)
        signs = "==============="
        date = datetime.now().isoformat()
        result = "\n\n{}\nNew test\n{}\n{}\n{}".format(signs, date, signs, j)
        file.write(result)
    pass






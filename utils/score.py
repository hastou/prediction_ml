from sklearn.metrics import r2_score


def r2_score_adjusted(y_true, y_pred, n, p):
    """
    Compute the r² score adjusted
    :param y_true: y mesured/real
    :param y_pred: y predicted
    :param n:
    :param p:
    :return: r² score adjusted, numeric value

    .. seealso:: calculate_r2_adjusted_score
    """
    r2 = r2_score(y_true, y_pred)
    r_adj = calculate_r2_adjusted_score(r2, n, p)
    return r_adj


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

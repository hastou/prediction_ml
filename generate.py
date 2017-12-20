import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.regression import mean_absolute_error, r2_score

from prediction.data import get_data



def main(establishment_number=0):

    # Get data
    cols_to_drop = [
        "date_timestamp",
        "Vacances_A",
        "Vacances_B",
        "library",
    ]
    cols_weather = [
        'rainfall',
        'temperature',
        'humidity',
        'pressure',
        'pressure_variation',
        'pressure_variation_3h',
    ]

    data = get_data(
        columns_to_drop=cols_to_drop + cols_weather,
        threshold_visitors=0, drop_na=True,
        establishment_number=establishment_number
    )
    X_train, y_train, X_test, y_test = data

    # Train model and predict
    model = RandomForestRegressor(n_estimators=500, min_samples_split=2, random_state=1)
    y_train_array = y_train.iloc[:, 0]
    model.fit(X_train, y_train_array)
    y_pred = model.predict(X_test).reshape(-1, 1)


    # Write results
    total = np.concatenate([X_test, y_test, y_pred], axis=1)
    df = pd.DataFrame(total, columns=[*list(X_test), "Real visitor number", "Predicted visitor number"])
    df.to_csv("data/results.csv", sep=";", decimal=",")

    # Show scores
    r2 = r2_score(y_test, y_pred)
    max_visitors = y_test.max(axis=0).max()
    mean_visitors = y_test.mean(axis=0).mean()
    mean_error = mean_absolute_error(y_test, y_pred)

    print(
        """
        score R2   : {}
        max        : {}
        mean       : {}
        mean error : {}
        """.format(r2, max_visitors, mean_visitors, mean_error)
    )


if __name__ == "__main__":
    main(2)


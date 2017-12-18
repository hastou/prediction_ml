import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from models.polynomial_regression import PolynomialRegression

from establishments_data import EstablishmentsData
from processed_data import ProcessedData
from models.models import Models

if __name__ == "__main__":
    establishments_data = EstablishmentsData([
        "data/per_half_hour/biblio_1_per_half_hour.csv",
        "data/per_half_hour/biblio_2_per_half_hour.csv",
        "data/per_half_hour/biblio_3_per_half_hour.csv"
    ])
    processed_data = ProcessedData(establishments_data, True)
    models = Models(models=(
            # LinearRegression(n_jobs=-1),
            # PolynomialRegression(6),
            # PolynomialRegression(7),
            # PolynomialRegression(8),
            # DecisionTreeRegressor(random_state=0),
            RandomForestRegressor(n_estimators=100,  random_state=0),
    ), x_labels=(
            "day_of_week",
            "hour",
            "minute",
            "day_of_year",
            "establishment"
    ))


    models.compute_models(processed_data.processed_data[processed_data.processed_data["Date"] < "2017-01-01"])

    models.compute_scores(processed_data.processed_data[processed_data.processed_data["Date"] >= "2017-01-01"])
    print("end")
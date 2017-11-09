import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
regressor = LinearRegression()


def separate_learn_and_test_data(data):
    return data[data["Date"].dt.year > 2016]

def separate(inputs):
    verif_set = inputs[inputs["Date"].dt.year > 2016]
    trainning_set = inputs[inputs["Date"].dt.year <= 2016]
    (verif_X, verif_y) = get_x_y(verif_set)
    (training_X, training_y) = get_x_y(trainning_set)
    return (training_X, training_y, verif_X, verif_y)


def get_x_y(inputs):
    x = inputs.loc[:, ["weekday", "hours"]]
    y = inputs.loc[:, ["Visiteurs presents"]]
    return (x, y)

def get_filtered_data():
    inputs = pd.read_csv("data/biblio_3_per_half_hour.csv", sep=";")
    inputs['Date'] = pd.to_datetime(inputs['Date'], format='%d/%m/%Y %H:%M')
    inputs['weekday'] = inputs['Date'].map(lambda x: x.weekday())
    inputs["hours"] = inputs["Date"].dt.strftime("%H%M")
    filtered_inputs = inputs[inputs["Visiteurs presents"] > 0]
    # filtered_inputs = filtered_inputs[filtered_inputs["Date"].dt.hour == 14]
    return filtered_inputs

def train_regressor_linear_regression():
    pass


filtered_data = get_filtered_data()
(training_X, training_y, test_X, test_y) = separate(filtered_data)

regressor.fit(training_X, training_y)
y_pred = regressor.predict(test_X)

score = r2_score(test_y, y_pred)
print(score)

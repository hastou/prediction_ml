import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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

inputs = pd.read_csv("data/biblio_3_per_half_hour.csv", sep=";")
inputs['Date'] = pd.to_datetime(inputs['Date'], format='%d/%m/%Y %H:%M')
inputs['weekday'] = inputs['Date'].map(lambda x: x.weekday())
inputs["hours"] = inputs["Date"].dt.strftime("%H%M")
inputs["Visiteurs presents"] = pd.to_numeric(inputs["Visiteurs presents"])
# print(inputs["hours"])
filtered_inputs = inputs[inputs["Visiteurs presents"] > 0]
filtered_inputs = filtered_inputs[filtered_inputs["Date"].dt.hour == 14]
(training_X, training_y, test_X, test_y) = separate(filtered_inputs)


# X = filtered_inputs.loc[:, ["Date", "weekday"]]
# y = filtered_inputs.loc[:, ["Visiteurs presents"]]

# print(X)
# print(separate_learn_and_test_data(X))

def show_data(plot, test_X, test_y, title="", x_label="", y_label="", color="red", nbins=20):
    # plt.figure()
    plt.subplot(plot)
    plt.scatter(test_X, test_y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.locator_params(axis="y", tight=None, nbins=nbins)
    # plt.show()


regressor.fit(training_X, training_y)
y_pred = regressor.predict(test_X)
# plt.scatter(test_X["weekday"], test_y, color="red")
# plt.scatter(test_X["hours"], test_y, color="orange")
# plt.plot((range(0, 7)), regressor.predict((range(0, 7))), color="blue")
# plt.title("Salary vs Experience (Training set")
# plt.xlabel("Years of experience")
# plt.ylabel("Salary")
# plt.show()

show_data(211, test_X["weekday"], test_y, title="Weekday", x_label="Weekday", y_label="Visteurs présents")
show_data(212, test_X["hours"], test_y, title="Hours", x_label="Hours", y_label="Visiteurs présents")
plt.show()



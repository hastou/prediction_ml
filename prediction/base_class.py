import pandas as pd
from sklearn.metrics import r2_score

implement_this = "Please implement this method"


class BaseClass:
    def fit(self, X_train, y_train):
        raise NotImplementedError(implement_this)

    def predict(self, X_test):
        raise NotImplementedError(implement_this)


class DataClass:

    file_path = ""
    csv_sep = ";"

    def __init__(self, file_path="data.csv", sep=";"):
        self.file_path = file_path
        self.csv_sep = sep

    def get_filtered_data(self):
        inputs = pd.read_csv(self.path_file, sep=self.csv_sep)
        inputs['Date'] = pd.to_datetime(inputs['Date'], format='%d/%m/%Y %H:%M')
        inputs['weekday'] = inputs['Date'].map(lambda x: x.weekday())
        inputs["hours"] = inputs["Date"].dt.strftime("%H%M")
        filtered_inputs = inputs[inputs["Visiteurs presents"] > 0]
        return filtered_inputs

    def _get_x_y(inputs):
        x = inputs.loc[:, ["weekday", "hours"]]
        y = inputs.loc[:, ["Visiteurs presents"]]
        return (x, y)

    def get_separated_data(self):
        filtered_data = self.filtered_data
        verif_set = filtered_data[filtered_data["Date"].dt.year > 2016]
        trainning_set = filtered_data[filtered_data["Date"].dt.year <= 2016]
        (test_X, test_y) = self._get_x_y(verif_set)
        (training_X, training_y) = self._get_x_y(trainning_set)
        return (training_X, training_y, test_X, test_y)

    filtered_data = property(get_filtered_data)
    separeted_data = property(get_separated_data)


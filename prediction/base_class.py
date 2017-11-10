import pandas as pd


class DataClass:

    values = {
        "file_path": "data.csv",
        "csv_sep": ";",
        "year_to_separate": 2016,
        "x_parameters": ("weekday", "hours"),
        "y_parameters": ("Visiteurs presents"),
        "field_with_minimal_value": "Visiteurs presents",
        "minimum_value": 0,

    }

    def __init__(self, **kwargs):
        for k in self.values.keys():
            if k in kwargs:
                self.values[k] = kwargs[k]
        pass

    @property
    def filtered_data(self):
        inputs = pd.read_csv(self.values["file_path"], sep=self.values["csv_sep"])
        inputs['Date'] = pd.to_datetime(inputs['Date'], format='%d/%m/%Y %H:%M')
        inputs['weekday'] = inputs['Date'].map(lambda x: x.weekday())
        inputs["hours"] = inputs["Date"].dt.strftime("%H%M")
        filtered_inputs = inputs[inputs[self.values["field_with_minimal_value"]] > self.values["minimum_value"]]
        return filtered_inputs

    def _get_x_y(self, inputs):
        x = inputs.loc[:, self.values["x_parameters"]]
        y = inputs.loc[:, self.values["y_parameters"]]
        return x, y

    @property
    def separated_data(self):
        filtered_data = self.filtered_data
        year = self.values["year_to_separate"]
        test_set = filtered_data[filtered_data["Date"].dt.year > year]
        training_set = filtered_data[filtered_data["Date"].dt.year <= year]
        (test_X, test_y) = self._get_x_y(test_set)
        (training_X, training_y) = self._get_x_y(training_set)
        return training_X, training_y, test_X, test_y

    @property
    def x_parameters_size(self):
        x_parameters = self.values["x_parameters"]
        return len(x_parameters)

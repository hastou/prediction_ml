from sklearn.linear_model import LinearRegression
from sklearn.metrics.regression import mean_absolute_error


class Models:
    def __init__(self, models=(
            LinearRegression(n_jobs=-1),
    ), x_labels=(
            "day_of_week",
            "hour",
            "minute"
    ), y_label="Visiteurs presents"):
        self.models = models
        self.x_labels = x_labels
        self.y_label = y_label

    def compute_models(self, processed_data):
        x = processed_data.loc[:, self.x_labels]
        y = processed_data.loc[:, self.y_label]

        for model_instance in self.models:
            model_instance.fit(x, y)

    def compute_scores(self, processed_data):
        x = processed_data.loc[:, self.x_labels]
        y = processed_data.loc[:, self.y_label]

        for model_instance in self.models:
            score = model_instance.score(x, y)
            mean_absolute_error_value = mean_absolute_error(y, model_instance.predict(x))
            print(Models.get_model_description(model_instance) + " r2 : " + str(score))
            print(Models.get_model_description(model_instance) + " mean error : " + str(mean_absolute_error_value))


    @staticmethod
    def get_model_description(model):
        if type(model).__name__ == "PolynomialRegression":
            return type(model).__name__ + " " + str(model.degree) + "°"
        else:
            return type(model).__name__


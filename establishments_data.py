import pandas as pd


class EstablishmentsData:
    def __init__(self, establishments_data_paths):
        self.establishments_data = []

        for k, establishment_data_path in enumerate(establishments_data_paths):
            self.establishments_data.append(pd.read_csv(establishment_data_path, sep=";"))



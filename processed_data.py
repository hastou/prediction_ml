import pandas as pd
import datetime

class ProcessedData:
    def __init__(self, establishments_data, use_file_buffer=False):
        if (use_file_buffer):
            self.processed_data = pd.read_csv("data/processed_data_buffer", encoding='utf-8')
        else:
            self.processed_data = ProcessedData.process_establishments(establishments_data.establishments_data)
            self.processed_data.to_csv("data/processed_data_buffer", encoding='utf-8')

    @staticmethod
    def process_establishments(establishments_data):
        data = None

        for k, establishment_data in enumerate(establishments_data):
            establishment_data["establishment"] = k
            if k == 0:
                data = establishment_data.copy()
            else:
                data = pd.concat([data, establishment_data], ignore_index=True)


        print("Correcting values")
        data[data["Visiteurs presents"] < 0] = 0

        print("Reading dates")
        data.loc[:, "Date"] = pd.to_datetime(data.loc[:, "Date"])

        print("Removing weird dates")
        data = data.drop(data[data["Date"] < datetime.datetime(2015, 1, 1, 0, 0, 0, 0)].index)

        print("Creating day of year column")
        data["day_of_year"] = pd.DatetimeIndex(data["Date"]).dayofyear

        print("Creating day of week column")
        data["day_of_week"] = pd.DatetimeIndex(data["Date"]).dayofweek

        print("Creating hour column")
        data["hour"] = pd.DatetimeIndex(data["Date"]).hour

        print("Creating minute column")
        data["minute"] = pd.DatetimeIndex(data["Date"]).minute

        return data



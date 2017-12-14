from glob import glob
import pandas as pd
import numpy as np


from utils.model_testing import test_models, write_new_results
from utils.data import interpolate_na


path = "data/per_half_hour"


def get_all_files():
    files = glob(path + "/*.csv")
    csvs = []
    for file in files:
        csv = pd.read_csv(file, sep=";", decimal=",")
        csv = csv[csv.iloc[:, 1] != 0]
        csvs.append(csv)
    df = pd.concat(csvs, ignore_index=True)

    # Get holidays
    holidays_all = pd.read_csv("data/holidays.csv", encoding="latin_1", sep=";", decimal=",")
    holidays = holidays_all.loc[:, ["Date", "Vacances C", "Férié"]]
    holidays["Date"] = pd.to_datetime(holidays["Date"], infer_datetime_format=True)
    # holidays["ts"] = holidays["Date"].values.astype(np.int64) // 10 ** 9

    # Get weather
    # weather_all = pd.read_csv("data/meteo_paris.zip", sep=";", decimal=",", na_values="mq")
    # weather_cols = ["t", "rr24", "tn24", "t", "tx24", "tend24", "tw", "u", "ff", ]
    # weather = weather_all.loc[:, ["formattedDate"] + weather_cols]
    # weather.rename(index=str, columns={"formattedDate": "Date"}, inplace=True)
    # # weather.replace(np.nan, 0, inplace=True)
    # weather["Date"] = pd.to_datetime(weather["Date"], infer_datetime_format=True)
    # # "//" not a comment but a integer division
    # weather["ts"] = weather["Date"].values.astype(np.int64) // 10 ** 9
    #
    # weather = interpolate_na(weather, "ts", weather_cols)

    return df


df = get_all_files()
# print(df)
















from glob import glob
import pandas as pd
from utils.model_testing import test_models, write_new_results


path = "data/per_half_hour"


def get_all_files():
    all = None
    files = glob(path + "/*.csv")
    csvs = []
    for file in files:
        csv = pd.read_csv(file, sep=";", decimal=",")
        csv = csv[csv.iloc[:, 1] != 0]
        csvs.append(csv)
    df = pd.concat(csvs, ignore_index=True)

    holidays = pd.read_csv("data/holidays.csv", encoding="latin_1")
    meteo = pd.read_csv("data/meteo_paris.zip")

    return df


get_all_files()


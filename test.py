from glob import glob
from os.path import join
import pandas as pd
from utils.model_testing import test_models, write_new_results


path = join("data", "per_half_hour")


def get_all_files():
    all = None
    files = glob(join((path, "*.csv")))
    csvs = []
    for file in files:
        csv = pd.read_csv(file, sep=";", decimal=",")
        csv = csv[csv.iloc[:, 1] != 0]
        csvs.append(csv)
    df = pd.concat(csvs, ignore_index=True)
    return df




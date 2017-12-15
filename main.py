from prediction.data import get_data
from prediction.test_classes import test_models



if __name__ == "__main__":
    cols = [
        "date_timestamp",
        # "day_of_year",
        "Vacances_A",
        "Vacances_B",
        "Vacances_C",
        # "Férié",
    ]
    # cols = []
    print("Begin")
    data_lib_1 = get_data(establishment_number=1, columns_to_drop=cols)
    print("Data loaded")
    results = test_models("library_1", data_lib_1)
    # print(results)

    for r in results:
        print("{} : {} / {}".format(
            r["model_name"],
            r["score"],
            r["score_adjusted"],
        ))




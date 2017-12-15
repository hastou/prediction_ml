from prediction.data import get_data
from prediction.test_classes import test_models
import time


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
    begin = time.time()
    data_lib_1 = get_data(establishment_number=1, columns_to_drop=cols)
    mid = time.time()
    print("Data loaded :", mid - begin)
    results = test_models("library_1", data_lib_1)
    print("End :", time.time() - mid)
    # print(results)

    for r in results:
        print("{} : {} / {}".format(
            r["model_name"],
            r["score"],
            r["score_adjusted"],
        ))




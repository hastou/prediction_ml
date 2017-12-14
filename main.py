from prediction.data import get_data
from prediction.test_classes import test_models



if __name__ == "__main__":
    data_lib_1 = get_data(library_number=1)
    results = test_models("library_1", data_lib_1)
    print(results)




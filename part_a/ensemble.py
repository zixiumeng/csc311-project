from knn import knn_impute_by_user
from utils import *
from sklearn.impute import KNNImputer
from scipy import sparse
from item_response import *

import numpy as np
import scipy.sparse as sp


import matplotlib.pyplot as plt


def bootstrap(data):
    length = len(data['user_id'])
    b_data = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }

    for i in np.random.randint(0,length,length):
        b_data["user_id"].append(data["user_id"][i])
        b_data["question_id"].append(data["question_id"][i])
        b_data["is_correct"].append(data["is_correct"][i])
    # print(b_data)
    return b_data



def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    bagged_data_set = []
    for i in range(3):
        bagged_data_set.append(bootstrap(train_data))
    val_results = []
    test_results = []
    for d in bagged_data_set:
        iteration = 25
        theta, beta, val_acc_lst, val_lld_lst, train_lld_lst = irt(d, val_data, 0.01, iteration)
        val_results.append(evaluate(val_data, theta, beta))
        test_results.append(evaluate(test_data, theta, beta))

    theta, beta, val_acc_lst, val_lld_lst, train_lld_lst = irt(train_data, val_data, 0.01, iteration)
    print("-------------original data set-----------")
    print(f"Final validation and test accuracy:")
    print(f"validation accuracy: {evaluate(val_data, theta, beta)}")
    print(f"test accuracy: {evaluate(test_data, theta, beta)} ")
    print("------------------------------------------")

    print("-------------results-------------")
    print(f"Final validation and test accuracy:")
    print(f"validation accuracy: {np.mean(val_results)}")
    print(f"test accuracy: {np.mean(test_results)} ")
    print("-------------completed-----------")




    # matrix = np.array(list(train_data[key] for key in train_data.keys()), dtype=float)


    # matrix = sparse.coo_matrix(matrix).toarray()
    # print(matrix)
    # sparse_matrix = load_train_sparse("../data").toarray()
    # print(sparse_matrix)

    # print(np.size(matrix))

    #
    #
    # print(knn_impute_by_user(sparse_matrix, val_data, 11))


    # print(length)



if __name__ == "__main__":
    main()

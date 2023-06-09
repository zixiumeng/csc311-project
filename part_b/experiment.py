from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta, alpha):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    user_id = data['user_id']
    question_id = data['question_id']
    is_correct = data['is_correct']

    for i, q in enumerate(user_id):  # q represents the actual user_id
        alpha_i = alpha[q]
        theta_i = theta[q]  # ability of student
        beta_j = beta[question_id[i]]  # difficulty of question
        c_ij = is_correct[i]
        p = alpha_i * theta_i - beta_j

        log_lklihood += c_ij*np.log(np.exp(p)/(1 + np.exp(p))) + (1-c_ij)*(np.log(1 - np.exp(p)/(1 + np.exp(p))))



    # #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta, alpha):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    user_id = data['user_id']
    question_id = data['question_id']
    is_correct = data['is_correct']

    for i, q in enumerate(user_id):
        alpha_i = alpha[q]
        theta_i = theta[q]
        beta_j = beta[question_id[i]]
        c_ij = is_correct[i]
        # in order to maximize lld, use gradient ascent.
        theta[q] += lr * (alpha_i * c_ij - alpha_i * sigmoid(
            alpha_i * theta_i - beta_j))
        beta[question_id[i]] += lr * (
                    -c_ij + sigmoid(alpha_i * theta_i - beta_j))
        alpha[q] += lr * (theta_i * c_ij - theta_i * sigmoid(
            alpha_i * theta_i - beta_j))


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta, alpha


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    user_id = data['user_id']
    question_id = data['question_id']
    is_correct = data['is_correct']

    # theta = np.full(max(user_id)+1, -np.mean(is_correct))
    theta = np.zeros(max(user_id) + 1)
    alpha = np.ones(max(user_id) + 1)
    beta = np.zeros(max(question_id) + 1)

    val_acc_lst = []
    train_lld_lst = []
    val_lld_lst = []

    for i in range(iterations):
        print(alpha.sum())
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, alpha=alpha)
        train_lld_lst.append(neg_lld)
        val_lld_lst.append(neg_log_likelihood(val_data, theta, beta, alpha))

        score = evaluate(data=val_data, theta=theta, beta=beta, alpha=alpha)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta, alpha = update_theta_beta(data, lr, theta, beta, alpha)

    return alpha, theta, beta, val_acc_lst, val_lld_lst, train_lld_lst


def evaluate(data, theta, beta, alpha):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (alpha[u] * theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    iteration = 25
    alpha, theta, beta, val_acc_lst, val_lld_lst, train_lld_lst = irt(train_data,
                                                               val_data, 0.01,
                                                               iteration)

    print("-------------original data set-----------")
    print(f"Final validation and test accuracy:")
    print(f"validation accuracy: {evaluate(val_data, theta, beta,alpha)}")
    print(f"test accuracy: {evaluate(test_data, theta, beta,alpha)} ")
    print("------------------------------------------")
    iteration_lst = range(1, iteration + 1)
    # # Plot the x and y values using Matplotlib
    # plt.plot(iteration_lst, train_lld_lst)
    # plt.xlabel('iteration')
    # plt.ylabel('negative log-likelihood')
    # plt.title('training nlld vs iteration')
    # plt.show()

    # plt.plot(iteration_lst, val_lld_lst)
    # plt.xlabel('iteration')
    # plt.ylabel('negative log-likelihood')
    # plt.title('validation nlld vs iteration')
    # plt.show()

    # val_acc = evaluate(val_data, theta, beta)
    # test_acc = evaluate(test_data, theta, beta)
    # print(f'The final validation accuracy is {val_acc}, the test accuracy is {test_acc}. ')

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

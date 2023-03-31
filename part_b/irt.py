from utils import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))

def neg_log_likelihood(data, theta, beta, guess, alpha):
    """ Compute the negative log-likelihood.
    You may optionally replace the function arguments to receive a matrix.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param guess: Vector
    :param alpha: Vector
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

    for i, q in enumerate(user_id): # q represents the actual user_id
        theta_i = theta[q] # ability of student
        beta_j = beta[question_id[i]] # difficulty of question
        g_j = guess[question_id[i]]
        a_j = alpha[question_id[i]]
        c_ij = is_correct[i]

        # log_lklihood += (theta_i - beta_j) * c_ij - np.logaddexp(0, (theta_i - beta_j))
        exp = np.exp(a_j * (theta_i - beta_j))
        if (g_j + exp) < 0 or (1 - g_j) < 0:
            print('here is the error')
            print(g_j + exp)
        log_lklihood += c_ij * np.log(g_j + exp) + (1 - c_ij) * np.log(1 - g_j) - np.log(1 + exp)
    # #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta, guess, alpha):
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
    :param guess: Vector
    :param alpha: Vector
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
        theta_i = theta[q]
        beta_j = beta[question_id[i]]
        g_j = guess[question_id[i]]
        a_j = alpha[question_id[i]]
        c_ij = is_correct[i]
        # in order to maximize lld, use gradient ascent.
        exp = np.exp(a_j * (theta_i - beta_j))
        sigmoid_1 = sigmoid(a_j * (theta_i - beta_j))
        sigmoid_g = exp / (g_j + exp)

        theta[q] += lr * (c_ij * a_j * sigmoid_g - a_j * sigmoid_1)
        beta[question_id[i]] += lr * (-a_j * c_ij * sigmoid_g + a_j * sigmoid_1)
        # guess[question_id[i]] += lr * (c_ij * 1 / (g_j + exp) - (1 - c_ij) / (1 - g_j))
        alpha[question_id[i]] += lr * (c_ij * (theta_i - beta_j) * sigmoid_g - (theta_i - beta_j) * sigmoid_1)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta, guess, alpha


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
    beta = np.zeros(max(question_id) + 1)
    alpha = np.ones(max(question_id) + 1)
    # guess = np.zeros(max(question_id) + 1)
    guess = np.full((max(question_id) + 1,), 0.25)

    val_acc_lst = []
    train_lld_lst = []
    val_lld_lst = []

    for i in range(iterations):
        print(theta.sum())
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, guess=guess, alpha=alpha)
        train_lld_lst.append(neg_lld)
        val_lld_lst.append(neg_log_likelihood(val_data, theta, beta, guess=guess, alpha=alpha))

        score = evaluate(data=val_data, theta=theta, beta=beta, guess=guess, alpha=alpha)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta, guess, alpha = update_theta_beta(data, lr, theta, beta, guess, alpha)

    return theta, beta, guess, alpha, val_acc_lst, val_lld_lst, train_lld_lst


def evaluate(data, theta, beta, guess, alpha):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param guess: Vector
    :param alpha: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        # u = data["user_id"][i]
        # x = (theta[u] - beta[q]).sum()
        beta_j = beta[q]
        theta_i = theta[data["user_id"][i]]
        a_j = alpha[q]
        g_j = guess[q]
        p_a = (g_j + np.exp(a_j * (theta_i - beta_j))) / (1 + np.exp(a_j * (theta_i - beta_j)))
        # p_a = sigmoid(x)
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
    iteration = 60
    theta, beta, guess, alpha, val_acc_lst, val_lld_lst, train_lld_lst = irt(train_data, val_data, 0.005, iteration)
    iteration_lst = range(1, iteration+1)
    # Plot the x and y values using Matplotlib
    plt.plot(iteration_lst, train_lld_lst)
    plt.xlabel('iteration')
    plt.ylabel('negative log-likelihood')
    plt.title('training nlld vs iteration')
    plt.show()

    plt.plot(iteration_lst, val_lld_lst)
    plt.xlabel('iteration')
    plt.ylabel('negative log-likelihood')
    plt.title('validation nlld vs iteration')
    plt.show()

    val_acc = evaluate(val_data, theta, beta, guess, alpha)
    test_acc = evaluate(test_data, theta, beta, guess, alpha)
    print(f'The final validation accuracy is {val_acc}, the test accuracy is {test_acc}. ')

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    # j_1 = 1
    # j_2 = 10
    # j_3 = 100
    # # x_axis is theta, y_variable is prob
    # theta = np.sort(theta)
    # probs_1 = sigmoid(theta - beta[j_1])
    # probs_2 = sigmoid(theta - beta[j_2])
    # probs_3 = sigmoid(theta - beta[j_3])
    #
    # plt.plot(theta, probs_1, color='red', label='question1')
    # plt.plot(theta, probs_2, color='yellow', label='question2')
    # plt.plot(theta, probs_3, color='blue', label='question3')
    # plt.xlabel('theta')
    # plt.ylabel('p(c_ij=1)')
    # plt.title('probability of getting the answer correct vs student ability')
    # plt.legend()
    # plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

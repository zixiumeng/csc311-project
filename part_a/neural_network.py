from matplotlib import pyplot as plt
from numpy import average
from torch import le

from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

from tqdm import tqdm
import numpy as np


# from matplotlib.pyplot import plot as plt


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = torch.nn.Linear(num_question, k)
        self.h = torch.nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = inputs
        out = self.g(out)
        out = F.sigmoid(out)
        out = self.h(out)

        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, weight_decay=0):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function. 

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    num_student = train_data.shape[0]
    num_questions = train_data.shape[1]

    # max_acc = 0
    train_stats = {
        'epoch': [],
        'loss': [],
        'acc': []
    }
    validation_stats = {
        'epoch': [],
        'loss': [],
        'acc': []
    }

    # lst_accuracy = []
    # lst_loss = []
    for epoch in tqdm(range(num_epoch)):
        train_loss = 0.
        total_train_accurate_num = 0
        aggregate_answered_questions = 0
        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            target2 = train_data[user_id]

            # zeroing gradients after each iteration
            optimizer.zero_grad()
            output = model(inputs)  # !!!!!!!!!
            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]
            loss = torch.sum((output - target) ** 2.) + model.get_weight_norm() * (weight_decay / 2)
            # backward pass for computing the gradients of the loss w.r.t to learnable parameters
            loss.backward()  # calculate output.grad() and target.grad()
            train_loss += loss.item()

            # total_train_accurate_num += get_train_accurate_num(output, target2)
            # updating the parameters after each iteration
            optimizer.step()
            # record train_stats
            accuracy, total_answered_questions = get_train_accuracy(output, target2)
            total_train_accurate_num += accuracy * total_answered_questions
            aggregate_answered_questions += total_answered_questions
            train_stats['loss'].append(loss.item() / total_answered_questions)
            train_stats['acc'].append(accuracy)
            train_stats['epoch'].append(epoch + user_id / num_student)
        valid_acc, validation_total_loss = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
        print(total_train_accurate_num)

        # train_stats['loss'].append(train_loss)
        # train_stats['acc'].append(total_train_accurate_num / aggregate_answered_questions)
        # train_stats['epoch'].append(epoch)

        validation_stats['loss'].append(validation_total_loss)
        validation_stats['acc'].append(valid_acc)
        validation_stats['epoch'].append(epoch+1)

    return train_stats, validation_stats
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0
    total_loss = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1

        # calculate the loss
        diff_square = (guess - valid_data["is_correct"][i]) ** 2
        total_loss += diff_square

    model.train()

    print(total)

    return correct / float(total), total_loss / float(total)


# def get_train_accurate_num(model_output, real_output):
#     model_output_lst = model_output.tolist()[0]
#     real_output_lst = real_output.tolist()
#
#     assert (len(model_output_lst) == len(real_output_lst))
#
#     n = len(model_output_lst)
#     total_accurate_num = 0
#     for i in range(n):
#         shit =  real_output_lst[i]
#         if model_output_lst[i] == real_output_lst[i]:
#             total_accurate_num += 1
#
#     return total_accurate_num


def get_train_accuracy(model_output, real_output):
    model_output_lst = model_output.tolist()[0]
    real_output_lst = real_output.tolist()

    assert (len(model_output_lst) == len(real_output_lst))

    n = len(model_output_lst)
    total_comparisons = n
    total_accurate_num = 0
    for i in range(n):
        model_outcome = model_output_lst[i] > 0.5
        real_outcome = real_output_lst[i]
        if np.isnan(real_outcome):
            total_comparisons -=1
            continue
        if model_outcome == real_outcome:
            total_accurate_num += 1

    return total_accurate_num / total_comparisons, total_comparisons


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    lst_k = [10, 50, 100, 200, 500]
    lst_lr = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    lst_epoch = [2, 5, 10]

    # (c)
    # Using GPUs in PyTorch is pretty straightforward
    if torch.cuda.is_available():
        print('using GPU')
        device = torch.device('cuda')
    else:
        device = 'cpu'

    # # initialize the max_accuracy
    max_accuracy = 0
    optimal_k = 0
    optimal_epoch = 0
    optimal_lr = 0
    for k in lst_k:

        # Set optimization hyperparameters.
        for lr in lst_lr:
            for num_epoch in lst_epoch:
                model = AutoEncoder(num_question=train_matrix.shape[1], k=k)
                model.to(device)
                lamb = 0.0  # we don;t use it

                train_stats, validation_stats = train(model, lr, lamb, train_matrix, zero_train_matrix,
                                                      valid_data, num_epoch)
                # avg_accuracy = average(lst_accuracy)
                result_accuracy = max(validation_stats['acc'])
                if result_accuracy > max_accuracy:
                    max_accuracy = result_accuracy
                    optimal_k = k
                    # optimal_model = result_model
                    optimal_epoch = num_epoch
                    optimal_lr = lr

        print("k= ", k)
    print(optimal_k)
    print(optimal_lr)
    print(optimal_epoch)

    # (d)
    # # choose the device to run data
    # if torch.cuda.is_available():
    #     print('using GPU')
    #     device = torch.device('cuda')
    # else:
    #     device = 'cpu'

    model = AutoEncoder(num_question=train_matrix.shape[1], k=optimal_k)
    model.to(device)

    lamb = 0.0  # we don;t use it
    num_epoch = optimal_epoch #TODO: to change back to optimal_epoch
    lr = optimal_lr   #TODO: to change back to optimal_lr

    # for valid data
    train_stats, validation_stats = train(model, lr, lamb, train_matrix, zero_train_matrix,
                                          valid_data, num_epoch, 0)

    # plt.plot(lst_epoch, accuracy_lst, label="accuracy")
    # plt.title("Accuracy vs. Epoch")
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.show()

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 6))

    axes[0].plot(train_stats['epoch'], train_stats['loss'], label='train', )
    axes[0].plot(validation_stats['epoch'], validation_stats['loss'], label='validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')

    axes[1].plot(train_stats['epoch'], train_stats['acc'], label='train')
    axes[1].plot(validation_stats['epoch'], validation_stats['acc'], label='validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')

    plt.legend()
    plt.show()

    # for test data
    model = AutoEncoder(num_question=train_matrix.shape[1], k=optimal_k)
    model.to(device)
    train_stats, test_stats = train(model, lr, lamb, train_matrix, zero_train_matrix,
                                    test_data, num_epoch, 0)
    print(test_stats['acc'])

    # (e)
    weight_decay_lst = [0, 0.001, 0.01, 0.1, 1]
    max_accuracy = 0
    optimal_wd = 0
    optimal_train_stats = []
    optimal_validation_stats = []
    for weight_decay in weight_decay_lst:
        model = AutoEncoder(num_question=train_matrix.shape[1], k=optimal_k)
        model.to(device)
        train_stats, valid_stats = train(model, lr, lamb, train_matrix, zero_train_matrix,
                                         valid_data, num_epoch, weight_decay)

        if max(valid_stats['acc']) > max_accuracy:
            max_accuracy = max(valid_stats['acc'])
            optimal_wd = weight_decay
            optimal_train_stats = train_stats
            optimal_validation_stats = valid_stats

    print("optimal_wd:", optimal_wd)
    print("max_valid_accuracy:", max_accuracy)

    # for test_data
    model = AutoEncoder(num_question=train_matrix.shape[1], k=optimal_k)
    model.to(device)
    train_stats, test_stats = train(model, lr, lamb, train_matrix, zero_train_matrix,
                                     test_data, num_epoch, optimal_wd)
    print(test_stats['acc'])

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 6))

    axes[0].plot(optimal_train_stats['epoch'], optimal_train_stats['loss'], label='train', )
    axes[0].plot(optimal_validation_stats['epoch'], optimal_validation_stats['loss'], label='validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')

    axes[1].plot(optimal_train_stats['epoch'], optimal_train_stats['acc'], label='train')
    axes[1].plot(optimal_validation_stats['epoch'], optimal_validation_stats['acc'], label='validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')

    plt.legend()
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

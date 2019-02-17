import numpy as np
from sklearn import preprocessing


def softmax(x):
    """
    Compute softmax
    :param x:
    :return:
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def softmax_new(x):
    """
    Compute softmax
    :param x:
    :return:
    """
    new_x = np.zeros_like(x)
    for i in range(np.shape(x)[0]):
        new_x_i = softmax(x[i])
        new_x[i] = new_x_i
    return new_x

def sigmoid(x):
    """
    Compute sigmoid
    :param x:
    :return:
    """
    s_x = 1. / (1. + np.exp(np.negative(x.astype(np.float128))))
    return s_x

def shift_to_one(x):
    """
    shift x's center to 1.
    :param x:
    :return:
    """
    x_np = np.array(x)
    x_np = (x_np - x_np.min()) / (x_np.max() - x_np.min())
    x_shifted = x_np + 0.5
    return x_shifted.tolist()

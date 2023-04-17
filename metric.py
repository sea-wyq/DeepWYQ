"""Methods to compute common machine learning metrics"""

import numpy as np

def accuracy(preds, targets):
    total_num = len(preds)
    hit_num = int(np.sum(preds == targets))
    accuracy = 1.0 * hit_num / total_num
    info = {"hit_num": hit_num, "total_num": total_num}
    return accuracy, info

def softmax(x, t=1.0, axis=-1):
    x_ = x / t
    x_max = np.max(x_, axis=axis, keepdims=True)
    exps = np.exp(x_ - x_max)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def log_softmax(x, t=1.0, axis=-1):
    x_ = x / t
    x_max = np.max(x_, axis=axis, keepdims=True)
    exps = np.exp(x_ - x_max)
    exp_sum = np.sum(exps, axis=axis, keepdims=True)
    return x_ - x_max - np.log(exp_sum)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
"""Loss functions"""

import numpy as np
from metric import log_softmax
from metric import sigmoid
from metric import softmax


class Loss:

    def loss(self, *args, **kwargs):
        raise NotImplementedError

    def grad(self, *args, **kwargs):
        raise NotImplementedError


class SoftmaxCrossEntropy(Loss):

    def __init__(self, T=1.0, weights=None):
        self._weights = np.asarray(weights) if weights is not None else weights
        self._T = T

    def loss(self, logits, labels):
        nll = -(log_softmax(logits, t=self._T, axis=1) * labels).sum(axis=1)
        if self._weights is not None:
            nll *= self._weights[np.argmax(labels, axis=1)]
        return np.sum(nll) / labels.shape[0]

    def grad(self, logits, labels):
        grads = softmax(logits, t=self._T) - labels
        if self._weights is not None:
            grads *= self._weights
        return grads / labels.shape[0]


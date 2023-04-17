"""Example code for MNIST classification."""

import argparse
import os
import time

import numpy as np
from data_iterator import BatchIterator
from dataset import MNIST
from layer import Conv2D, Dense, Flatten, MaxPool2D, ReLU
import seeder
import metric
from net import Net
from loss import SoftmaxCrossEntropy
from optimizer import Adam
from model import Model

def main():
    if args.seed >= 0:
        seeder.random_seed(args.seed)

    mnist = MNIST(args.data_dir, one_hot=True)
    train_x, train_y = mnist.train_set
    test_x, test_y = mnist.test_set
    train_x = train_x.reshape((-1, 28, 28, 1))
    test_x = test_x.reshape((-1, 28, 28, 1))
    net = Net([
        Conv2D(kernel=[5, 5, 1, 6], stride=[1, 1]), ReLU(), MaxPool2D(pool_size=[2, 2], stride=[2, 2]),
        Conv2D(kernel=[5, 5, 6, 16], stride=[1, 1]), ReLU(), MaxPool2D(pool_size=[2, 2], stride=[2, 2]),
        Flatten(),
        Dense(120),  ReLU(),
        Dense(84), ReLU(),
        Dense(10)
    ])
    loss = SoftmaxCrossEntropy()
    optimizer = Adam(lr=args.lr)
    model = Model(net=net, loss=loss, optimizer=optimizer)

    if args.model_path is not None:
        model.load(args.model_path)
        evaluate(model, test_x, test_y)
    else:
        iterator = BatchIterator(batch_size=args.batch_size)
        for epoch in range(args.num_ep):
            t_start = time.time()
            for batch in iterator(train_x, train_y):
                pred = model.forward(batch.inputs)
                loss, grads = model.backward(pred, batch.targets)
                model.apply_grads(grads)
            print(f"Epoch {epoch} time cost: {time.time() - t_start}")
            evaluate(model, test_x, test_y)

        # save model
        if not os.path.isdir(args.model_dir):
            os.makedirs(args.model_dir)
        model_name = f"mnist-epoch{args.num_ep}.pkl"
        model_path = os.path.join(args.model_dir, model_name)
        model.save(model_path)
        print(f"Model saved in {model_path}")


def evaluate(model, test_x, test_y):
    model.is_training = False
    test_pred = model.forward(test_x)
    test_pred_idx = np.argmax(test_pred, axis=1)
    test_y_idx = np.argmax(test_y, axis=1)
    accuracy, info = metric.accuracy(test_pred_idx, test_y_idx)
    model.is_training = True
    print(f"accuracy: {accuracy:.4f} info: {info}")


if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(curr_dir, "data"))
    parser.add_argument("--model_dir", type=str,
                        default=os.path.join(curr_dir, "models"))
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--num_ep", default=8, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    args = parser.parse_args()
    main()

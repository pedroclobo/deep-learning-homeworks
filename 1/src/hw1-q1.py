#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a
        raise NotImplementedError


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        def softmax(z):
            return np.exp(z) / np.sum(np.exp(z))

        y_i_hat = softmax(self.W.dot(x_i))

        y_i_one_hot = np.zeros(y_i_hat.shape)
        y_i_one_hot[y_i] = 1

        self.W -= learning_rate * np.outer(y_i_hat - y_i_one_hot, x_i)


class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size):
        def softmax(z):
            scores = np.exp(z - np.max(z)) # avoid overflows
            return scores / np.sum(scores)

        def relu(x):
            return np.maximum(0, x)

        def cross_entropy(y_hat, y):
            return -np.sum(y * np.log(y_hat))

        self.g = relu
        self.o = softmax
        self.loss = cross_entropy

        units = [n_features, hidden_size, n_classes]

        self.W = [np.random.normal(0.1, 0.1, (units[i + 1], units[i]))
                  for i in range(len(units) - 1)]
        self.b = [np.zeros(units[i]) for i in range(1, len(units))]

    def predict(self, x_i):
        num_layers = len(self.W)
        self.hs = []

        for i in range(num_layers):
            h = x_i if i == 0 else self.hs[i - 1]
            z = self.W[i].dot(h) + self.b[i]
            if i < num_layers - 1:
                self.hs += [self.g(z)]

        self.y_i_hat = self.o(z)

        return self.y_i_hat

    def backward(self, x_i, y_i):
        self.grad_w = []
        self.grad_b = []

        for i in range(len(self.W) - 1, -1, -1):
            h = x_i if i == 0 else self.hs[i - 1]
            if i == len(self.W) - 1:
                grad_z = self.y_i_hat - y_i
            else:
                relu_dx = np.array([k >= 0 for k in self.hs[i - 1]])
                grad_z = self.W[i + 1].T.dot(grad_z) * relu_dx

            self.grad_w += [np.outer(grad_z, h)]
            self.grad_b += [grad_z]

        self.grad_w.reverse()
        self.grad_b.reverse()

    def update_weights(self, learning_rate):
        for i in range(len(self.W)):
            self.W[i] -= learning_rate * self.grad_w[i]
            self.b[i] -= learning_rate * self.grad_b[i]

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        n_correct = 0
        n_possible = y.shape[0]

        for x_i, y_i in zip(X, y):
            y_i_hat = self.predict(x_i)
            n_correct += np.argmax(y_i_hat) == y_i

        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        for x_i, y_i in zip(X, y):
            y_i_hat = self.predict(x_i)

            y_i_one_hot = np.zeros(y_i_hat.shape)
            y_i_one_hot[y_i] = 1

            self.backward(x_i, y_i_one_hot)
            self.update_weights(learning_rate)

        return self.loss(y_i_hat, y_i_one_hot)


def plot(epochs, train_accs, val_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    plt.show()

def plot_loss(epochs, loss):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []

    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )

        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs)
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss)


if __name__ == '__main__':
    main()

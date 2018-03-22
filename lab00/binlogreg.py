import numpy as np
import data
from matplotlib import pyplot as plt

param_niter = 1000
param_delta = 0.01


def sigm(x):
    return 1 / (1 + np.exp(-x))


def binlogreg_classify(X, w, b):
    return sigm(X.dot(w) + b)


def binlogreg_train(X, Y_):
    N = len(X)
    D = X.shape[1]
    w = np.random.randn(D, 1)
    b = 0

    for i in range(param_niter):
        scores = X.dot(w) + b
        probs = sigm(scores)
        loss = np.sum(-np.log(probs))/N

        if i % 100 == 0:
            print("iteration {}: loss {}".format(i, loss))

        dL_dscores = probs - Y_.reshape(-1, 1)

        grad_w = X.transpose().dot(dL_dscores)
        grad_b = np.sum(dL_dscores)

        w += -param_delta * grad_w
        b += -param_delta * grad_b

    return w, b


def binlogreg_decfun(w, b):
    return lambda X: binlogreg_classify(X, w, b)


if __name__ == '__main__':
    np.random.seed(100)

    X, Y_ = data.sample_gauss_2d(2, 100)

    w, b = binlogreg_train(X, Y_)

    probs = binlogreg_classify(X, w, b)
    Y = [1 if prob > 0.5 else 0 for prob in probs]

    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.T.argsort()].T)
    print("accuracy: {}, recall: {}, precision: {}, AP: {}"
          .format(accuracy, recall, precision, AP))

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(binlogreg_decfun(w, b), rect)
    data.graph_data(X, Y_, Y)

    plt.show()

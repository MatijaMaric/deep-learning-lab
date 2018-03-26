import numpy as np
import data
from matplotlib import pyplot as plt

param_niter = 100
param_delta = 0.1


def stable_softmax(x):
    exp_x_shifted = np.exp(x - np.max(x))
    probs = exp_x_shifted / np.sum(exp_x_shifted)
    return probs



def logreg_train(X, Y_):
    N, D = X.shape
    C = np.max(Y_) + 1

    # X ... N x D
    # Y ... N x C

    W = np.random.randn(C, D) # C x D
    b = np.zeros(C) # 1 x C

    for i in range(0, param_niter):
        scores = X.dot(W.T) + b       # N x C
        exp_scores = np.exp(scores) # N x C

        sum_exp = exp_scores.sum(axis=1) # N x 1

        probs = exp_scores / sum_exp.reshape(-1,1) # N x C
        correct_class_prob = probs[range(len(X)), Y_]

        correct_class_logprobs = -np.log(correct_class_prob)

        loss = correct_class_logprobs.sum() # scalar

        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))
        
        dL_ds = probs # N x C
        dL_ds[range(len(X)), Y_] -= 1

        grad_W = dL_ds.T.dot(X) # D x C
        grad_b = dL_ds.sum(axis=0)    # 1 x C

        W += -param_delta * grad_W / N
        b += -param_delta * grad_b / N

    return W, b


def logreg_classify(X, W, b):
    scores = X.dot(W.T) + b
    exp_scores = np.exp(scores)

    sumexp = exp_scores.sum(axis=1)
    return exp_scores / sumexp.reshape(-1,1)


def logreg_decfun(W, b):
    return lambda X: logreg_classify(X, W, b).argmax(axis=1)


if __name__ == '__main__':
    np.random.seed(100)

    X, Y_ = data.sample_gauss_2d(4, 100)

    W, b = logreg_train(X, Y_)

    probs = logreg_classify(X, W, b)
    Y = np.argmax(probs, axis=1)

    #data.eval_perf_multi(Y, Y_)

    rect = (np.min(X, axis=0), np.max(X, axis=0))

    data.graph_surface(logreg_decfun(W, b), rect)
    data.graph_data(X, Y_, Y)

    plt.show()

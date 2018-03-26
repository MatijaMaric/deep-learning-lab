import numpy as np
import data
import matplotlib.pyplot as plt

param_niter = 10000
param_delta = 0.1
param_lambda = 1e-3

K = 6
C = 2
N = 30

class fcann2:

    def __init__(self, H=5):
        self.H = H

    def fcann2_train(self, X, Y_):
        N, D = X.shape
        C = max(Y_) + 1

        W1 = np.random.randn(self.H, D)
        b1 = np.zeros(self.H)

        W2 = np.random.randn(C, self.H)
        b2 = np.zeros(C)

        for i in range(0, param_niter):
            scores1 = X.dot(W1.T) + b1
            H1 = np.maximum(scores1, 0)

            scores2 = H1.dot(W2.T) + b2
            exp_scores2 = np.exp(scores2)

            sum_exp2 = exp_scores2.sum(axis=1)

            probs2 = exp_scores2 / sum_exp2.reshape(-1,1)
            correct_class_prob = probs2[range(len(X)), Y_]

            correct_class_logprobs = -np.log(correct_class_prob)

            loss = correct_class_logprobs.sum()

            if i % 1000 == 0:
                print("iteration {}: loss {}".format(i, loss))

            dS2 = probs2
            dS2[range(len(X)), Y_] -= 1

            dW2 = dS2.T.dot(H1)
            db2 = dS2.sum(axis=0)

            dH1 = dS2.dot(W2)

            dS1 = dH1
            dS1[scores1 <= 0] = 0

            dW1 = dS1.T.dot(X)
            db1 = dS1.sum(axis=0)

            W1 += -param_delta * dW1 / N 
            b1 += -param_delta * db1 / N
            W2 += -param_delta * dW2 / N
            b2 += -param_delta * db2 / N

        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2


    def fcann2_classify(self, X):
        S1 = np.dot(X, self.W1.T) + self.b1
        H1 = np.maximum(S1, 0)
        S2 = np.dot(H1, self.W2.T) + self.b2

        exp_scores = np.exp(S2)
        sum_exp = exp_scores.sum(axis=1)

        probs = exp_scores / sum_exp.reshape(-1,1)
        return probs


if __name__ == '__main__':
    np.random.seed(100)

    X, Y_ = data.sample_gmm_2d(K, C, N)
    
    clf = fcann2(H=5)

    clf.fcann2_train(X, Y_)
    dec_fun = lambda X: clf.fcann2_classify(X)[:,1]
    probs = dec_fun(X)

    Y = probs > 0.5
    rect = (np.min(X, axis=0), np.max(X, axis=0))

    data.graph_surface(dec_fun, rect, offset=0.5)
    data.graph_data(X, Y_, Y)

    plt.show()

import numpy as np
import matplotlib.pyplot as plt


class Random2DGaussian:

    minx = 0
    maxx = 10
    miny = 0
    maxy = 10
    scale = 5

    def __init__(self):
        dx, dy = self.maxx - self.minx, self.maxy - self.miny
        mean = (self.minx, self.miny) + np.random.random_sample(2)*(dx, dy)
        eigen = (np.random.random_sample(2) * (dx, dy) / self.scale)**2
        D = np.diag(eigen)
        theta = np.random.random_sample() * np.pi * 2
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        cov = R.transpose().dot(D).dot(R)
        self.get_sample = lambda n: np.random.multivariate_normal(mean, cov, n)

def as_one_hot(C, Y):
    n = len(Y)
    Yoh = np.zeros((n, C))
    Yoh[range(n), Y] = 1
    return Yoh


def sample_gauss_2d(C, N, one_hot=False):
    X = []
    y = []
    for i in range(C):
        X.extend(Random2DGaussian().get_sample(N))
        y.extend([i]*N)
    if one_hot:
        return np.array(X), np.array(y), as_one_hot(C, y)
    return np.array(X), np.array(y)


def sample_gmm_2d(K, C, N, one_hot=False):
    Gs = [Random2DGaussian() for _ in range(K)]
    Gc = np.random.randint(0, C, K)

    X = np.array([g.get_sample(N) for g in Gs]).reshape(-1, 2)
    Y = np.repeat(Gc, N)

    if one_hot:
        return X, Y, as_one_hot(C, Y)
    return X, Y

def eval_perf_binary(Y, Y_):
    tp = np.sum(np.logical_and(Y == Y_, Y_ == 1))
    fn = np.sum(np.logical_and(Y != Y_, Y_ == 1))
    tn = np.sum(np.logical_and(Y == Y_, Y_ == 0))
    fp = np.sum(np.logical_and(Y != Y_, Y_ == 0))

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + fn + tn + fp)
    return accuracy, recall, precision


def eval_AP(Y):
    N = len(Y)
    pos = np.sum(Y)
    neg = N - pos

    tp = pos
    tn = 0
    fn = 0
    fp = neg

    sumprec = 0

    for x in Y:
        precision = tp / (tp + fp)

        if x:
            sumprec += precision

        tp -= x
        fn += x
        fp -= not x
        tn += not x

    return (sumprec / pos)[0]


def graph_data(X, Y_, Y, special=[]):
    palette = ([0.5, 0.5, 0.5], [1, 1, 1], [0.2, 0.2, 0.2])
    colors = np.tile([0.0, 0.0, 0.0], (Y_.shape[0], 1))
    for i in range(len(palette)):
        colors[Y_ == i] = palette[i]

    sizes = np.repeat(20, len(Y_))
    sizes[special] *= 2

    good = (Y_ == Y)
    plt.scatter(X[good, 0], X[good, 1], c=colors[good], s=sizes[good],
                marker='o', edgecolors=[0, 0, 0])

    bad = (Y_ != Y)
    plt.scatter(X[bad, 0], X[bad, 1], c=colors[bad], s=sizes[bad],
                marker='s', edgecolors=[0, 0, 0])


def graph_surface(fun, rect, offset=0.5, width=800, height=800):
    X = np.linspace(rect[0][1], rect[1][1], width)
    Y = np.linspace(rect[0][0], rect[1][0], height)
    x0, x1 = np.meshgrid(Y, X)
    grid = np.stack((x0.flatten(), x1.flatten()), axis=1)

    vals = fun(grid).reshape((width, height))

    #delta = offset if offset else 0
    #maxval = max(np.max(vals) - delta, -(np.min(vals) - delta))

    plt.pcolormesh(x0, x1, vals, #vmin=delta-maxval, vmax=delta+maxval,
                   cmap='jet')

    if offset is not None:
        plt.contour(x0, x1, vals, colors='black', levels=[offset])


def confusion_matrix(Y, Y_):
    Cs = sorted(set(Y) | set(Y_))
    C = len(Cs)
    mat = np.zeros((C, C), dtype='int')
    pairs = np.vstack((Y, Y_)).T

    for i, Ci in Cs:
        for j, Cj in Cs:
            mat[i][j] = (pairs == (Ci, Cj)).all(axis=1).sum()
    
    return mat


if __name__ == '__main__':
    np.random.seed(100)

    G = Random2DGaussian()
    X = G.get_sample(100)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
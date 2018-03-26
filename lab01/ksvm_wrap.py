from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import data


class KSVMWrap():
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.clf = SVC(C=param_svm_c, gamma=param_svm_gamma, probability=True)
        self.clf.fit(X, Y_)
    

    def predict(self, X):
        return self.clf.predict(X)
    

    def get_scores(self, X):
        return self.clf.predict_proba(X)
    

    def support(self):
        return self.clf.support_

if __name__ == '__main__':
    np.random.seed(100)

    X, Y_ = data.sample_gmm_2d(6, 2, 10)

    clf = KSVMWrap(X, Y_)
    
    dec_fun = lambda X: clf.get_scores(X)[:, 1]
    probs = clf.get_scores(X)
    Y = probs.argmax(axis=1)

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(dec_fun, rect)
    data.graph_data(X, Y_, Y, special=clf.support())
    
    plt.show()

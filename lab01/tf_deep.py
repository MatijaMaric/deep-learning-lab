import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data

class TFDeep:
    def __init__(self, layers, param_delta=0.1, param_lambda=0.2, activation=tf.nn.relu):
        D = layers[0]
        C = layers[-1]

        n_layers = len(layers)-1

        self.X = tf.placeholder(tf.float32, [None, D])
        self.Yoh_ = tf.placeholder(tf.float32, [None, C])

        activations = [activation] * (n_layers-1) + [tf.nn.softmax]

        H = self.X
        reg_loss = 0
        for i, (prev, next) in enumerate(zip(layers, layers[1:])):
            W = tf.Variable(tf.random_normal([next, prev]), name='W{}'.format(i))
            b = tf.Variable(tf.zeros([next]), name='b{}'.format(i))

            S = tf.matmul(H, W, transpose_b=True) + b
            H = activations[i](S)

            reg_loss += tf.nn.l2_loss(W)

        self.probs = H    

        log_probs = -tf.log(self.probs+1e-10)
        err_loss = tf.reduce_sum(self.Yoh_ * log_probs, 1)
        self.loss = tf.reduce_mean(err_loss) + param_lambda * reg_loss

        trainer = tf.train.GradientDescentOptimizer(param_delta)
        self.train_step = trainer.minimize(self.loss)
        
        self.session = tf.Session()


    def train(self, X, Yoh_, param_niter):
        
        self.session.run(tf.initialize_all_variables())
        data_dict = {self.X: X, self.Yoh_: Yoh_}

        for i in range(param_niter):
            val_loss, _ = self.session.run([self.loss, self.train_step], feed_dict=data_dict)
            if i % 100 == 0:
                print("iteration {}: loss {}".format(i, val_loss))


    def eval(self, X):
        probs = self.session.run(self.probs, {self.X: X})
        return probs


if __name__ == '__main__':
    np.random.seed(100)
    tf.set_random_seed(100)
    
    X, Y_, Yoh_ = data.sample_gmm_2d(6, 2, 10, one_hot=True)

    layers = [X.shape[1], 10, 10, Yoh_.shape[1]]
    nn = TFDeep(layers, 0.05, 1e-4, activation=tf.nn.relu)
    nn.train(X, Yoh_, 4000)

    probs = nn.eval(X)
    Y = probs.argmax(axis=1)

    dec_fun = lambda X: nn.eval(X)[:,1]
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(dec_fun, rect)
    data.graph_data(X, Y_, Y)
    plt.show()

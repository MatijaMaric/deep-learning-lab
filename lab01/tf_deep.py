import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data


class TFDeep:
    def __init__(self, layers, param_delta=0.1, param_lambda=0.2,
                 activation=tf.nn.relu, param_epsilon=1e-9):
        D = layers[0]
        C = layers[-1]

        n_layers = len(layers)-1

        self.X = tf.placeholder(tf.float32, [None, D])
        self.Yoh_ = tf.placeholder(tf.float32, [None, C])
        self.is_training = tf.placeholder(tf.bool)

        self.Ws = []
        self.bs = []

        activations = [activation] * (n_layers-1) + [tf.nn.softmax]

        H = self.X
        reg_loss = 0
        for i, (prev, next) in enumerate(zip(layers, layers[1:])):
            W = tf.Variable(tf.random_normal(
                [next, prev]), name='W{}'.format(i))
            b = tf.Variable(tf.zeros([next]), name='b{}'.format(i))

            self.Ws.append(W)
            self.bs.append(b)

            S = tf.matmul(H, W, transpose_b=True) + b
            if i != n_layers:
                BN = self.batch_norm_wrapper(S, self.is_training)
            else:
                BN = S
            H = activations[i](BN)

            reg_loss += tf.nn.l2_loss(W)

        self.probs = H

        log_probs = -tf.log(self.probs+1e-10)
        err_loss = tf.reduce_sum(self.Yoh_ * log_probs, 1)
        self.loss = tf.reduce_mean(err_loss) + param_lambda * reg_loss

        trainer = tf.train.GradientDescentOptimizer(param_delta)
        self.train_step = trainer.minimize(self.loss)

        self.session = tf.Session()

    def batch_norm_wrapper(self, inputs, is_training, decay=0.999,
                           epsilon=1e-9):

        scale = tf.Variable(tf.ones([inputs.shape[1]]))
        beta = tf.Variable(tf.zeros([inputs.shape[1]]))
        pop_mean = tf.Variable(tf.zeros([inputs.shape[1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.shape[1]]), trainable=False)

        def do_training():
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                                  pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                                                 batch_mean, batch_var, beta,
                                                 scale, epsilon)

        def do_testing():
            return tf.nn.batch_normalization(inputs,
                                             pop_mean, pop_var, beta,
                                             scale, epsilon)

        return tf.cond(self.is_training, do_training, do_testing)

    def train(self, X, Yoh_, param_niter=1000, print_step=100, verbose=True):

        self.session.run(tf.initialize_all_variables())
        data_dict = {self.X: X, self.Yoh_: Yoh_, self.is_training: True}

        for i in range(param_niter):
            val_loss, _ = self.session.run(
                [self.loss, self.train_step], feed_dict=data_dict)
            if i % print_step == 0 and verbose:
                print("iteration {}: loss {}".format(i, val_loss))

    def eval(self, X):
        probs = self.session.run(
            self.probs, {self.X: X, self.is_training: False})
        return probs

    def get_weights(self):
        return self.session.run(self.Ws)

    def _shuffle(self, X, Yoh_):
        perm = np.random.permutation(len(X))
        return X[perm], Yoh_[perm]

    def _split(self, X, Yoh_, ratio=0.8):
        X, Yoh_ = self._shuffle(X, Yoh_)
        split = int(ratio * len(X))
        return X[:split], X[split:], Yoh_[:split], Yoh_[split:]

    def train_mb(self, X, Yoh_, n_epochs=1000, batch_size=50,
                 validation_ratio=0.8, print_step=100):

        self.session.run(tf.initialize_all_variables())

        prev_loss = float('inf')
        now_loss = float('inf')

        X_train, X_val, Y_train, Y_val = self._split(X, Yoh_, validation_ratio)
        n_samples = len(X_train)
        n_batches = int(n_samples/batch_size)

        for epoch in range(n_epochs):
            X_train, Y_train = self._shuffle(X_train, Y_train)

            # training
            i = 0
            avg_loss = 0
            while i < n_samples:
                batch_X, batch_Y = X_train[i:i +
                                           batch_size], Y_train[i:i+batch_size]
                data_dict = {self.X: batch_X,
                             self.Yoh_: batch_Y, self.is_training: True}
                val_loss, _ = self.session.run(
                    [self.loss, self.train_step], feed_dict=data_dict)

                avg_loss += val_loss / n_batches
                i += batch_size

            # validation
            data_dict = {self.X: X_val, self.Yoh_: Y_val,
                         self.is_training: False}
            [val_loss] = self.session.run([self.loss], feed_dict=data_dict)
            now_loss = min(now_loss, val_loss)
            if epoch % 50 == 0:
                if now_loss > prev_loss:
                    print("early stopping on epoch #{}".format(epoch))
                    print("epoch {}: avg_loss {}, val_loss {}".format(
                        epoch, avg_loss, val_loss))
                    break
                prev_loss = now_loss
                now_loss = float('inf')

            if epoch % print_step == 0:
                print("epoch {}: avg_loss {}, val_loss {}".format(
                    epoch, avg_loss, val_loss))


if __name__ == '__main__':
    np.random.seed(100)
    tf.set_random_seed(100)

    X, Y_, Yoh_ = data.sample_gmm_2d(6, 2, 10, one_hot=True)

    layers = [X.shape[1], 10, 10, Yoh_.shape[1]]
    nn = TFDeep(layers, 0.05, 1e-4, activation=tf.nn.relu)
    nn.train(X, Yoh_, 4000)

    probs = nn.eval(X)
    Y = probs.argmax(axis=1)

    def dec_fun(X): return nn.eval(X)[:, 1]
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(dec_fun, rect)
    data.graph_data(X, Y_, Y)
    plt.show()

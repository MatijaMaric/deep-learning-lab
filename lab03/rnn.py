import numpy as np
from sklearn.metrics import log_loss


def softmax(o):
    max_ = np.max(o, axis=2)
    max_ = max_[:, :, np.newaxis]
    exp_ = np.exp(o - max_)
    return exp_ / np.sum(exp_, axis=2, keepdims=True)


class RNN:

    def __init__(self, hidden_size, sequence_length,
                 vocab_size, learning_rate):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        self.U = np.random.normal(   # input projection
            size=[vocab_size, hidden_size], scale=1.0 / np.sqrt(hidden_size))
        # ... hidden-to-hidden projection
        self.W = np.random.normal(
            size=[hidden_size, hidden_size], scale=1.0 / np.sqrt(hidden_size))
        self.b = np.zeros([1, hidden_size])  # ... input bias

        self.V = np.random.normal(  # ... output projection
            size=[hidden_size, vocab_size], scale=1.0 / np.sqrt(vocab_size))
        self.c = np.zeros([1, vocab_size])  # ... output bias

        # memory of past gradients - rolling sum of squares for Adagrad
        self.memory_U, self.memory_W, self.memory_V = np.zeros_like(
            self.U), np.zeros_like(self.W), np.zeros_like(self.V)
        self.memory_b, self.memory_c = np.zeros_like(
            self.b), np.zeros_like(self.c)

    def rnn_step_forward(self, x, h_prev, U=None, W=None, b=None):
        # A single time step forward of a recurrent neural network with a
        # hyperbolic tangent nonlinearity.

        # x - input data (minibatch size x input dimension)
        # h_prev - previous hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)

        U = self.U if U is None else U
        W = self.W if W is None else W
        b = self.b if b is None else b

        h_current = np.tanh(np.dot(h_prev, W) + np.dot(x, U) + b)
        cache = (x, h_prev, h_current)

        # return the new hidden state and a tuple of
        # values needed for the backward step

        return h_current, cache

    def rnn_forward(self, x, h0, U=None, W=None, b=None):
        # Full unroll forward of the recurrent neural network with a
        # hyperbolic tangent nonlinearity

        # x - input data for the whole time-series
        #  (minibatch size x sequence_length x input dimension)
        # h0 - initial hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)

        U = self.U if U is None else U
        W = self.W if W is None else W
        b = self.b if b is None else b

        h, cache = [], []
        current_h, current_cache = h0, None

        sequences = x.transpose(1, 0, 2)

        # return the hidden states for the whole time series
        #  (T+1) and a tuple of values needed for the backward step

        for i in range(self.sequence_length):
            step = x[:, i, :]
            current_h, current_cache = self.rnn_step_forward(
                step, current_h, U, W, b)
            h.append(current_h)
            cache.append(current_cache)

        h = np.array(h).transpose(1, 0, 2)
        return h, cache

    def rnn_step_backward(self, grad_next, cache):
        # A single time step backward of a recurrent neural network with a
        # hyperbolic tangent nonlinearity.

        # grad_next - upstream gradient of the loss with
        #  respect to the next hidden state and current output
        # cache - cached information from the forward pass

        x, h_prev, h_current = cache

        dz = grad_next * (1 - h_current**2)

        dh_prev = np.dot(dz, self.W.T)
        dU = np.dot(x.T, dz)
        dW = np.dot(h_prev.T, dz)
        db = np.sum(dz, axis=0)

        # compute and return gradients with respect to each parameter
        # HINT: you can use the chain rule to compute the derivative of the
        # hyperbolic tangent function and use it to compute the gradient
        # with respect to the remaining parameters

        return dh_prev, dU, dW, db

    def rnn_backward(self, dh, cache):
        # Full unroll forward of the recurrent neural network with a
        # hyperbolic tangent nonlinearity

        dU, dW, db = np.zeros_like(self.U), np.zeros_like(
            self.W), np.zeros_like(self.b)

        dh_transposed = dh.transpose(1, 0, 2)

        dh_prev = np.zeros_like(dh_transposed[0])

        for dh_t, cache_t in reversed(list(zip(dh_transposed, cache))):
            dh_prev, dU_t, dW_t, db_t = self.rnn_step_backward(
                dh_t + dh_prev, cache_t)
            dU += dU_t
            dW += dW_t
            db += db_t

        # compute and return gradients with respect to each parameter
        # for the whole time series.
        # Why are we not computing the gradient with respect to inputs (x)?

        dU = np.clip(dU, -5, 5)
        dW = np.clip(dW, -5, 5)
        db = np.clip(db, -5, 5)

        return dU, dW, db

    def output(self, h, V=None, c=None):
        # Calculate the output probabilities of the network

        V = self.V if V is None else V
        c = self.c if c is None else c

        logits = np.dot(h, V) + c
        return softmax(logits)

    def output_loss_and_grads(self, h, y, V=None, c=None):
        # Calculate the loss of the network for each of the outputs

        # h - hidden states of the network for each timestep.
        #     the dimensionality of h is
        #  (batch size x sequence length x hidden size (the initial
        #  state is irrelevant for the output)
        # V - the output projection matrix of
        #  dimension hidden size x vocabulary size
        # c - the output bias of dimension vocabulary size x 1
        # y - the true class distribution - a tensor of dimension
        #     batch_size x sequence_length x vocabulary
        #  size - you need to do this conversion prior to
        #     passing the argument. A fast way to create a one-hot vector from
        #     an id could be something like the following code:

        #   y[batch_id][timestep] = np.zeros((vocabulary_size, 1))
        #   y[batch_id][timestep][batch_y[timestep]] = 1

        #     where y might be a list or a dictionary.

        V = self.V if V is None else V
        c = self.c if c is None else c

        batch_size = h.shape[0]

        # calculate the output (o) - unnormalized log probabilities of classes
        # calculate yhat - softmax of the output
        # calculate the cross-entropy loss
        # calculate the derivative of the cross-entropy
        #  softmax loss with respect to the output (o)
        # calculate the gradients with respect to the output parameters V and c
        # calculate the gradients with respect to the hidden layer h

        o = self.output(h, V, c)
        loss = log_loss(y.reshape(-1, self.vocab_size),
                        o.reshape(-1, self.vocab_size))

        dO = o - y

        dh, dV, dc = [], np.zeros_like(V), np.zeros_like(c)

        dO_t = dO.transpose(1, 0, 2)
        h_t = h.transpose(1, 0, 2)

        for dO_i, h_i in zip(dO_t, h_t):
            dV += np.dot(h_i.T, dO_i) / batch_size
            dc += np.average(dO_i, axis=0)
            dh.append(np.dot(dO_i, V.T))

        dh = np.array(dh).transpose(1, 0, 2)

        return loss, dh, dV, dc

    def update(self, dU, dW, db, dV, dc):

        # update memory matrices
        # perform the Adagrad update of parameters

        epsilon = 1e-7

        self.memory_U += np.square(dU)
        self.memory_W += np.square(dW)
        self.memory_b += np.square(db)
        self.memory_V += np.square(dV)
        self.memory_c += np.square(dc)

        self.U -= self.learning_rate * dU / np.sqrt(self.memory_U + epsilon)
        self.W -= self.learning_rate * dW / np.sqrt(self.memory_W + epsilon)
        self.b -= self.learning_rate * db / np.sqrt(self.memory_b + epsilon)
        self.V -= self.learning_rate * dV / np.sqrt(self.memory_V + epsilon)
        self.c -= self.learning_rate * dc / np.sqrt(self.memory_c + epsilon)

    def step(self, h0, x_oh, y_oh):
        h, cache = self.rnn_forward(x_oh, h0)
        loss, dh, dV, dc = self.output_loss_and_grads(h, y_oh)
        dU, dW, db = self.rnn_backward(dh, cache)

        dU = np.clip(dU, -5, 5)
        dW = np.clip(dW, -5, 5)
        db = np.clip(db, -5, 5)
        dV = np.clip(dV, -5, 5)
        dc = np.clip(dc, -5, 5)

        self.update(dU, dW, db, dV, dc)

        return loss, h[:, -1, :]

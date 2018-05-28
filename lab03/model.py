from dataset import Dataset
from rnn import RNN
import numpy as np


def run_language_model(dataset, max_epochs, seed, hidden_size=100,
                       sequence_length=30, learning_rate=1e-1,
                       sample_every=100, n_sample=200):

    vocab_size = len(dataset.sorted_chars)
    # initialize the recurrent network
    rnn = RNN(hidden_size, sequence_length, vocab_size, learning_rate)

    current_epoch = 0
    batch = 0

    h0 = np.zeros((dataset.batch_size, hidden_size))

    cum_loss = 0

    while current_epoch < max_epochs:
        e, x, y = dataset.next_minibatch()

        if e:
            current_epoch += 1
            h0 = np.zeros_like(h0)
            # why do we reset the hidden state here?

        # One-hot transform the x and y batches
        x_oh, y_oh = one_hot(x, vocab_size), one_hot(y, vocab_size)

        # Run the recurrent network on the current batch
        # Since we are using windows of a short length of characters,
        # the step function should return the hidden state at the end
        # of the unroll. You should then use that hidden state as the
        # input for the next minibatch. In this way, we artificially
        # preserve context between batches.
        loss, h0 = rnn.step(h0, x_oh, y_oh)
        cum_loss = 0.9 * cum_loss + 0.1 * loss

        if batch % sample_every == 0:
            # run sampling (2.2)
            print("Batch: %d/%d, loss: %f, epoch: %d/%d, avg loss: %f" %
                  (batch % dataset.num_batches, dataset.num_batches, loss,
                   current_epoch, max_epochs, cum_loss))
            sample_ = sample(rnn, seed, n_sample, dataset)
            print("===================")
            print(sample_)
            print("===================")
        batch += 1

    return rnn


def one_hot(batch, size):

    def _oh(x, size):
        x_oh = np.zeros((x.shape[0], size))
        x_oh[np.arange(x.shape[0]), x] = 1
        return x_oh

    if batch.ndim == 1:
        return _oh(batch, size)
    else:
        return np.array([_oh(s, size) for s in batch])


def sample(rnn, seed, n_sample, dataset):
    # h0, seed_onehot, sample = None, None, None
    # inicijalizirati h0 na vektor nula
    # seed string pretvoriti u one-hot reprezentaciju ulaza

    h0 = np.zeros([1, rnn.hidden_size])
    seed_encoded = dataset.encode(seed)
    seed_onehot = one_hot(seed_encoded, rnn.vocab_size)

    h = h0
    for char in seed_onehot:
        h, _ = rnn.rnn_step_forward(char[np.newaxis, :], h)

    sample = np.zeros((n_sample, ), dtype=np.int32)
    sample[:len(seed)] = seed_encoded

    for i in range(len(seed), n_sample):
        out = rnn.output(h[np.newaxis, :, :])
        sample[i] = np.random.choice(np.arange(out.shape[-1]), p=out.ravel())

        next_input = np.zeros([1, rnn.vocab_size])
        next_input[0, sample[i]] = 1

        h, _ = rnn.rnn_step_forward(next_input, h)

    return "".join(dataset.decode(sample))


if __name__ == "__main__":
    dataset = Dataset()
    dataset.preprocess("data/selected_conversations.txt")
    dataset.create_minibatches()

    run_language_model(dataset, 50, sequence_length=dataset.sequence_length,
                       seed="HAN:\nIs that good or bad?\n\n")

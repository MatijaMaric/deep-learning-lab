import tensorflow as tf
import numpy as np
import pickle
import os
import math
import skimage as ski
import skimage.io
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

DATA_DIR = '/home/matija/datasets/CIFAR10'
save_dir = '/home/matija/FER/deep-learning/out/tf_cifar'
img_height = 32
img_width = 32
num_channels = 3
num_classes = 10
lr = 1e-2
batch_size = 200
loss_step = 5
draw_step = 100
acc_step = 50
max_epochs = 1  # 35
weight_decay = 1e-4

valid_size = 5000
loss_batch_size = 5000


def to_one_hot(y, C=10):
    one_hot = [0] * C
    one_hot[y] = 1
    return one_hot


def shuffle_data(data_x, data_y):
    indices = np.arange(data_x.shape[0])
    np.random.shuffle(indices)
    shuffled_data_x = np.ascontiguousarray(data_x[indices])
    shuffled_data_y = np.ascontiguousarray(data_y[indices])
    return shuffled_data_x, shuffled_data_y


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict


def load_dataset():
    train_x = np.ndarray((0, img_height * img_width *
                          num_channels), dtype=np.float32)
    train_y = []
    for i in range(1, 6):
        subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
        train_x = np.vstack((train_x, subset['data']))
        train_y += subset['labels']
    train_x = train_x.reshape(
        (-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
    train_y = np.array(train_y, dtype=np.int32)

    subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
    test_x = subset['data'].reshape(
        (-1, num_channels,
         img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
    test_y = np.array(subset['labels'], dtype=np.int32)

    class_names = np.array(unpickle(os.path.join(
        DATA_DIR, 'batches.meta'))['label_names'])

    train_x, train_y = shuffle_data(train_x, train_y)
    valid_x = train_x[:valid_size, ...]
    valid_y = train_y[:valid_size, ...]
    train_x = train_x[valid_size:, ...]
    train_y = train_y[valid_size:, ...]
    data_mean = train_x.mean((0, 1, 2))
    data_std = train_x.std((0, 1, 2))

    train_x = (train_x - data_mean) / data_std
    valid_x = (valid_x - data_mean) / data_std
    test_x = (test_x - data_mean) / data_std

    train_x = train_x.reshape([-1, 32, 32, 3])
    valid_x = valid_x.reshape([-1, 32, 32, 3])
    test_x = test_x.reshape([-1, 32, 32, 3])
    train_y = np.array(list(map(to_one_hot, train_y)))
    test_y = np.array(list(map(to_one_hot, test_y)))
    valid_y = np.array(list(map(to_one_hot, valid_y)))

    return train_x, train_y, test_x, test_y, valid_x, valid_y, data_mean, data_std, class_names


def conv2d(inputs, filters, kernel_size=[5, 5], activation=tf.nn.relu,
           regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
           name=None):
    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding='same',
        activation=activation,
        kernel_regularizer=regularizer,
        name=name
    )


def max_pool(inputs, pool_size=[3, 3], strides=2, name=None):
    return tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=pool_size,
        strides=strides,
        name=name
    )


def dense(inputs, units, activation=tf.nn.relu,
          regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
          name=None):
    return tf.layers.dense(
        inputs=inputs,
        units=units,
        activation=activation,
        kernel_regularizer=regularizer,
        name=name
    )


def build_model(inputs):
    input_layer = tf.reshape(inputs, [-1, 32, 32, 3])

    conv1 = conv2d(input_layer, 16, name="conv1")
    pool1 = max_pool(conv1, name="pool1")

    conv2 = conv2d(pool1, 32, name="conv2")
    pool2 = max_pool(conv2, name="pool2")

    flat = tf.contrib.layers.flatten(pool2)

    dense1 = dense(flat, 256, name="dense1")
    dense2 = dense(dense1, 128, name="dense2")
    logits = dense(dense2, 10, activation=None,
                   regularizer=None, name="logits")

    return logits


def get_batches(x, y, batch_size):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    for i in range(0, x.shape[0] - batch_size + 1, batch_size):
        indices_range = indices[i:i + batch_size]
        yield x[indices_range], y[indices_range]


def draw_conv_filters(session, epoch, step, name, layer, save_dir):
    weights = session.run(tf.trainable_variables()[layer]).copy()
    num_filters = weights.shape[3]
    num_channels = weights.shape[2]
    k = weights.shape[0]
    assert weights.shape[0] == weights.shape[1]
    weights -= weights.min()
    weights /= weights.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols-1) * border
    height = rows * k + (rows-1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r+k, c:c+k, :] = weights[:, :, :, i]
    filename = '%s_epoch_%02d_step_%06d.png' % (name, epoch, step)
    ski.io.imsave(os.path.join(save_dir, filename), img)


def evaluate(sess, logits, loss, input_x, input_y, class_names):
    output = np.array()
    for batch in get_batches(train_x, train_y, loss_batch_size):
        out = logits.eval(feed_dict={x: batch[0],
                                     y_: batch[1]})
        output = np.vstack((output, out))

        y_pred = np.argmax(output, 1)
        y_true = np.argmax(input_y, 1)

        cfm = confusion_matrix(y_true, y_pred)
        print(cfm)


def plot_training_progress(save_dir, data):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize(16, 8))

    linewidth = 2
    legend_size = 10
    train_color = 'm'
    val_color = 'c'

    num_points = len(data['train_loss'])
    x_data = np.linspace(1, num_points, num_points)
    ax1.set_title('Cross-entropy loss')
    ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='train')
    ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
             linewidth=linewidth, linestyle='-', label='validation')
    ax1.legend(loc='upper right', fontsize=legend_size)
    ax2.set_title('Average class accuracy')
    ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='train')
    ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
             linewidth=linewidth, linestyle='-', label='validation')
    ax3.plot(x_data, data['lr'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='learning_rate')
    ax3.legend(loc='upper left', fontsize=legend_size)

    save_path = os.path.join(save_dir, 'training_plot.pdf')
    print('Plotting in: ', save_path)
    plt.savefig(save_path)


def main(_):
    train_x, train_y, test_x, test_y, valid_x, valid_y, mean, std, class_names = load_dataset()

    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y_ = tf.placeholder(tf.int32, [None, 10])

    y_conv = build_model(x)

    err_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    )
    reg_loss = tf.losses.get_regularization_loss()

    loss = err_loss + weight_decay * reg_loss

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        0.01, global_step, 500, 0.95, staircase=True)
    train_step = tf.train.MomentumOptimizer(
        learning_rate, 0.9).minimize(loss, global_step=global_step)

    # train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    num_examples = train_x.shape[0]
    num_batches = num_examples // batch_size

    data = {}

    data['train_loss'] = []
    data['valid_loss'] = []
    data['train_acc'] = []
    data['valid_acc'] = []
    data['lr'] = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, max_epochs+1):
            for (i, batch) in enumerate(get_batches(train_x, train_y,
                                                    batch_size)):
                train_step.run(feed_dict={
                    x: batch[0],
                    y_: batch[1]
                })
                if i % loss_step == 0:
                    batch_acc, batch_loss = sess.run([accuracy, loss],
                                                     feed_dict={x: batch[0],
                                                                y_: batch[1]})
                    print('epoch %d: step %d/%d, batch loss=%g, batch acc=%g' %
                          (epoch, i*batch_size, num_examples,
                           batch_loss, batch_acc), end="\r", flush=True)
                if i % draw_step == 0:
                    draw_conv_filters(
                        sess, epoch, i*batch_size, "conv1", 0, save_dir
                    )
            print("")

            train_loss, train_acc = 0, 0
            did = 0
            print('epoch %d: training loss/accuracy %d/%d' %
                  (epoch, did, num_examples), end="\r", flush=True)
            for batch in get_batches(train_x, train_y, loss_batch_size):
                batch_acc, batch_loss = sess.run([accuracy, loss],
                                                 feed_dict={x: batch[0],
                                                            y_: batch[1]})
                train_acc += batch_acc
                train_loss += batch_loss

                did += loss_batch_size

                print('epoch %d: training loss/accuracy %d/%d' %
                      (epoch, did, num_examples), end="\r", flush=True)

            train_loss /= num_examples / loss_batch_size
            train_acc /= num_examples / loss_batch_size

            print('epoch %d: training loss %g, training accuracy %g' %
                  (epoch, train_loss, train_acc))

            valid_acc, valid_loss = sess.run([accuracy, loss], feed_dict={
                x: valid_x,
                y_: valid_y
            })

            data['train_loss'] += [train_loss]
            data['train_acc'] += [train_acc]
            data['valid_loss'] += [valid_loss]
            data['valid_acc'] += [valid_acc]
            data['lr'] += [sess.run(learning_rate)]

            print('epoch %d: validation loss %g, validation accuracy %g' %
                  (epoch, valid_loss, valid_acc))

        test_acc, test_loss = 0, 0
        num_test_examples = test_x.shape[0]
        did = 0
        print('epoch %d: test loss/accuracy %d/%d' %
              (epoch, did, num_test_examples))
        for batch in get_batches(test_x, test_y, loss_batch_size):
            batch_acc, batch_loss = sess.run([accuracy, loss],
                                             feed_dict={x: batch[0],
                                                        y_: batch[1]})
            test_acc += batch_acc
            test_loss += batch_loss

            did += loss_batch_size

            print('epoch %d: test loss/accuracy %d/%d' %
                  (epoch, did, num_test_examples))

        test_loss /= num_test_examples / loss_batch_size
        test_acc /= num_test_examples / loss_batch_size

        print('epoch %d: test loss %g, test accuracy %g' %
              (epoch, test_loss, test_acc))

        # evaluate(sess, y_conv, loss, test_x, test_y, class_names)
        # plot_training_progress(save_dir, data)


if __name__ == '__main__':
    tf.app.run(main=main)

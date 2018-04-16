import tensorflow as tf
import numpy as np
import tempfile
import os
import math
import skimage as ski
import skimage.io

from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = '/home/matija/datasets/MNIST'
save_dir = '/home/matija/FER/deep-learning/out/tf_l2'

max_epochs = 8
batch_size = 50
weight_decay = 1e-2
lr_policy = {1: 1e-1, 3: 1e-2, 5: 1e-3, 7: 1e-4}

loss_step = 5
draw_step = 100
acc_step = 50


def draw_conv_filters(session, epoch, step, name, layer, save_dir):
    weights = session.run(tf.trainable_variables()[layer])
    k, k, C, num_filters = weights.shape

    w = weights.copy().swapaxes(0, 3).swapaxes(1, 2)
    w = w.reshape(num_filters, C, k, k)

    w -= w.min()
    w /= w.max()

    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols-1) * border
    height = rows * k + (rows-1) * border
    for j in range(1):
        img = np.zeros([height, width])
        for i in range(num_filters):
            r = int(i / cols) * (k + border)
            c = int(i % cols) * (k + border)
            img[r:r+k, c:c+k] = w[i, j]

    img = img.reshape(height, width)
    filename = '%s_epoch_%02d_step_%06d_input_%03d.png' % (
        name, epoch, step, i
    )
    ski.io.imsave(os.path.join(save_dir, filename), img)


def conv2d(inputs, filters, name, kernel_size=[5, 5], activation=None):
    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation=activation,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(
            scale=weight_decay))


def max_pool(inputs, name, pool_size=2, strides=2):
    return tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=pool_size,
        strides=strides)


def dense(inputs, units, name, activation=tf.nn.relu,
          kernel_regularizer=tf.contrib.layers.l2_regularizer(
              scale=weight_decay)):
    return tf.layers.dense(
        inputs=inputs,
        units=units,
        activation=activation,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        kernel_regularizer=kernel_regularizer)


def build_model(inputs):
    input_layer = tf.reshape(inputs, [-1, 28, 28, 1])

    conv1 = conv2d(input_layer, 16, name='conv1')
    pool1 = max_pool(conv1, name='pool1')

    h1 = tf.nn.relu(pool1)

    conv2 = conv2d(h1, 32, name='conv2')
    pool2 = max_pool(conv2, name='pool2')

    h2 = tf.nn.relu(pool2)

    flat = tf.contrib.layers.flatten(h2, scope='flatten')

    dense1 = dense(flat, 512, name='dense1')
    logits = dense(dense1, 10, activation=None,
                   kernel_regularizer=None, name='logits')

    return logits


def main(_):
    mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])

    y_ = tf.placeholder(tf.int64, [None, 10])

    y_conv = build_model(x)

    err_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    reg_loss = tf.losses.get_regularization_loss()

    loss = err_loss + weight_decay * reg_loss

    lr = tf.placeholder(tf.float32)

    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    num_examples = mnist.train.num_examples
    num_batches = num_examples // batch_size

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, max_epochs+1):
            if epoch in lr_policy:
                lr_val = lr_policy[epoch]
            for i in range(num_batches):
                batch = mnist.train.next_batch(batch_size)
                train_step.run(feed_dict={
                    x: batch[0],
                    y_: batch[1],
                    lr: lr_val
                })
                if i % loss_step == 0:
                    batch_loss = loss.eval(feed_dict={
                        x: batch[0],
                        y_: batch[1]
                    })
                    print('epoch %d: step %d/%d, batch loss=%g' %
                          (epoch, i*batch_size, num_examples, batch_loss))
                if i % draw_step == 0:
                    draw_conv_filters(
                        sess, epoch, i*batch_size, "conv1", 0, save_dir)
                if i % acc_step == 0:
                    train_acc = accuracy.eval(feed_dict={
                        x: batch[0],
                        y_: batch[1]
                    })
                    print("Train accuracy = %.2f" % train_acc)

            valid_acc = accuracy.eval(feed_dict={
                x: mnist.validation.images,
                y_: mnist.validation.labels
            })

            valid_loss = loss.eval(feed_dict={
                x: mnist.validation.images,
                y_: mnist.validation.labels
            })

            print('epoch %d: validation loss %g, validation accuracy %g' %
                  (epoch, valid_loss, valid_acc))

        test_acc = accuracy.eval(feed_dict={
            x: mnist.test.images,
            y_: mnist.test.labels
        })

        test_loss = loss.eval(feed_dict={
            x: mnist.test.images,
            y_: mnist.test.labels
        })

        print('test loss: %g, test accuracy: %g' % (test_loss, test_acc))


if __name__ == '__main__':
    tf.app.run(main=main)

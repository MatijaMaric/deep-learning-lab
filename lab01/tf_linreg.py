import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = tf.placeholder(tf.float32, [None])
Y_ = tf.placeholder(tf.float32, [None])
a = tf.Variable(0.0)
b = tf.Variable(0.0)

Y = a * X + b

loss = (Y-Y_)**2

trainer = tf.train.GradientDescentOptimizer(0.1)
train_op = trainer.minimize(loss)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

Xs = np.array([1, 2])
Ys = np.array([3, 5])


for i in range(100):
    val_loss, _, val_a, val_b = sess.run([loss, train_op, a, b],
        feed_dict={X: [1,2], Y_: [3,5]})
    print(i, val_loss, val_a, val_b)

plt.plot(Xs, val_a * Xs + val_b)
plt.scatter(Xs, Ys, marker='o')
plt.show()
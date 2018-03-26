import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

f = lambda x: x + 3

N = 100
Xs = np.random.uniform(-10, 10, N)
Ys = f(Xs) + np.random.normal(0, 0.1, N)
Xs, Ys = Xs.reshape(-1, 1), Ys.reshape(-1, 1)

X  = tf.placeholder(tf.float32, [None, 1])
Y_ = tf.placeholder(tf.float32, [None, 1])
a = tf.Variable(0.0)
b = tf.Variable(0.0)

Y = a*X + b
loss  = 1./(2*N) * (Y-Y_)**2

trainer = tf.train.GradientDescentOptimizer(0.01)
train_op = trainer.minimize(loss)
grads = trainer.compute_gradients(loss, [a, b])
optimize = trainer.apply_gradients(grads)
grads = tf.Print(grads, [grads], 'Status:')

grad_a = (1/N) * tf.matmul(Y-Y_,  X, transpose_a=True)
grad_b = (1/N) * tf.reduce_sum(Y-Y_)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
plt.scatter(Xs, Ys, marker='o')

for i in range(1000):
    val_loss, val_grads, val_grad_a, val_grad_b = sess.run([loss, grads, grad_a, grad_b], feed_dict={X: Xs, Y_: Ys})
    _, val_a, val_b = sess.run([train_op, a, b], feed_dict={X: Xs, Y_: Ys})

    if i% 100 == 0:
        print(val_a, val_b, val_loss.sum())
        print(val_grads)
        print(val_grad_a, val_grad_b)

plt.plot(Xs, val_a*Xs + val_b)
plt.show()

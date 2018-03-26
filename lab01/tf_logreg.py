import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data

class TFLogreg:
	def __init__(self, D, C, param_delta=0.5, param_lambda=0.1):
		"""Arguments:
			- D: dimensions of each datapoint 
			- C: number of classes
			- param_delta: training step
		"""

		# definicija podataka i parametara:
		# definirati self.X, self.Yoh_, self.W, self.b
		# ...

		self.X = tf.placeholder(tf.float32, [None, D])
		self.Yoh_ = tf.placeholder(tf.float32, [None, C])
		self.W = tf.Variable(tf.random_normal([C, D]))
		self.b = tf.Variable(tf.zeros([C]))

		# formulacija modela: izračunati self.probs
		#   koristiti: tf.matmul, tf.nn.softmax
		# ...

		scores = tf.matmul(self.X, self.W, transpose_b=True) + self.b
		self.probs = tf.nn.softmax(scores)

		# formulacija gubitka: self.loss
		#   koristiti: tf.log, tf.reduce_sum, tf.reduce_mean
		# ...

		log_probs = -tf.log(self.probs)
		err_loss = tf.reduce_sum(self.Yoh_ * log_probs, 1)
		reg_loss = tf.nn.l2_loss(self.W)
		self.loss = tf.reduce_mean(err_loss) + param_lambda * reg_loss

		# formulacija operacije učenja: self.train_step
		#   koristiti: tf.train.GradientDescentOptimizer,
		#              tf.train.GradientDescentOptimizer.minimize
		# ...

		trainer = tf.train.GradientDescentOptimizer(param_delta)
		self.train_step = trainer.minimize(self.loss)

		# instanciranje izvedbenog konteksta: self.session
		#   koristiti: tf.Session
		# ...

		self.session = tf.Session()

	def train(self, X, Yoh_, param_niter):
		"""Arguments:
			- X: actual datapoints [NxD]
			- Yoh_: one-hot encoded labels [NxC]
			- param_niter: number of iterations
		"""
		# incijalizacija parametara
		#   koristiti: tf.initialize_all_variables
		# ...

		self.session.run(tf.initialize_all_variables())
		data_dict = {self.X: X, self.Yoh_: Yoh_}

		# optimizacijska petlja
		#   koristiti: tf.Session.run
		# ...
		for i in range(param_niter):
			val_loss, _ = self.session.run([self.loss, self.train_step], feed_dict=data_dict)
			if i % 100 == 0:
				print("iteration {}: loss {}".format(i, val_loss))

	def eval(self, X):
		"""Arguments:
			- X: actual datapoints [NxD]
			Returns: predicted class probabilites [NxC]
		"""
		#   koristiti: tf.Session.run
		probs = self.session.run(self.probs, {self.X: X})
		return probs

if __name__ == '__main__':
	np.random.seed(100)
	tf.set_random_seed(100)

	X, Y_, Yoh_ = data.sample_gauss_2d(3, 100, one_hot=True)

	_, D = X.shape
	_, C = Yoh_.shape

	tflr = TFLogreg(D, C, 0.1, 0.25)
	tflr.train(X, Yoh_, 1000)

	probs = tflr.eval(X)

	Y = probs.argmax(axis=1)
	dec_fun = lambda X: tflr.eval(X).argmax(axis=1)

	rect = (np.min(X, axis=0), np.max(X, axis=0))

	data.graph_surface(dec_fun, rect)
	data.graph_data(X, Y_, Y)
	plt.show()

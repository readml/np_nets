import numpy as np

class ActivationFunc(object):
	"""Base class for activation functions"""
	def __init__(self, activation_function, deriv):
		self.activation_function = activation_function
		self.deriv = deriv

	def evaluate_func(self, inputs):
		return np.vectorize(self.activation_function)(inputs)

	def eval_deriv_func(self, inputs):
		return np.vectorize(self.deriv)(inputs)

class Sigmoid(ActivationFunc):
	def __init__(self):
		super(Sigmoid, self).__init__(self.sigmoid, self.sigmoid_deriv)

	def sigmoid(self, x):
		return (1.0/(1.0+np.exp(-x)))

	def sigmoid_deriv(self, x):
		return x*(1-x)

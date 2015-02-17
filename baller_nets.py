import numpy as np
import baller_act_fcns as act_funcs
from scipy.optimize import fmin_cg


class Layer():
	def __init__(self, units, activation_function, is_last_layer = False):
		"""Vectorized implemenation of a layer"""
		self.units = units #number of units		
		self.activation_function = activation_function #should act on a numpy vector and return one
		self.is_last_layer = is_last_layer
		self._init_activations()
		self._init_activation_errors()

	def _init_activations(self):
		if self.is_last_layer:
			self.activations = np.ones((self.units,1))
		else: 
			self.activations = np.ones((self.units+1,1)) #for the bias
		
	def _init_activation_errors(self):
		if self.is_last_layer:
			self.activation_errors = np.zeros((self.units,1))
		else: 
			self.activation_errors = np.zeros((self.units+1,1))

	def calculate_activations(self, prior_layer_activation, weight_matrix):
		"""returns activations as a vector"""
		summed_input = np.dot(weight_matrix, prior_layer_activation)
		if self.is_last_layer:
			self.activations[:] = self.activation_function.evaluate_func(summed_input)
		else:
			self.activations[:-1] = self.activation_function.evaluate_func(summed_input)
		return self.activations
		
	def calc_deriv_activation(self):
		"""calculate the derivative of the activation for this layer"""
		return self.activation_function.eval_deriv_func(self.activations)

	def calc_error(self, next_layer_acts, weight_matrix):
		"""look at the next layer or outputs to calc the error of the activations"""
		if self.is_last_layer:
			self.activation_errors = self.activations - next_layer_acts #should be replaced with proper error func
		else:
			a = np.dot(np.transpose(weight_matrix),next_layer_acts)
			b = self.calc_deriv_activation()
			self.activation_errors = np.multiply(a,b)
		return self.activation_errors

class Network():
	def __init__(self, num_units, act_list, lam):
		self.num_units = num_units
		self.act_list = act_list
		self.layer_list = []
		self.lam = lam
		for num, act in zip(self.num_units[:-1], self.act_list[:-1]):
			self.layer_list.append(Layer(num, act))		
		self.layer_list.append(Layer(num_units[-1], act_list[-1], is_last_layer = True))
		self._init_weights_random()
		self._init_zero_accumulators() #set them to zero

	def _init_weights_random(self):
		"""initialize network weights in each layer"""
		self.weight_matrices = []
		self.shapes_list = []
		for i in range(len(self.layer_list)-1):			
			self.weight_matrices.append(np.random.rand(self.num_units[i+1], self.num_units[i]+1)) #succeeding layer, current layer with the bias
			self.shapes_list.append(np.shape(self.weight_matrices[i]))

	def _init_weights_as_one(self):
		"""initialize network weights in each layer"""
		self.weight_matrices = []
		self.shapes_list = []
		for i in range(len(self.layer_list)-1):
			self.weight_matrices.append(np.ones((self.num_units[i+1], self.num_units[i]+1)))
			self.shapes_list.append(np.shape(self.weight_matrices[i]))

	def _init_zero_accumulators(self):
		"""initialize network weights in each layer"""
		self.accumulators = []
		for i in range(len(self.layer_list)-1):
			self.accumulators.append(np.zeros((self.num_units[i+1], self.num_units[i]+1))) #succeeding layer, current layer with the bias

	def _unroll_matrices(self, matrices):
		unrolled = []
		for matrix in matrices:
			unrolled = np.hstack([unrolled, np.ravel(matrix)])
		return unrolled

	def _reform_matrices(self, vector):
		matrices = []
		vector_index = 0
		for rows, columns in self.shapes_list:
			matrix_size = rows*columns
			matrices.append(np.reshape(vector[vector_index:vector_index+matrix_size], (rows, columns)))
			vector_index += matrix_size
		return matrices

	def train_net(self, X, Y):
		"""np matrix X of input data and np matrix Y of output data, rows are samples, columns are features"""
		self.X = X
		self.Y = Y
		self._init_weights_random
		init_weights_vector = self._unroll_matrices(self.weight_matrices)
		solution = fmin_cg(self.evaluate_cost, init_weights_vector, 
				fprime = self.calc_error_derivs, maxiter = 400)
		return solution

	def predict(self, X, weights_vector):
		self.X = X
		self.weight_matrices = self._reform_matrices(weights_vector)
		
		samples = X.shape[0]
		Y = np.zeros((samples, len(self.layer_list[-1].activations)))
		for i in range(samples):
			self.forward_prop(X[i,:])
			Y[i] = self.layer_list[-1].activations

		return Y

	def calc_error_derivs(self, weights_vector):
		"""find the dCost/dweights"""
		X = self.X
		Y = self.Y
		lam = self.lam
		self.weight_matrices = self._reform_matrices(weights_vector)
		self._init_zero_accumulators() #set them to zero
		samples = X.shape[0]
		for i in range(samples):
			self.forward_prop(X[i,:])
			self.back_prop(Y[i,:])

		# error_derivs is the list of matrix derivatives of cost fcn wrt the weight matrices
		error_derivs = []
		for W in self.weight_matrices:
			error_derivs.append(lam*W) # the regularization

		for i in range(len(error_derivs)):
			error_derivs[i][:,-1] = np.zeros((error_derivs[i].shape[0],))
			error_derivs[i] += self.accumulators[i]/float(samples)

		return self._unroll_matrices(error_derivs)

	def evaluate_cost(self, weights_vector):
		"""Evaluate prediction cost, need to pass into weight matrices for the gradient descent code"""
		X = self.X
		Y = self.Y
		lam = self.lam
		self.weight_matrices = self._reform_matrices(weights_vector)
		samples = X.shape[0]

		total_log_cost = 0
		for i in range(samples):
			self.forward_prop(X[i,:])
			hypothesis = self.layer_list[-1].activations 
			y = Y[i,:]
			log_cost = y*np.log(hypothesis) + (np.ones(y.shape) - y)*np.log(np.ones(hypothesis.shape) - hypothesis)
			total_log_cost += np.sum(log_cost)
		
		squared_weights = 0
		for i in range(len(self.weight_matrices)):
			squared_weights += np.sum((self.weight_matrices[i][:,-1])**2) # don't regularize bias weights
		reg_term = lam*1.0/(2*samples)*squared_weights

		return -1.0/samples*total_log_cost + reg_term 		

	def forward_prop(self, inputs):
		"""run forward prop"""
		assert len(inputs) == self.num_units[0]
		self.layer_list[0].activations[:-1] = np.expand_dims(inputs, 1) #preserves bias as -1	
		for i in range(1,len(self.layer_list)):
			self.layer_list[i].calculate_activations(self.layer_list[i-1].activations, 
				self.weight_matrices[i-1])			

	def back_prop(self, y):
		"""run back prop"""
		self._calc_error(y)
		for i in range(len(self.accumulators)-1):
			self.accumulators[i] += np.dot(self.layer_list[i+1].activation_errors[:-1], 
				np.transpose(self.layer_list[i].activations))
		index = len(self.accumulators)-1
		self.accumulators[index] += np.dot(self.layer_list[index+1].activation_errors, 
				np.transpose(self.layer_list[index].activations)) #the bias is absent

	def _calc_error(self, y):
		"""calcs error func wrt y and next layer for all layers"""
		self.layer_list[-1].calc_error(y, weight_matrix = None)
		for i in range(len(self.layer_list)-2,0,-1): #not the input layer bc range is right exclusive
			self.layer_list[i].calc_error(self.layer_list[i+1].activations, self.weight_matrices[i])

def initialize_sigmoid_network(num_units):
	"""initialize network with all sigmoid activation functions"""
	return Network(num_units, [act_funcs.Sigmoid()]*(len(num_units)), 0.001)


if __name__ == '__main__':
		
	#print [act_funcs.Sigmoid()]*(len(num_units)-1)
	net = initialize_sigmoid_network([3,2,1])
	X  = np.array([[1,3,4],[6,1,-1],[1,3,4],[6,1,-1]])
	Y = np.array([[1],[0],[1],[0]])
	w = net.train_net(X,Y)
	X2 = np.array([[0,0,0],[1,1,2]])
	print w 
	print net.predict(X,w)
	# with few samples, performance seems highly dependent on random initialization

	"""X3 = np.array([[1,3,4],[6,1,-1]])
	Y3 = np.array([[1],[0]])
	w3 = net.train_net(X3, Y3)
	print net.predict(X3, w3)"""
	
		#print net.__dict__

	print "hello"	




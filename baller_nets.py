import numpy as np
import baller_act_fcns as act_funcs
import scipy.io
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

	def calc_error(self, next_layer_acts_errors, next_is_last, weight_matrix):
		"""look at the next layer or outputs to calc the error of the activations"""
		if self.is_last_layer:
			self.activation_errors = self.activations - next_layer_acts_errors #should be replaced with proper error func
			"""y = next_layer_acts_errors
			hypothesis = self.activations
			log_cost = np.divide(y,hypothesis) - np.divide((np.ones(y.shape) - y),np.ones(hypothesis.shape) - hypothesis)
			log_cost *= -1
			self.activation_errors = log_cost
			"""
		else:
			if not(next_is_last):
				next_layer_acts_errors = next_layer_acts_errors[:-1] #to get rid of the bias term
			a = np.dot(np.transpose(weight_matrix),next_layer_acts_errors)
			#b = self.calc_deriv_activation()
			b = (self.activations * (1-self.activations)) #should be equivalent
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

	# these fcns could be generalized?
	def _init_weights_random(self):
		"""initialize network weights in each layer"""
		self.weight_matrices = []
		self.shapes_list = []
		for i in range(len(self.layer_list)-1):			
			self.weight_matrices.append(np.random.randn(self.num_units[i+1], self.num_units[i]+1)) #succeeding layer, current layer with the bias
			self.shapes_list.append(np.shape(self.weight_matrices[i]))

	def _init_weights_as_one(self):
		"""initialize network weights in each layer"""
		self.weight_matrices = []
		self.shapes_list = []
		for i in range(len(self.layer_list)-1):
			self.weight_matrices.append(np.ones((self.num_units[i+1], self.num_units[i]+1)))
			self.shapes_list.append(np.shape(self.weight_matrices[i]))
	
	def _init_weights_as_zero(self):
		"""initialize network weights in each layer as zero for testing purposes"""
		self.weight_matrices = []
		self.shapes_list = []
		for i in range(len(self.layer_list)-1):
			self.weight_matrices.append(np.zeros((self.num_units[i+1], self.num_units[i]+1)))
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
		self._init_weights_random()
		init_weights_vector = self._unroll_matrices(self.weight_matrices)
		solution = fmin_cg(self.evaluate_cost, init_weights_vector, 
				fprime = self.calc_error_derivs, maxiter = 200)
		return solution

	def predict(self, X, weights_vector):
		self.X = X
		self.weight_matrices = self._reform_matrices(weights_vector)
		
		samples = X.shape[0]
		Y = np.zeros((samples, len(self.layer_list[-1].activations)))
		for i in range(samples):
			self.forward_prop(X[i,:])
			Y[i] = self.layer_list[-1].activations[:,0]

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
			self.forward_prop(X[i,:]) #propagate the activations
			self.back_prop(Y[i,:]) #compute activation errors

		# error_derivs is the list of matrix derivatives of cost fcn wrt the weight matrices
		error_derivs = []
		for W in self.weight_matrices:
			error_derivs.append(lam*W) # initialize with the regularization term

		for i in range(len(error_derivs)):
			#zero the bias term to eliminate regularization for it
			error_derivs[i][:,-1] = np.zeros((error_derivs[i].shape[0],)) 
			#then add the result of the accumulators, from backprop
			error_derivs[i] += self.accumulators[i]/float(samples)

		return self._unroll_matrices(error_derivs)

	def evaluate_cost(self, weights_vector):
		"""Evaluate prediction cost, need to pass into weight matrices for the gradient descent code"""
		X = self.X
		Y = self.Y
		lam = self.lam
		self.weight_matrices = self._reform_matrices(weights_vector)
		# get the matrices back, this is dependent on the shapes of the matrices stored
		samples = X.shape[0]

		total_log_cost = 0
		for i in range(samples):
			self.forward_prop(X[i,:])
			hypothesis = self.layer_list[-1].activations 
			y = Y[i,:]
			# cost fcn is correct
			log_cost = y*np.log(hypothesis) + (np.ones(y.shape) - y)*np.log(np.ones(hypothesis.shape) - hypothesis)
			total_log_cost += np.sum(log_cost) #sign is correct (negated below)
		
		sq_cost = 0
		for i in range(samples):
			self.forward_prop(X[i,:])
			hypothesis = self.layer_list[-1].activations 
			y = Y[i,:]
			sq_cost += np.sum(0.5*(y-hypothesis)**2)

		squared_weights = 0
		for i in range(len(self.weight_matrices)):
			squared_weights += np.sum((self.weight_matrices[i][:,-1])**2) # don't regularize bias weights
		reg_term = lam*1.0/(2*samples)*squared_weights

		#return sq_cost/samples + reg_term
		return -1.0/samples*total_log_cost + reg_term 		

	def forward_prop(self, inputs):
		"""run forward prop"""
		assert len(inputs) == self.num_units[0]
		self.layer_list[0].activations[:-1] = np.expand_dims(inputs, 1) #set input layer, preserve bias as -1	
		for i in range(1,len(self.layer_list)): #exclude the 
			self.layer_list[i].calculate_activations(self.layer_list[i-1].activations, 
				self.weight_matrices[i-1])			

	def back_prop(self, y):
		"""run back prop, finding values for the accumulators"""
		self._calc_error(y)
		for i in range(len(self.accumulators)-1):
			self.accumulators[i] += np.dot(self.layer_list[i+1].activation_errors[:-1], 
				np.transpose(self.layer_list[i].activations))
		index = len(self.accumulators)-1
		self.accumulators[index] += np.dot(self.layer_list[index+1].activation_errors, 
				np.transpose(self.layer_list[index].activations)) #the bias is absent

	def _calc_error(self, y):
		"""calcs error func wrt y and next layer for all layers"""
		self.layer_list[-1].calc_error(y, next_is_last = False, weight_matrix = None)
		for i in range(len(self.layer_list)-2,0,-1): #not the input layer bc range is right exclusive
			self.layer_list[i].calc_error(self.layer_list[i+1].activation_errors, self.layer_list[i+1].is_last_layer, self.weight_matrices[i])
			### NOTE: CHANGED NEXT ACTIVATIONS TO NEXT ACTIVATION ERRORS

	def gradient_check(self):
		"""numerically estimates gradient and prints for user to check against backprop"""		
		self._init_weights_random()
		weights_vector = self._unroll_matrices(self.weight_matrices)
		numerical_sol = self.estimate_gradient(weights_vector)
		backprop_sol = self.calc_error_derivs(weights_vector)

		return numerical_sol, backprop_sol, np.mean(np.abs(numerical_sol-backprop_sol))

	def estimate_gradient(self,weights_vector):
		epsilon = 10e-4
		grad_approx = np.empty( weights_vector.shape)
		for i in xrange(len(weights_vector)):
			larger_theta = np.empty(weights_vector.shape)
			smaller_theta = np.empty(weights_vector.shape)
			np.copyto(larger_theta,weights_vector)
			np.copyto(smaller_theta,weights_vector)
			larger_theta[i] += epsilon
			smaller_theta[i] += -epsilon
			grad_approx[i] = (self.evaluate_cost(larger_theta) - self.evaluate_cost(smaller_theta) )/(2.0*epsilon)
		return grad_approx


def load_coursera_data_file():
	data = scipy.io.loadmat("ex3data1.mat")
	y = data['y']
	X = data['X']
	all_data = np.hstack([y,X])
	np.random.shuffle(all_data) #randomly shuffle data so test and training sets will see all examples
	num_examples = all_data.shape[0]
	
	split_set_index = 30
	training_y = all_data[:split_set_index,0:1]
	training_X = all_data[:split_set_index,1:]
	training_set = { 'y':training_y,'X':training_X }
	test_y = all_data[split_set_index:40,0:1]
	test_X = all_data[split_set_index:40,1:]
	test_set = { 'y':test_y,'X':test_X }
	return training_X, test_X, training_y, test_y

def initialize_sigmoid_network(num_units, lam):
	"""initialize network with all sigmoid activation functions"""
	return Network(num_units, [act_funcs.Sigmoid()]*(len(num_units)), lam)

def run_MNIST():
	X_train, X_test, Y_train, Y_test = load_coursera_data_file()
	net_m = initialize_sigmoid_network([400,10], 0.1)
	sol = net_m.train_net(X_train, Y_train)
	print sol
	pred = net_m.predict(X_test, sol)
	print pred
	print Y_test

if __name__ == '__main__':
		
	#print [act_funcs.Sigmoid()]*(len(num_units)-1)

	#run_MNIST()
	net = initialize_sigmoid_network([3,2,2,1], lam = 0.00)
	X  = np.array([[1,3,4],[6,1,-1],[1,3,3],[6,1,0]])
	Y = np.array([[1],[0],[1],[0]])
	#print net.weight_matrices
	#print net.predict(X,net.weight_matrices)
	#w = net.train_net(X,Y)
	#print "updated weights:"
	#print w 
	#print net.predict(X,w)
	net.X = X
	net.Y = Y
	a, b, c = net.gradient_check()
	print zip(a,b)
	
	print c 
	#X2 = np.array([[0,0,0],[1,1,2]])
	
	# with few samples, performance seems highly dependent on random initialization
	
	"""X3 = np.array([[1,3,4],[6,1,-1]])
	Y3 = np.array([[1],[0]])
	w3 = net.train_net(X3, Y3)
	print net.predict(X3, w3)"""
	#print net.__dict__
	print "hello"
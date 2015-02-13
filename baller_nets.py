import numpy as np
import baller_act_fcns as act_funcs

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
			a = np.dot(np.transpose(self.weight_matrix),next_layer_acts)
			b = self.calc_deriv_activation(self.activations)
			self.activation_errors = np.multipy(a,b)
		return self.activation_errors


class Network():
	def __init__(self, num_units, act_list):
		self.num_units = num_units
		self.act_list = act_list
		self.layer_list = []
		for num, act in zip(self.num_units[:-1], self.act_list[:-1]):
			self.layer_list.append(Layer(num, act))		
		self.layer_list.append(Layer(num_units[-1], act_list[-1], is_last_layer = True))
		self._init_weights_as_one()
		
	def _init_weights(self):
		"""initialize network weights in each layer"""
		self.weight_matrices = []
		for i in range(len(self.layer_list)-1):
			self.weight_matrices.append(np.random.rand(self.num_units[i+1], self.num_units[i]+1)) #succeeding layer, current layer with the bias

	def _init_weights_as_one(self):
		"""initialize network weights in each layer"""
		self.weight_matrices = []
		for i in range(len(self.layer_list)-1):
			self.weight_matrices.append(np.ones((self.num_units[i+1], self.num_units[i]+1)))
		
	def forward_prop(self, inputs):
		"""run forward prop"""
		assert len(inputs) == self.num_units[0]
		self.layer_list[0].activations[:-1] = np.expand_dims(inputs, 1) #preserves bias as -1	
		for i in range(1,len(self.layer_list)):
			self.layer_list[i].calculate_activations(self.layer_list[i-1].activations, self.weight_matrices[i-1])			

	def back_prop(self, y):
		"""run back prop"""
		self._calc_error(y)


	def _calc_error(self, y):
		"""calcs error func wrt y and next layer for all layers"""
		self.layer_list[-1].calc_error(y, weight_matrix = None)
		for i in range(len(self.layer_list)-2,0,-1): #not the input layer bc range is right exclusive
			self.layer_list[i].calc_error(self.layer_list[i+1].activations, weight_matrices[i])

def initialize_sigmoid_network(num_units):
	"""initialize network with all sigmoid activation functions"""
	return Network(num_units, [act_funcs.Sigmoid()]*(len(num_units)))


if __name__ == '__main__':
	
	#print [act_funcs.Sigmoid()]*(len(num_units)-1)
	net = initialize_sigmoid_network([3,2,1])
	#print net.weight_matrices
	net.forward_prop([1,1,1])
	print net.layer_list[-1].activations
	#print net.num_units
	#print net.layer_list
	#print net.act_list
	#print net.__dict__

	print "hello"	




import numpy as np
import baller_act_fcns as act_funcs

class Layer():
	def __init__(self, units, activation_function):
		"""Vectorized implemenation of a layer"""
		self.units = units #number of units		
		self.activation_function = activation_function #should act on a numpy vector and return one
		
	def calculate_activations(self, prev_activation, weight_matrix):
		"""returns activations as a vector"""
		summed_input = np.dot(weight_matrix, prev_activation)
		return self.activation_function.evaluate_func(summed_input)

	def calc_deriv_activation(self, past_activation):
		"""calculate the derivative of the activation for this layer"""
		return self.activation_function.eval_deriv_func(past_activation)

class Network():
	def __init__(self, num_units, act_list):
		self.num_units = num_units
		self.act_list = act_list
		self.layer_list = []
		
		for num, act in zip(num_units, act_list):
			self.layer_list.append(Layer(num, act))		
		self._init_weights_as_one()
		self._init_neuron_outputs()

	def _init_weights(self):
		"""initialize network weights in each layer"""
		self.weight_matrices = []
		for i in range(len(self.layer_list)):
			self.weight_matrices.append(np.random.rand(self.num_units[i+1], self.num_units[i]+1))

	def _init_weights_as_one(self):
		"""initialize network weights in each layer"""
		self.weight_matrices = []
		for i in range(len(self.layer_list)):
			self.weight_matrices.append(np.ones((self.num_units[i+1], self.num_units[i]+1)))

	def _init_neuron_outputs(self):
		"""initialize neuron outputs from prev forward prop run, starts everything
		including the biases as ones"""
		self.neuron_outputs = []
		for i in range(len(self.layer_list)):
			self.neuron_outputs.append(np.ones((self.num_units[i]+1, 1))) #for the bias unit
		self.neuron_outputs.append(np.ones(self.num_units[-1],1)) #last one doesn't have bias unit
		print self.neuron_outputs
	
	def _init_output_error(self):
		self.output_error = []
		for i in range(1,len(self.layer_list)):
			self.output_error.append(np.ones((self.num_units[i]+1, 1))) 
		self.output_error.append(np.ones(self.num_units[-1],1))
		
		
	def forward_prop(self, inputs):
		"""run forward prop"""
		assert len(inputs) == self.num_units[0]
		self.neuron_outputs[0][:-1] = np.expand_dims(inputs, 1) #preserves bias as -1
		for i in range(1,len(self.layer_list)+1):
			self.neuron_outputs[i][:-1] = self.layer_list[i-1].calculate_activations(self.neuron_outputs[i-1], self.weight_matrices[i-1])

	def back_prop(self, y):
		"""run back prop"""
		self._calc_error(y)


	def _calc_error(self, y):
		"""calcs error func wrt y for some output layer outputs"""
		 self.output_error[-1] = self.neuron_outputs[-1] - y #replace with proper error func
		 for i in range(len(self.layer_list)-2,0,-1): #not the input layer bc range is right exclusive
		 	a = np.dot(np.transpose(self.weight_matrices[i]),self.output_error[i+1])
		 	b = self.layer_list[i].calc_deriv_activation(self.neuron_outputs[i])
		 	self.output_error[i] = np.multipy(a,b)


def initialize_sigmoid_network(num_units):
	"""initialize network with all sigmoid activation functions"""
	return Network(num_units, [act_funcs.Sigmoid()]*(len(num_units)-1))


if __name__ == '__main__':
	s = act_funcs.Sigmoid()
	print s.evaluate_func(np.array([1,2,4]))

	net = initialize_sigmoid_network([3,2,1])
	#print net.weight_matrices
	net.forward_prop([1,1,1])
	print net.neuron_outputs
	#print net.num_units
	#print net.layer_list
	#print net.act_list
	#print net.__dict__

	print "hello"	




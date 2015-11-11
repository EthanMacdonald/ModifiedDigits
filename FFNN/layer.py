import numpy as np

from theano import *

from scipy.special import expit

class Layer(object):
	"""
	A fully-connected layer of a feedforward neural network with a 
	sigmoid activation function.
	"""

	def _init_W(self):
		high = 4*numpy.sqrt(6./(self.input_n+self.layer_n))
		low = -1*high
		self.W = np.asarray(self.rng.uniform(high=high, low=low, size=(self.layer_n, self.input_data.shape[0])))

	def __init__(self, input_data, input_n, layer_n, W=None, b=None, first=False):
		"""
		Initializes the layer.
	
		Args:
			rng: A numpy.random.RandomState used to generate initial weights.
			input_data: A theano.tensor.dmatrix of dimensions (example_n, input_n) 
				filled with the inputs from the previous layer.
			input_n: An int representing the number of inputs from the previous
				layer.
			layer_n: An int representing how many nodes this layer should have.
			W: W is a matrix of weights with dimensions (input_n, layer_n).
				In other words, each column in W corresponds to the set of weights 
				entering a particular node in the current layer. Likewise, each row 
				corresponds to the set of weights leaving a particular node in the 
				previous layer.
			b: bias vector of shape (layer_n,)
			activation: theano op or function used as the activation function

		Raises:
			Nothing.
		"""

		self.rng = np.random
		self.input_data = np.append(input_data, [[1]], 0)
		self.input_n = input_n
		self.layer_n = layer_n
		self.W = W
		self.activation = lambda x: 1.0/(1.0 + np.exp(-x))
		self.deltas = None

		if self.W is None: self._init_W()
		np.dot(self.W, self.input_data)
		self.output = self.activation(np.dot(self.W, self.input_data))

	def get_output(self, input_data, dropout):
		self.input_data = np.append(input_data, [[1]], 0)
		self.output = self.activation(np.dot(self.W, self.input_data))
		self.output *= np.random.binomial(1,1.0-dropout,self.output.shape)
		return self.output
import numpy as np

from theano import *
import theano.tensor as t

class Layer(object):
	"""
	A fully-connected layer of a feedforward neural network with a 
	sigmoid activation function.
	"""

	def _init_W(self):
		"""
		Initialize W with rng.

		Args:
			Nothing.

		Raises:
			Nothing.
		"""
		high = 4*numpy.sqrt(6./(self.input_n+self.layer_n))
		low = -1*high
		self.W = theano.shared(
					value = np.asarray(self.rng.uniform(high=high, low=low, size=(self.input_data.shape[1], self.layer_n)), 
					dtype = theano.config.floatX),
					name = 'W',
					borrow = True
				)

	def _init_b(self):
		"""
		Initalize b with rng.

		Args:
			Nothing.

		Raises:
			Nothing.
		"""
		self.b = theano.shared(
				value = np.zeros((self.layer_n,), dtype=theano.config.floatX),
				name = 'b',
				borrow = True
				)

	def __init__(self, rng, input_data, input_n, layer_n, W=None, b=None):
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

		self.rng = rng
		self.input_data = input_data
		self.input_n = input_n
		self.layer_n = layer_n
		self.W = W
		self.b = b
		self.activation = theano.tensor.nnet.sigmoid #TODO: Support for other activation functions?/non-theano implementation?
		self.deltas = None

		if self.W is None: self._init_W()
		if self.b is None: self._init_b()
		self.output = self.activation(t.dot(self.input_data, self.W) + self.b)

	def get_output(self, input_data):
		"""
		Calculates output for a given input. Updates self.input_data and self.output in the process.

		Args:
			input_data: np.array with dimensions (self.input_data.shape[1], self.input_n)

		Returns:
			theano.tensor.var.TensorVariable
		"""
		self.input_data = input_data
		self.output = self.activation(t.dot(self.input_data, self.W) + self.b).eval()
		return self.output

	def param_shapes(self):
		shapes = {
			'input_data': self.input_data.shape,
			'input_n' : self.input_n,
			'layer_n' :self.layer_n,
			'W' :self.W.shape.eval(),
			'b' :self.b.shape.eval(),
			'deltas' :self.deltas.shape,
		}
		return shapes
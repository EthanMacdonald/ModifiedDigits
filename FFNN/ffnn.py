import numpy as np

from theano import *
import theano.tensor as t

from layer import Layer

from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

class FFNN(object):

	def __init__(self, rng, input_data, layer_ns):
		"""
		Initializes the feed forward neural network.
	
		Args:
			rng: A numpy.random.RandomState used to generate initial weights.
			input_data: A theano.tensor.dmatrix of dimensions (number of examples, layer_ns[0]) 
				filled with the inputs from the images.
			layer_ns: a list of ints representing how many nodes each layer should have

		Raises:
			Nothing.
		"""
		
		self.rng = rng
		self.current_output = input_data 
		self.layer_ns = layer_ns

		self.layers = []
		for i in range(len(layer_ns)-1):
			self.layers += [Layer(rng=rng, input_data=self.current_output, input_n=self.layer_ns[i], output_n=self.layer_ns[i+1])]
			self.current_output = self.layers[-1].output.eval()

		self.output_layer = LogisticRegression()

	def train_batch(self, batch_data, correct_output, step_size):
		"""
		Given a batch of data, train the network
		"""
		self.gradient_descent(self.backprop(self.forward_pass(batch_data), correct_output), step_size)

	def forward_pass(self, batch_data):
		"""
		Given a batch of data, propogate it through the network 
		and return the output
		"""
		input_data = batch_data
		for layer in self.layers:
			input_data = layer.get_output(input_data)
		return input_data

	def backprop(self, observed, correct):
		"""
		Given an obeserved output and a correct output, calculate
		the correction for each node in each layer.
		Return a list of arrays, one for each layer.
		"""
		deltas = []
		for i in range(len(observed)):
			if i == 0:
				deltas = []
		return True

	def gradient_descent(self, deltas, step_size):
		"""
		Given a set of deltas and a step size, perform gradient descent on each weight.
		"""
		return True
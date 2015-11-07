import time
import numpy as np

from theano import *
import theano.tensor as t

from layer import Layer

from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

from sklearn.base import BaseEstimator

class FFNN(BaseEstimator):

	def __init__(self, input_data, layer_ns, step_size, termination, rng=np.random):
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
		self.step_size = step_size
		self.termination = termination

		self.layers = []
		for i in range(0,len(layer_ns)):
			print "...layer %d size %d" % (i, layer_ns[i])
			self.layers += [Layer(rng=rng, input_data=self.current_output, input_n=self.layer_ns[i-1], layer_n=layer_ns[i])]
			self.current_output = self.layers[-1].output.eval()
		print "Number of layers"
		print len(self.layers)

	def forward_pass(self, batch_data):
		"""
		Given a batch of data, propogate it through the network 
		and return the output
		"""
		print "Forward pass: "
		input_data = batch_data
		for layer in self.layers:
			input_data = layer.get_output(input_data)
			print input_data.shape
		return input_data

	def backprop(self, observed, correct):
		"""
		Given an observed output and a correct output, calculate
		the correction for each node in each layer.
		Returns a list of arrays, one for each layer.

		Args:
			observed: np.ndarray of shape (self.input_data.shape[0], 10)
			correct: np.ndarray of shape (self.input_data.shape[0], 10)

		Returns:
			list of size len(self.layer_ns) containing np.ndarrays of shape (self.layer_ns[i],)
		"""
		# TODO: This is spitting out numbers in the correct format, but are their values correct?
		print "Backprop: "
		last_delta = t.sum(((correct - observed)*observed*(1-observed)), axis=0)
		last_W = self.layers[-1].W
		self.layers[-1].deltas = last_delta.eval()
		print np.sum(self.layers[-1].deltas, axis=0)
		# Calculate new deltas for each layer
		for i in range(len(self.layers)-2,-1,-1):
			print "layer %d" % i
			O = self.layers[i].output
			last_delta = t.sum(t.sum(((1 - O)*O), axis=0)*t.transpose(last_W*last_delta), axis=0)
			self.layers[i].deltas = last_delta.eval()
			last_W = self.layers[i].W
			print np.sum(self.layers[i].deltas, axis=0)
		
	def score(self, X, y):
		output = self.forward_pass(X)
		for row in output:
			zeros = np.zeros(self.layer_ns[-1])
			zeros[np.argmax(row)] = 1
			row = zeros
		score = np.zeros(len(output))
		for i in range(len(output)):
			if np.array_equal(output[i], y[i]):
				score[i] = 1
		return np.sum(score)/len(score)

	def fit(self, input_data, correct_output, batch_size=100):
		"""
		Train the network
		"""
		for i in range(batch_size, len(input_data), batch_size):
			print "...training up to example %d" % i
			self.backprop(self.forward_pass(input_data[i-batch_size:i]), correct_output[i-batch_size:i])
			self.gradient_descent()
			delta = t.sum(self.layers[-1].deltas, axis=0)
			print delta.eval()
			iters = 0
			while (self.termination < (delta.eval() ** 2)) and (iters < 5):
				print delta.eval()
				self.backprop(self.forward_pass(input_data[i-batch_size:i]), correct_output[i-batch_size:i])
				self.gradient_descent()
				delta = t.sum(self.layers[-1].deltas, axis=0)
				iters += 1

	def gradient_descent(self):
		"""
		Given a set of deltas and a step size, perform gradient descent on each weight.
		"""
		# TODO: This is also spitting out numbers in the correct format, but are the values correct?
		print "Gradient descent: "
		for i in range(len(self.layers)-1,-1,-1):
			print "layer %d" % i
			layer_n = self.layers[i].layer_n
			inputs = np.array([np.sum(self.layers[i].input_data, axis=0),]*layer_n)
			self.layers[i].W = self.layers[i].W + self.step_size*self.layers[i].deltas*np.transpose(inputs)
			print self.layers[i].param_shapes()
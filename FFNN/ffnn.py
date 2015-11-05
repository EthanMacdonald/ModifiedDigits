import numpy as np

from theano import *
import theano.tensor as t

from ffnn.layer import Layer

from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

class FFNN(object):

	def get_error(self):
		#TODO

	def __init__(self, rng, input_data, input_n, output_n, hidden_n):
		
		self.rng = rng # Random number generator
		self.input_data = input_data # Input data
		self.input_n = input_n # Size of input (this is a list)
		self.output_n = output_n # Size of output (this is a list)
		self.hidden_n = hidden_n # Number of hidden layers
		# TODO: crunch input_n, output_n, and hidden_n into one list/array/matrix

		self.input_layer = Layer(rng=rng, input_data=self.input_data, input_n=input_n[i], output_n=input_n[i+1])
		self.output = self.input_layer.output

		self.hidden_layers = []
		for i in range(hidden_n):
			self.hidden_layers += Layer(rng=rng, input_data=self.output, input_n=input_n[i], output_n=input_n[i+1])
			self.output = self.hidden_layers[-1].output

		self.output_layer = Layer(rng=rng, input_data=self.output, input_n=input_n[hidden_n], output_n=input_n[hidden_n+1])
		self.output = self.output_layer.output

		self.error = self.get_error()

	#TODO: Add more functions —— train, score, etc.
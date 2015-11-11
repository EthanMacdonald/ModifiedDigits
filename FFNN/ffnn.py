import time, sys
import numpy as np

from layer import Layer

from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

from sklearn.base import BaseEstimator

class FFNN(BaseEstimator):

	def __init__(self, input_data, layer_ns, step_size, termination, dropout=0.0):
		self.input_data = input_data.T
		self.layer_ns = layer_ns
		self.step_size = step_size
		self.termination = termination
		self.dropout = dropout
		self.current_output = self.input_data
		self.layers = []

		for i in range(0,len(layer_ns)):
			self.layers += [Layer(input_data=self.current_output, input_n=self.layer_ns[i-1], layer_n=layer_ns[i])]
			self.current_output = self.layers[-1].output

		self.layers[-1].dropout = 0.0

	def forward_pass(self, input_data):
		input_data *= np.random.binomial(1,1.0-self.dropout,input_data.shape)
		for layer in self.layers:
			input_data = layer.get_output(input_data, self.dropout)
		return input_data

	def _calculate_output_delta(self, O, y):
		temp = np.subtract(1, O)
		temp2 = np.subtract(y,O)
		temp = np.multiply(O, temp)
		temp = np.multiply(temp, temp2)
		self.layers[-1].deltas = temp # Save output deltas
		return temp

	def backprop(self, O, y):
		last_delta = self._calculate_output_delta(O, y)
		last_W = self.layers[-1].W.T[:-1]
		# Calculate new deltas for each layer
		for i in range(len(self.layers)-2,-1,-1):
			O = self.layers[i].output
			wdelta = np.dot(last_W,last_delta)
			obs = np.multiply(O, np.subtract(1, O))
			last_delta = np.multiply(obs,wdelta)
			last_W = self.layers[i].W.T[:-1]
			self.layers[i].deltas = last_delta

	def gradient_descent(self):
		for i in range(len(self.layers)-1,-1,-1):
			layer_n = self.layers[i].layer_n
			inputs = np.hstack([self.layers[i].input_data]*layer_n).T
			deltas = self.layers[i].deltas
			dinputs = np.multiply(deltas,inputs)
			self.layers[i].W += self.step_size*dinputs

	def _diff_one_hots(self, X, y):
		score = np.zeros(len(X))
		for i in range(len(X)):
			if np.array_equal(X[i], y[i]):
				score[i] = 1
		return score

	def _to_one_hot(self, X):
		new_X = []
		for row in X:
			zeros = np.zeros(len(row))
			m = np.argmax(row)
			zeros[m] = 1
			new_X += [zeros]
		return np.array(new_X)

	def _from_one_hot(self, X):
		new_X = []
		for row in X:
			new_X += [np.argmax(row)]
		return np.array(new_X)

	def score(self, X, y):
		score = 0
		for i in range(len(X)):
			row = X[i]
			Xr = self.forward_pass(row.reshape(len(row),1))
			Xr = np.argmax(Xr)
			yr = np.argmax(y[i])
			if Xr == yr: score += 1
		return float(score)/len(X)

	def fit(self, input_data, correct_output, valid_input, valid_output, batch_size=100):
		"""
		Train the network
		"""

		epoch = 1
		old_error = 10000000.
		new_error = 1.
		while (self.termination < old_error-new_error):
			print "Epoch " + str(epoch)
			for i in range(0, len(input_data)-batch_size, batch_size):
				self.backprop(self.forward_pass(input_data[i:i+batch_size].T), correct_output[i:i+batch_size].T)
				self.gradient_descent()
			score = self.score(valid_input, valid_output)
			old_error = new_error
			new_error = 1-score
			print "Score: %.15f, Error: %.15f, Difference: %.15f" % (score, new_error, old_error-new_error)
			epoch += 1
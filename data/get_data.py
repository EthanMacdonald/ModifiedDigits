import numpy as np
from collections import namedtuple

def get_data(verbose=True):
	if verbose: print "Start Data Load"
	train_inputs = np.load('data/train_inputs.npy')
	if verbose: print "....training input shape: " + str(train_inputs.shape)
	train_outputs = np.load('data/train_outputs.npy')
	if verbose: print "....training output shape: " + str(train_outputs.shape)
	test_inputs = np.load('data/test_inputs.npy')
	if verbose: print "....testing input shape: " + str(test_inputs.shape)
	test_outputs = np.load('data/test_outputs.npy')
	if verbose: print "....testing output shape: " + str(test_outputs.shape)
	if verbose: print "End Data Load"
	dataset = namedtuple('Dataset', ['train_inputs', 'train_outputs', 'test_inputs', 'test_outputs'])
	return dataset(train_inputs, train_outputs, test_inputs, test_outputs)
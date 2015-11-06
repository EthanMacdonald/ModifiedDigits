import numpy as np
from ffnn.ffnn import FFNN

inputs = np.load('data/train_inputs.npy')
outputs = np.load('data/train_outputs_matrix.npy')

layer_ns = [inputs.shape[1], inputs.shape[1]/2, inputs.shape[1]/4, inputs.shape[1]/8, 10]

ffnn = FFNN(np.random, inputs[:100], layer_ns)

bp = ffnn.backprop(ffnn.forward_pass(inputs[:100]), outputs[:100])
ffnn.gradient_descent(0.1)

ffnn.train_batch(inputs[:100], outputs[:100], 0.01)

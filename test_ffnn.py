import numpy as np
from ffnn.ffnn import FFNN

inputs = np.load('data/train_inputs.npy')
outputs = np.load('data/train_outputs_matrix.npy')

layer_ns = [inputs.shape[1], inputs.shape[1]/2, inputs.shape[1]/4, inputs.shape[1]/8, 10]

ffnn = FFNN(np.random, inputs[:100], layer_ns)

end = ffnn.backprop(ffnn.forward_pass(inputs[:100]), outputs[:100])
print layer_ns
print [x.shape for x in end]
print [layer.deltas.shape for layer in ffnn.layers]
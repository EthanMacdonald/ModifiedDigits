import itertools, pickle, time, json
import numpy as np

from sklearn.cross_validation import KFold

from ffnn.ffnn import FFNN

train_inputs_path = 'data/train_inputs.npy'
train_outputs_path = 'data/train_outputs_matrix.npy'
test_inputs_path = 'data/test_inputs.npy'
test_outputs_path = 'data/test_outputs_matrix.npy'

train_inputs = np.load(train_inputs_path)
train_outputs = np.load(train_outputs_path)
test_inputs = np.load(test_inputs_path)
test_outputs = np.load(test_outputs_path)

def list_permutations(max_length, min_length, max_nodes, min_nodes, node_skip=100):
	print "Generating permutations...."
	lst = []
	for i in range(min_length, max_length):
		lst += [list(x) + [10] for x in itertools.product(range(min_nodes, max_nodes, node_skip), repeat=i)]
	print "Done permutations..."
	return lst

################################################
# Configuration settings — change as necessary #
################################################
layer_ns = list_permutations(max_length=5, min_length=2, max_nodes=2010, min_nodes=10, node_skip=100)
params = {
	"layer_ns": layer_ns, 
	"step_size": [10,1,.1], 
	"termination": [10,1,.1]
	}
max_example = 900
batch_size = 599
################################################

best_score = 0.0
best_params = None
best_model = None

number_of_models = len(params['layer_ns'])*len(params['step_size'])*len(params['termination'])
remaining_models = number_of_models

print "Total number of models: " + str(number_of_models)

for architecture in params['layer_ns']:
	for step_size in params['step_size']:
		for termination in params['termination']:
			start = time.time()
			print "##########################################################################"
			print "NEW MODEL: #" + str(remaining_models)
			print {'layer_ns': architecture, 'step_size': step_size, 'termination': termination}
			print "##########################################################################"
			scores = []
			for train_index, test_index in KFold(train_inputs[:max_example].shape[0], n_folds=3):
				X_train, X_test = train_inputs[train_index], train_inputs[test_index]
				y_train, y_test = train_outputs[train_index], train_outputs[test_index]
				ffnn = FFNN(X_train, architecture, step_size, termination)
				ffnn.fit(X_train, y_train, batch_size=batch_size)
				scores += [ffnn.score(X_test, y_test)]
			score_avg = sum(scores)/len(scores)
			print "Score average: " + str(score_avg)
			if score_avg > best_score:
				print "##########################################################################"
				print "NEW MAX SCORE: " + str(score_avg) 
				best_score = score_avg
				best_params = {'score': best_score, 'layer_ns': architecture, 'step_size': step_size, 'termination': termination}
				best_model = ffnn
				with open('best_ffnn.json', 'a') as out:
					json.dump(best_params, out)
					out.write('\n')
			remaining_models -= 1
			print "Estimated time remaining: " + str((time.time() - start)*remaining_models)

print "##########################################################################"
print "Best Score:"
print best_score
print "##########################################################################"
print "Best Params:"
print best_params
print "##########################################################################"
with open('overall_best_ffnn.json', 'a') as out:
	best_params['score'] = best_score
	json.dump(best_params, out)
	out.write('\n')
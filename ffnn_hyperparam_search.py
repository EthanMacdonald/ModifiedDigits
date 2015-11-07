import itertools, pickle, time, json, random
import numpy as np

from sklearn.cross_validation import KFold

from ffnn.ffnn import FFNN

################################################
# Initialization/helper functions              #
################################################

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
layer_ns = list_permutations(max_length=3, min_length=1, max_nodes=2011, min_nodes=10, node_skip=50)
params = {
	"layer_ns": layer_ns, 
	"step_size": [10,1,.1,.01,.001,.0001,.00001,.000001,.0000001], 
	"termination": [10,1,.1,.01,.001,.0001,.00001,.000001,.0000001]
	}
max_example = None
batch_size = 50
################################################

best_score = 0.0
best_params = None
best_model = None
avg_time = 0

models = [(architecture, step_size, termination) for architecture in params['layer_ns'] for step_size in params['step_size'] for termination in params['termination']]

number_of_models = len(models)
remaining_models = number_of_models

print "Total number of models: " + str(number_of_models)

for i in range(number_of_models):
	architecture, step_size, termination = random.choice(models)
	models.remove((architecture, step_size, termination))
	current_params = {'layer_ns': architecture, 'step_size': step_size, 'termination': termination, 'batch_size': batch_size, 'max_example': max_example}
	print "##########################################################################"
	print "Model #%d" % remaining_models
	print current_params
	print "##########################################################################"
	start = time.time()
	scores = []
	epoch = 1
	epoch_start = time.time()
	for train_index, test_index in KFold(train_inputs[:max_example].shape[0], n_folds=3):
		X_train, X_test = train_inputs[train_index], train_inputs[test_index]
		y_train, y_test = train_outputs[train_index], train_outputs[test_index]
		ffnn = FFNN(X_train[:batch_size], architecture, step_size, termination)
		ffnn.fit(X_train, y_train, batch_size=batch_size)
		score = ffnn.score(X_test, y_test)
		print "Epoch " + str(epoch) + " score: " + str(score)
		print "Epoch " + str(epoch) + " time: " + str(time.time() - epoch_start)
		scores += [score]
		epoch += 1
		epoch_start = time.time()
	score_avg = sum(scores)/len(scores)
	current_params['score_avg'] = score_avg
	print "Score average: " + str(score_avg)
	if score_avg > best_score:
		print "NEW MAX SCORE: " + str(score_avg) 
		best_score = score_avg
		best_params = current_params
		with open('best_ffnn.json', 'a') as out:
			json.dump(best_params, out)
			out.write('\n')
	remaining_models -= 1
	print "Total time: " + str((time.time() - start))
	n = (number_of_models - remaining_models)
	avg_time = (n-1)*avg_time/n + (time.time() - start)
	print "Average time: " + str(avg_time)
	print "Estimated remining time: " + str(avg_time*remaining_models)

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
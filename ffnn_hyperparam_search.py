import itertools, pickle, time, json, random
import numpy as np

from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA

from ffnn.ffnn import FFNN

################################################
# Initialization/helper functions              #
################################################

train_inputs_path = 'data/train_inputs_pca.npy'
train_outputs_path = 'data/train_outputs.npy'
test_inputs_path = 'data/test_inputs_pca.npy'
test_outputs_path = 'data/test_outputs.npy'

train_inputs = np.load(train_inputs_path)
train_outputs = np.load(train_outputs_path)
test_inputs = np.load(test_inputs_path)
test_outputs = np.load(test_outputs_path)

def list_permutations(max_length, min_length, max_nodes, min_nodes, node_skip=100):
	lst = []
	for i in range(min_length, max_length):
		lst += [list(x) + [10] for x in itertools.product(range(min_nodes, max_nodes, node_skip), repeat=i)]
	return lst

################################################
# Configuration settings — change as necessary #
################################################
layer_ns = list_permutations(max_length=3, min_length=1, max_nodes=51, min_nodes=10, node_skip=5)
params = {
	"layer_ns": layer_ns, 
	"step_size": [.1,.01,.001,.0001,.00001,.000001,.0000001], 
	"termination": [.1,.01,.001,.0001,.00001,.000001,.0000001],
	"dropout": [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75]
	}
max_example = None
batch_size = 1
################################################

best_score = 0.0
best_params = None
best_model = None
avg_time = 0

models = [(architecture, step_size, termination, dropout) for architecture in params['layer_ns'] for step_size in params['step_size'] for termination in params['termination'] for dropout in params['dropout']]
#models = [([25,10],.0001,0.00000001,0.0)]

number_of_models = len(models)
remaining_models = number_of_models

print "Total number of models: " + str(number_of_models)

for i in range(number_of_models):
	model = random.choice(models)
	architecture, step_size, termination, dropout = model
	models.remove(model)
	current_params = {'layer_ns': architecture, 'step_size': step_size, 'termination': termination, 'batch_size': batch_size, 'max_example': max_example, 'dropout': dropout}
	print "##########################################################################"
	print "Model #%d" % remaining_models
	print current_params
	print "##########################################################################"
	start = time.time()
	scores = []
	cv = 1
	cv_start = time.time()
	for train_index, test_index in KFold(train_inputs[:max_example].shape[0], n_folds=3):
		X_train, X_test = train_inputs[train_index], train_inputs[test_index]
		y_train, y_test = train_outputs[train_index], train_outputs[test_index]
		ffnn = FFNN(X_train[:batch_size], architecture, step_size, termination, dropout=dropout)
		ffnn.fit(X_train, y_train, X_test, y_test, batch_size=batch_size)
		score = ffnn.score(X_test, y_test)
		print "CV " + str(cv) + " score: " + str(score)
		print "CV " + str(cv) + " time: " + str(time.time() - cv_start)
		scores += [score]
		cv += 1
		cv_start = time.time()
	score_avg = sum(scores)/len(scores)
	current_params['score_avg'] = score_avg
	current_params['train_time'] = str((time.time() - start))
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
	n = float((number_of_models - remaining_models))
	print "n: %d" % n
	avg_time = (((n-1.0)/n)*avg_time) + (time.time() - start)/n
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
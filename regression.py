import time
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

from data.get_data import get_data

################################################
# Configuration settings — change as necessary #
################################################
verbose = True
timer = True

params = [{}]

train_limit = 5000
test_limit = 5000
kfolds = 3
################################################

data = get_data()

if timer: start_time = time.time()
if verbose: print "Start Training"
log_reg = LogisticRegression()
grid = GridSearchCV(log_reg, params, cv=kfolds)
grid.fit(data.train_inputs[:train_limit], data.train_outputs[:train_limit])
if verbose: print "End Training"

if verbose: print "Start Scoring"
print grid.score(data.test_inputs[:test_limit], data.test_outputs[:test_limit])
if verbose: print "End Scoring"
if timer: print "Elapsed time: " + str(time.time() - start_time)

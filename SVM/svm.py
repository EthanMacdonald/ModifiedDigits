import numpy as np
from sklearn.decomposition import PCA

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold


"""
run pca
## use SVMS with gaussian RBF kernels
## two parameters:
tweak cost of misclassification -- ie where to set the trade off between
making the dividing hyperplane more complicated versus how much we want the plane to divide the training dataset exactly). 
gamma -- i.e.e how tight we want the guassian rbf to be 
"""



DATA_PATH_TRAIN = "../DATA/data_as_images/train_images_subset" 
DATA_PATH_TEST = "../DATA/data_as_images/text_images_subset"

NUM_OF_TRAIN = 1000
NUM_OF_TEST = 200


def read_data_run_pca(test=False):
	X=[]
	Y=[]

	with open(DATA_PATH_TRAIN, 'r') as tr:
		X.append(f.readline())

	with open(DATA_PATH_TEST, 'r') as te: 
		Y.append(te.split(",")[1]

	##PCR REDUCTION FIRST 

	if (not test):
		X = #array of images
		pca = PCA(n_components=2)
		X_reduced = pca.fit(X)

		Y = 


	


if __name__== 'main':


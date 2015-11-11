import numpy as np

from sklearn.decomposition import RandomizedPCA

train_inputs_path = 'data/train_inputs.npy'
test_inputs_path = 'data/test_inputs.npy'

train_inputs = np.load(train_inputs_path)
test_inputs = np.load(test_inputs_path)

pca = RandomizedPCA(whiten=True)
pca_train= pca.fit_transform(train_inputs)
pca_test = pca.transform(test_inputs)

np.save('data/train_inputs_pca', pca_train)
np.save('data/test_inputs_pca', pca_test)
import numpy as np

from ffnn.ffnn import FFNN

X = np.asarray([[0,0],[0,1],[1,0],[1,1]])
y = np.asarray([[1,0],[0,1],[0,1],[1,0]])

ffnn = FFNN(X,[4,2],.1,.01)
ffnn.fit(X,y,X,y,batch_size=1)
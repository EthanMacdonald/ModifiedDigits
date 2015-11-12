from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


train_in = np.load('../data/train_inputs.npy')#.reshape(-1, 1, 48, 48)

train_out = np.load('../data/train_outputs.npy')
test_in = np.load('../data/test_inputs.npy')#.reshape(-1, 1, 48, 48)
test_out = np.load('../data/test_outputs.npy')


c=2.0
kernel = 'rbf'
params=[]


def plot_confusion_matrix(cm, title, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(5)
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def svm(X, Y, X_test, Y_test, pca=False):

	s = train_in.shape[0]
	X_tr = X[0:4*s/5]
	X_test = X[4*s/5:s]

	Y_tr = Y[0:4*s/5]
	Y_test = Y[4*s/5:s]

	print X_tr.shape, "Xshape"
	print Y_tr.shape, "Yshape"
	print X_test.shape, "test X shape"
	print Y_test.shape,"test Y shape"
	##########################
	param_grid = {'kernel':['rbf'], 'C':[1, 10], 'gamma':[.1,.01,.001]}
	param_grid2 = {'kernel':['linear'], 'C':[1, 10]}
	##########################

	grid = GridSearchCV(SVC(), param_grid=param_grid)
	grid.fit(X_tr,Y_tr)

	#clf = SVC(C=1.0, kernel='rbf',gamma=0.1)
	#clf.fit(X_tr,Y_tr)
	Y_predict = clf.predict(X_test)
	print Y_predict, "asdfasdf"

	writer = csv.writer(open('svm_test.csv','w'))
	writer.writerow([int(x) for x in Y_predict])

	#print clf.score(X_test,Y_test)

	print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_)) 

	#title = grid.best_params_
	##confusion matrix assembly:
	#cm = confusion_matrix(Y_test, Y_predict)
	#plot_confusion_matrix(cm,'%s'%title) 
	#plt.show()

	#plt.savefig("%s.png"),%(param)
	#plt.savefig("hello")

if __name__ == '__main__':
	svm(train_in, train_out, test_in, test_out)


# 
# 

import h5py
import sys,os
import numpy as np
from sklearn import cluster
from sklearn import metrics

# contaainer_train = str(sys.argv[1])
# container_test = str(sys.argv[2])

container_train = 'training_set_01.hdf5'
container_test = 'testing_set_01.hdf5'

mnist_train = h5py.File(container_train)
mnist_test = h5py.File(container_test)

def purity(truth, pred):
	conf = metrics.confusion_matrix(truth,pred)
	t = 0
	for i in range(0,10):
		t += np.amax(conf[i,:])
	return float(t)/np.sum(conf)

mnist_estimator_feats1 = cluster.KMeans(10, n_jobs=-1)
mnist_estimator_feats2 = cluster.KMeans(10, n_jobs=-1)
mnist_estimator_feats1.fit(mnist_train['feats_1'])
mnist_estimator_feats2.fit(mnist_train['feats_2'])

print('Clustering performance evaluation - train_feats_1')
print('Homogeneity Score: '+ str(metrics.homogeneity_score(mnist_train['label'],mnist_estimator_feats1.labels_)))
print('Completeness Score: '+ str(metrics.completeness_score(mnist_train['label'],mnist_estimator_feats1.labels_)))
print('Adjusted Mutual Information Score: '+  str(metrics.adjusted_mutual_info_score(mnist_train['label'],mnist_estimator_feats1.labels_)))
print('Adjusted Rand Index Score: '+  str(metrics.adjusted_rand_score(mnist_train['label'],mnist_estimator_feats1.labels_)))
print('Purity: '+str(purity(mnist_train['label'],mnist_estimator_feats1.labels_)))

print('\nClustering performance evaluation - train_feats_2')
print('Homogeneity Score: '+ str(metrics.homogeneity_score(mnist_train['label'],mnist_estimator_feats2.labels_)))
print('Completeness Score: '+ str(metrics.completeness_score(mnist_train['label'],mnist_estimator_feats2.labels_)))
print('Adjusted Mutual Information Score: '+  str(metrics.adjusted_mutual_info_score(mnist_train['label'],mnist_estimator_feats2.labels_)))
print('Adjusted Rand Index Score: '+  str(metrics.adjusted_rand_score(mnist_train['label'],mnist_estimator_feats2.labels_)))
print('Purity: '+str(purity(mnist_train['label'],mnist_estimator_feats2.labels_)))

# heatmap = np.zeros((10,10))
pred1 = mnist_estimator_feats1.predict(mnist_test['feats_1'])
pred2 = mnist_estimator_feats2.predict(mnist_test['feats_2'])

heatmap = metrics.confusion_matrix(pred1,pred2)




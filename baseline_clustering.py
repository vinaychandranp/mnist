# 
# 

import h5py
import sys,os
from sklearn import cluster
from sklearn import metrics

container = 'testing_set_01.hdf5'
# container = str(sys.argv[1])
mnist = h5py.File(container)

mnist_estimator = cluster.KMeans(10, n_jobs=-1)


mnist_estimator.fit(mnist['feats_1'])
print('Clustering performance evaluation - feats_1')
print('Homogeneity Score: '+ str(metrics.homogeneity_score(mnist['label'],mnist_estimator.labels_)))
print('Completeness Score: '+ str(metrics.completeness_score(mnist['label'],mnist_estimator.labels_)))
print('Adjusted Mutual Information Score: '+  str(metrics.adjusted_mutual_info_score(mnist['label'],mnist_estimator.labels_)))
print('Adjusted Rand Index Score: '+  str(metrics.adjusted_rand_score(mnist['label'],mnist_estimator.labels_)))


mnist_estimator.fit(mnist['feats_2'])
print('Clustering performance evaluation - feats_2')
print('Homogeneity Score: '+ str(metrics.homogeneity_score(mnist['label'],mnist_estimator.labels_)))
print('Completeness Score: '+ str(metrics.completeness_score(mnist['label'],mnist_estimator.labels_)))
print('Adjusted Mutual Information Score: '+  str(metrics.adjusted_mutual_info_score(mnist['label'],mnist_estimator.labels_)))
print('Adjusted Rand Index Score: '+  str(metrics.adjusted_rand_score(mnist['label'],mnist_estimator.labels_)))


# 
# 

import h5py, pickle
import sys,os
import numpy as np
from sklearn import cluster
from sklearn import metrics

container_train = str(sys.argv[1])
container_test = str(sys.argv[2])

# container_train = 'training_set_01.hdf5'
# container_test = 'testing_set_01.hdf5'

mnist_train = h5py.File(container_train)
mnist_test = h5py.File(container_test)

def purity(truth, pred):
	conf = metrics.confusion_matrix(truth,pred)
	t = 0
	for i in range(0,10):
		t += np.amax(conf[i,:])
	return float(t)/np.sum(conf)

def print_stats(truth, pred):
	print('Homogeneity Score: '+ str(metrics.homogeneity_score(truth,pred)))
	print('Completeness Score: '+ str(metrics.completeness_score(truth,pred)))
	print('Adjusted Mutual Information Score: '+  str(metrics.adjusted_mutual_info_score(truth,pred)))
	print('Adjusted Rand Index Score: '+  str(metrics.adjusted_rand_score(truth,pred)))
	print('Purity: '+str(purity(truth,pred)))

def compute_heatmap(pred1,pred2):
	size = len(pred1)
	heatmap_ = np.zeros((10,10))
	for i in range(0,size):
		heatmap_[pred1[i],pred2[i]] += 1
	return heatmap_.astype(int)



mnist_estimator_feats1 = cluster.KMeans(10, n_jobs=-1)
mnist_estimator_feats2 = cluster.KMeans(10, n_jobs=-1)
mnist_estimator_feats1.fit(mnist_train['feats_1'])
mnist_estimator_feats2.fit(mnist_train['feats_2'])

print('Clustering performance evaluation - train_feats_1')
print_stats(mnist_train['label'],mnist_estimator_feats1.labels_)

print('\nClustering performance evaluation - train_feats_2')
print_stats(mnist_train['label'],mnist_estimator_feats2.labels_)


pred1 = mnist_estimator_feats1.predict(mnist_test['feats_1'])
pred2 = mnist_estimator_feats2.predict(mnist_test['feats_2'])

global_heatmap = compute_heatmap(pred1,pred2)

# Compute clusterwise heatmap
label = mnist_test['label']

indices = []
for i in range(0,10):
	indices.append([j for j,x in enumerate(label) if x == i ])

heatmaps = np.zeros((10,10,10))

for i in range(0,10):
	pred1 = mnist_estimator_feats1.predict([x for j,x in enumerate(mnist_test['feats_1']) if j in indices[i]])
	pred2 = mnist_estimator_feats2.predict([x for j,x in enumerate(mnist_test['feats_2']) if j in indices[i]])
	heatmaps[i] = compute_heatmap(pred1,pred2)
	# print(np.shape(metrics.confusion_matrix(pred1,pred2)))

pickle.dump((global_heatmap,heatmaps), open('heatmaps.p','wb'))
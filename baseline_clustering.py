#
#

import h5py
import pickle
import sys
import numpy as np
import argparse
from sklearn import cluster
from sklearn import metrics

parser = argparse.ArgumentParser()
parser.add_argument('-tr', '--train', metavar='FILE', dest='trainfile',
                    required=False, default=None,
                    help='HDF5 container for train data')
parser.add_argument('-te', '--test', metavar='FILE', dest='testfile',
                    help='HDF5 container for test data')
parser.add_argument('-nc', '--num_class', metavar='FILE', required=False,
                    dest='nc', default=10, help='Number of classes ')
opts = parser.parse_args(sys.argv[1:])
container_train = opts.trainfile
container_test = opts.testfile
nc = int(opts.nc)

# container_train = sys.argv[1]
# container_test = sys.argv[2]
# nc = int(sys.argv[3]x)

# container_train = 'training_set_01.hdf5'
# container_test = 'testing_set_01.hdf5'

mnist_train = h5py.File(container_train, 'r')
mnist_test = h5py.File(container_test, 'r')


def purity(truth, pred):
    conf = metrics.confusion_matrix(truth, pred)
    t = 0
    for i in range(0, nc):
        t += np.amax(conf[i, :])
    return float(t)/np.sum(conf)


def print_stats(truth, pred):
    print('Homogeneity Score: ' + str(metrics.homogeneity_score(truth, pred)))
    print('Completeness Score: ' +
          str(metrics.completeness_score(truth, pred)))
    print('Adjusted Mutual Information Score: ' +
          str(metrics.adjusted_mutual_info_score(truth, pred)))
    print('Adjusted Rand Index Score: ' +
          str(metrics.adjusted_rand_score(truth, pred)))
    print('Purity: ' +
          str(purity(truth, pred)))


def compute_heatmap(pred1, pred2):
    size = len(pred1)
    heatmap_ = np.zeros((nc, nc))
    for i in range(0, size):
        heatmap_[pred1[i], pred2[i]] += 1
    return heatmap_.astype(int)


mnist_estimator_feats1 = cluster.KMeans(nc, n_jobs=-1)
mnist_estimator_feats2 = cluster.KMeans(nc, n_jobs=-1)
mnist_estimator_feats1.fit(mnist_train['feats_1'])
mnist_estimator_feats2.fit(mnist_train['feats_2'])

print('Clustering performance evaluation - train_feats_1')
print_stats(mnist_train['label'], mnist_estimator_feats1.labels_)

print('\nClustering performance evaluation - train_feats_2')
print_stats(mnist_train['label'], mnist_estimator_feats2.labels_)

pred1 = mnist_estimator_feats1.predict(mnist_test['feats_1'])
pred2 = mnist_estimator_feats2.predict(mnist_test['feats_2'])

global_heatmap = compute_heatmap(pred1, pred2)

# Compute clusterwise heatmap
label = mnist_test['label']

indices = []
for i in range(0, nc):
    indices.append([j for j, x in enumerate(label) if x == i])

heatmaps = np.zeros((nc, nc, nc))

for i in range(0, nc):
    pred1 = mnist_estimator_feats1.predict([x for j, x in
                                            enumerate(mnist_test['feats_1'])
                                            if j in indices[i]])
    pred2 = mnist_estimator_feats2.predict([x for j, x in
                                            enumerate(mnist_test['feats_2'])
                                            if j in indices[i]])
    heatmaps[i] = compute_heatmap(pred1, pred2)


pickle.dump((global_heatmap, heatmaps), open('heatmaps.p', 'wb'))

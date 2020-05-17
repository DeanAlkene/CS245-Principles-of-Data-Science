import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metric_learn import ITML_Supervised
import sklearn
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import sys

sys.path.append("..")
from processData import loadDataDivided
import KNN

def runITML(X_train, X_test, y_train, y_test, n_cons):
    print("\nn_comp=%d\n"%(n_cons))
    transformer = ITML_Supervised(num_constraints=n_cons)
    transformer.fit(X_train, y_train)
    X_train_proj = transformer.transform(X_train)
    X_test_proj = transformer.transform(X_test)
    if n_cons == 2:
        np.save('X_train_'+str(n_cons)+'_ITMLt', X_train_proj)
        np.save('X_test_'+str(n_cons)+'_ITMLt', X_test_proj)
    return X_train_proj, X_test_proj, transformer

def cosine(x, y):
    s = np.linalg.norm(x, ord=2) * np.linalg.norm(y, ord=2)
    if s == 0:
        return 0
    return 1 - np.dot(x, y) / s

def main():
    dim_range = [2, 5, 10, 20, 50, 75, 100, 200, 300]
    k_range = [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 50, 100, 200, 500, 1000]

    X_train, X_test, y_train, y_test = loadDataDivided(ifSubDir=True, ifScale=True, suffix='')
    for dim in dim_range:
        print("dim: %d, method: ITML, metric: %s" % (dim, "true"))
        X_train_proj, X_test_proj, transformer = runITML(X_train, X_test, y_train, y_test, dim)
        #KNN.runKNN(X_train_proj, X_test_proj, y_train, y_test, k_range, metric='euclidean', metric_params=None,
        #           label=str(dim) + '_ITML_euclidean')
        print("Checkpoint")
        KNN.runKNN(X_train, X_test, y_train, y_test, k_range, metric=transformer.get_metric(), metric_params=None,
                   label=str(dim) + '_ITML_true')


if __name__ == '__main__':
    main()

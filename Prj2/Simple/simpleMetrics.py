import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_distances
import sys
sys.path.append("..")
from processData import loadDataDivided
import KNN

def main():
    kernel_range = ['linear', 'rbf', 'poly', 'sigmoid', 'cosine']
    dim_range = [50, 500, 2048]
    k_range = [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 50, 100, 200, 500, 1000]
    metric_range = ['euclidean', 'manhattan', 'chebyshev']
    for dim in dim_range:
        for kernel in kernel_range:
            if dim != 2048:
                X_train, X_test, y_train, y_test = loadDataDivided(ifSubDir=True, ifScale=True, suffix='_' + str(dim) + '_' + kernel)
            else:
                X_train, X_test, y_train, y_test = loadDataDivided(ifSubDir=True, ifScale=True, suffix='')
            for metric in metric_range:
                print("dim: %d, kernel: %s, metric: %s" % (dim, kernel, metric))
                KNN.runKNN(X_train, X_test, y_train, y_test, k_range, metric=metric, metric_params=None, label=str(dim) + '_' + kernel + '_' + metric)
    
    for dim in dim_range:
        for kernel in kernel_range:
            if dim != 2048:
                X_train, X_test, y_train, y_test = loadDataDivided(ifSubDir=True, ifScale=True, suffix='_' + str(dim) + '_' + kernel)
            else:
                X_train, X_test, y_train, y_test = loadDataDivided(ifSubDir=True, ifScale=True, suffix='')
            print("dim: %d, kernel: %s, metric: %s" % (dim, kernel, "cosine"))
            KNN.runKNN(X_train, X_test, y_train, y_test, k_range, metric=cosine_distances, metric_params=None, label=str(dim) + '_' + kernel + '_cosine')

if __name__ == '__main__':
    main()
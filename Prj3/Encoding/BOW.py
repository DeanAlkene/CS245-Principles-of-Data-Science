import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
import sys
sys.path.append("..")
import SVMmodel

y_file_name = "../AwA2-data/AwA2-labels.txt"

f_class_dict = np.load('../f_class_dict.npy', allow_pickle=True).item()
ld_sample = np.load('../LD_for_clustering.npy', allow_pickle=True)

def BOW(k):
    feature = []
    model = KMeans(n_clusters=k, copy_x=False, n_jobs=8)
    model.fit(ld_sample)

    for className, totalNum in f_class_dict.items():
        print("SS at %s" % (className))
        for idx in range(10001, totalNum + 1):
            ld = np.load(className + '/' + className + '_' + str(idx) + '.npy', allow_pickle=True)  # 2d np array
            bow = np.zeros((1, k))
            for des in ld:
                bow[model.predict(des.reshape(1, -1))[0]] += 1
            feature.append(bow)
    return np.vstack(feature)

def scale(feature, norm_method):
    if norm_method == 'L2':
        feature_scaled = Normalizer().fit_transform(feature)
    elif norm_method == 'Z-score':
        feature_scaled = StandardScaler().fit_transform(feature)
    elif norm_method == 'MinMax':
        feature_scaled = MinMaxScaler().fit_transform(feature)
    elif norm_method == 'L2+Z-score':
        feature_scaled_0 = Normalizer().fit_transform(feature)
        feature_scaled = StandardScaler().fit_transform(feature_scaled_0)
    elif norm_method == 'L2+MinMax':
        feature_scaled_0 = Normalizer().fit_transform(feature)
        feature_scaled = MinMaxScaler().fit_transform(feature_scaled_0)
    else:
        feature_scaled = feature
    return feature_scaled

def main():
    k_range = [32, 64, 128, 256, 512, 1024, 2048]
    scale_range = ['none', 'L2', 'Z-score', 'MinMax', 'L2+Z-score', 'L2+MinMax']
    C_range = [0.001, 0.01, 0.1, 1.0, 10]
    for k in k_range:
        print("BOW, k:%d" % (k))
        X = BOW(k)
        col_name = ['feature' + str(i) for i in range(k)]
        y = pd.read_csv(y_file_name, names=['label'])
        for method in scale_range:
            X_scaled = scale(X, method)
            X_scaled = pd.DataFrame(data=X_scaled, columns=col_name)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4)
            for C in C_range:
                linear_score = SVMmodel.runSVM(X_train, X_test, y_train, y_test, C, 'linear')
                rbf_score = SVMmodel.runSVM(X_train, X_test, y_train, y_test, C, 'rbf')
                with open('res_BOW.txt', "a") as f:
                    f.write("BOW with k=%d, scale=%s, SVM with %d kernel, C=%f, score=%f\n"%(k, method, 'linear', C, linear_score))
                    f.write("BOW with k=%d, scale=%s, SVM with %d kernel, C=%f, score=%f\n"%(k, method, 'rbf', C, rbf_score))
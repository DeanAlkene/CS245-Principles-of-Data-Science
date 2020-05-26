import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sys
sys.path.append("..")
import SVMmodel

y_file_name = "../AwA2-data/AwA2-labels.txt"

f_class_dict = np.load('../f_class_dict.npy').item()
ld_sample = np.load('../LD_for_clustering.npy')

def VLAD(k):
    feature = []
    model = KMeans(n_clusters=k, copy_x=False, n_jobs=8)
    model.fit(ld_sample)

    for className, totalNum in f_class_dict.items():
        print("SS at %s" % (className))
        for idx in range(10001, totalNum + 1):
            ld = np.load(className + '_' + str(idx) + '.npy')  # 2d np array
            vlad = [np.zeros((1, ld.shape[1])) for i in range(k)]
            for des in ld:
                label = model.predict(des)[0]
                vlad[label] += des - model.cluster_centers_[label]
            vlad = np.hstack(vlad)
            feature.append(vlad)
    return np.vstack(feature)

def main():
    k_range = [8, 16, 32, 64]
    C_range = [0.001, 0.01, 0.1, 1.0, 10]
    pca = KernelPCA(n_components=256, kernel='linear')
    lda = LinearDiscriminantAnalysis(n_components=40)
    for k in k_range:
        print("VLAD, k:%d" % (k))
        X = VLAD(k)
        X = StandardScaler().fit_transform(X)
        col_name = ['feature' + str(i) for i in range(X.shape[1])]
        X = pd.DataFrame(data=X, columns=col_name)
        y = pd.read_csv(y_file_name, names=['label'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

        print("PCA")
        pca.fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        for C in C_range:
            linear_score = SVMmodel.runSVM(X_train_pca, X_test_pca, y_train, y_test, C, 'linear')
            rbf_score = SVMmodel.runSVM(X_train_pca, X_test_pca, y_train, y_test, C, 'rbf')
            with open('res_VLAD_PCA.txt', "a") as f:
                f.write("VLAD with k=%d, Z-score, SVM with %d kernel, C=%f, score=%f\n"%(k, 'linear', C, linear_score))
                f.write("VLAD with k=%d, Z-score, SVM with %d kernel, C=%f, score=%f\n" % (k, 'rbf', C, rbf_score))

        print("LDA")
        lda.fit(X_train)
        X_train_lda = lda.transform(X_train)
        X_test_lda = lda.transform(X_test)
        for C in C_range:
            linear_score = SVMmodel.runSVM(X_train_lda, X_test_lda, y_train, y_test, C, 'linear')
            rbf_score = SVMmodel.runSVM(X_train_lda, X_test_lda, y_train, y_test, C, 'rbf')
            with open('res_VLAD_LDA.txt', "a") as f:
                f.write("VLAD with k=%d, Z-score, SVM with %d kernel, C=%f, score=%f\n"%(k, 'linear', C, linear_score))
                f.write("VLAD with k=%d, Z-score, SVM with %d kernel, C=%f, score=%f\n"%(k, 'rbf', C, rbf_score))
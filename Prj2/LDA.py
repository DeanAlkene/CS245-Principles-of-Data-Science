import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sys
sys.path.append("..")
from processData import loadDataDivided

def runLDA(X_train, X_test, y_train, y_test, comp_range):
    rbf_scores = []
    linear_scores = []
    for n_comp in comp_range:
        print("\nn_comp=%d\n"%(n_comp))
        transformer = LinearDiscriminantAnalysis(solver='svd', n_components=n_comp)
        transformer.fit(X_train, y_train)
        X_train_proj = transformer.transform(X_train)
        X_test_proj = transformer.transform(X_test)
        np.save('X_train_'+str(n_comp)+'_LDA', X_train_proj)
        np.save('X_test_'+str(n_comp)+'_LDA', X_test_proj)
    return 0

def main():
    comp_range = [2, 3, 5, 10, 20, 30, 40]
    X_train, X_test, y_train, y_test = loadDataDivided(ifSubDir=False)
    runLDA(X_train, X_test, y_train, y_test, comp_range)

if __name__ == '__main__':
    main()
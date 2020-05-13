import numpy as np
from metric_learn import NCA
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
import sys
sys.path.append("..")
from processData import loadDataDivided

def runNCA(X_train, X_test, y_train, y_test, comp_range):
    for n_comp in comp_range:
        print("\nn_comp=%d\n"%(n_comp))
        transformer = NCA(max_iter=1000,n_components=n_comp)
        transformer.fit(X_train, y_train)
        X_train_proj = transformer.transform(X_train)
        X_test_proj = transformer.transform(X_test)
        np.save('X_train_'+str(n_comp)+'_NCA', X_train_proj)
        np.save('X_test_'+str(n_comp)+'_NCA', X_test_proj)
    return 0

def main():
    comp_range = [2, 3, 5, 10, 20, 30, 40]
    X_train, X_test, y_train, y_test = loadDataDivided(ifSubDir=False)
    runNCA(X_train, X_test, y_train, y_test, comp_range)

if __name__ == '__main__':
    main()

# nca = NCA(max_iter=1000,n_components=2)
# X, y = make_classification()
# nca.fit(X, y)
# knn = KNeighborsClassifier(metric=nca.get_metric())
# knn.fit(X, y)
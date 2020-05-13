import numpy as np
from metric_learn import LSML_Supervised
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
import sys
sys.path.append("..")
from processData import loadDataDivided

def runLSML(X_train, X_test, y_train, y_test, cons_range):
    for n_cons in cons_range:
        print("\nn_comp=%d\n"%(n_cons))
        transformer = LSML_Supervised(num_constraints=n_cons)
        transformer.fit(X_train, y_train)
        X_train_proj = transformer.transform(X_train)
        X_test_proj = transformer.transform(X_test)
        np.save('X_train_'+str(n_cons)+'_LSML', X_train_proj)
        np.save('X_test_'+str(n_cons)+'_LSML', X_test_proj)
    return 0

def main():
    cons_range = [2, 10, 50, 75, 100, 200, 300]
    X_train, X_test, y_train, y_test = loadDataDivided(ifSubDir=False)
    runLSML(X_train, X_test, y_train, y_test, cons_range)

if __name__ == '__main__':
    main()

# nca = NCA(max_iter=1000,n_components=2)
# X, y = make_classification()
# nca.fit(X, y)
# knn = KNeighborsClassifier(metric=nca.get_metric())
# knn.fit(X, y)
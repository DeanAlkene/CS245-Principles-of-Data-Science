import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

X_file_name = "Animals_with_Attributes2/Features/ResNet101/AwA2-features.txt"
y_file_name = "Animals_with_Attributes2/Features/ResNet101/AwA2-labels.txt"
col_name = ['feature' + str(i) for i in range(2048)]
X_data = pd.read_csv(X_file_name, sep=' ', nrows=2000, names=col_name)
y_data = pd.read_csv(y_file_name, nrows=2000, names=['label'])
#X_data = pd.read_csv(X_file_name, sep=' ', names=col_name)
#y_data = pd.read_csv(y_file_name, names=['label'])

def coarseTuning(X, y, k=5):
    C_range = np.logspace(-5, 5, 11)
    cv_scores_rbf = []
    cv_scores_linear = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    for c in C_range:
        print("\nCross-Validation rbf: C=%f\n"%(c))
        model = SVC(C=c, kernel='rbf', gamma='auto', verbose=False)
        score = cross_val_score(model, X_train, y_train, cv=k, scoring='accuracy')
        cv_scores_rbf.append(score.mean())
    bestC_rbf = C_range[cv_scores_rbf.index(max(cv_scores_rbf))]
    rbfPlt = plt.plot(C_range, cv_scores_rbf, 'bo-', linewidth=2)
    rbfPlt.title('SVM with rbf kernel')
    rbfPlt.xlabel('C')
    rbfPlt.xscale('log')
    rbfPlt.xticks(np.logspace(-6, 6, 13))
    rbfPlt.ylabel('Accuracy')
    rbfPlt.savefig('CoarseTuningParam_rbf.jpg')

    for c in C_range:
        print("\nCross-Validation linear: C=%f\n"%(c))
        model = SVC(C=c, kernel='linear', gamma='auto', verbose=False)
        score = cross_val_score(model, X_train, y_train, cv=k, scoring='accuracy')
        cv_scores_linear.append(score.mean())
    bestC_linear = C_range[cv_scores_linear.index(max(cv_scores_linear))]
    linearPlt = plt.plot(C_range, cv_scores_linear, 'bo-', linewidth=2)
    linearPlt.title('SVM with linear kernel')
    linearPlt.xlabel('C')
    linearPlt.xscale('log')
    linearPlt.xticks(np.logspace(-6, 6, 13))
    linearPlt.ylabel('Accuracy')
    linearPlt.savefig('CoarseTuningParam_linear.jpg')

    bestModel_rbf = SVC(C=bestC_rbf, kernel='rbf', gamma='auto', verbose=False)
    bestModel_rbf.fit(X_train, y_train)
    bestAcc_rbf = bestModel_rbf.score(X_test, y_test)

    bestModel_linear = SVC(C=bestC_linear, kernel='linear', gamma='auto', verbose=False)
    bestModel_linear.fit(X_train, y_train)
    bestAcc_linear = bestModel_linear.score(X_test, y_test)

    with open('res_coarse.txt', 'w') as f:
        f.write("rbf:\n")
        for i in range(len(C_range)):
            f.write("rbf: C = %f, acc = %f\n"%(C_range[i], cv_scores_rbf[i]))
        f.write("rbf: Best C = %f\n"%(bestC_rbf))
        f.write("rbf: acc = %f\n"%(bestAcc_rbf))
        for i in range(len(C_range)):
            f.write("linear: C = %f, acc = %f\n"%(C_range[i], cv_scores_linear[i]))
        f.write("linear: Best C = %f\n"%(bestC_linear))
        f.write("linear: acc = %f\n"%(bestAcc_linear))

def fineTuning(X, y, k=5):
    pass

def main():
    coarseTuning(X_data, y_data)

if __name__ == '__main__':
    main()

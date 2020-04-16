import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sys
sys.path.append("..")
from processData import loadDataDivided
import SVMmodel

def runSelectKBest(X_train, X_test, y_train, y_test, comp_range):
    rbf_scores = []
    linear_scores = []
    for n_comp in comp_range:
        print("\nn_comp=%d\n"%(n_comp))

        selector = SelectKBest(chi2, k=n_comp)
        selector.fit(X_train, y_train)
        X_train_sel = selector.transform(X_train)
        X_test_sel = selector.transform(X_test)

        if n_comp == 2:
            np.save('X_train_sel_2d_selectKBest', X_train_sel)
            np.save('X_test_sel_2d_selectKBest', X_test_sel)
        score_rbf = SVMmodel.runSVM(X_train_sel, X_test_sel, y_train, y_test, SVMmodel.getBestParam('rbf'), 'rbf')
        rbf_scores.append(score_rbf.mean())
        score_linear = SVMmodel.runSVM(X_train_sel, X_test_sel, y_train, y_test, SVMmodel.getBestParam('linear'), 'linear')
        linear_scores.append(score_linear.mean())
    return rbf_scores, linear_scores

def draw(comp_range, scores, kernel):
    bestIdx = np.argmax(scores)
    bestNComp = comp_range[bestIdx]
    bestAcc = scores[bestIdx]
    with open('res_selectKBest_' + kernel + '.txt', 'w') as f:
        for i in range(len(comp_range)):
            f.write(kernel + ": n_comp = %f, acc = %f\n"%(comp_range[i], scores[i]))
        f.write(kernel + ": Best n_comp = %f\n"%(bestNComp))
        f.write(kernel + ": acc = %f\n"%(bestAcc))

    plt.figure()
    plt.plot(comp_range, scores, 'bo-', linewidth=2)
    plt.title('selectKBest with SVM ' + kernel + ' kernel')
    plt.xlabel('n_components')
    plt.ylabel('Accuracy')
    plt.savefig('selectKBest_' + kernel + '.jpg')

def main():
    comp_range = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                  1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
    X_train, X_test, y_train, y_test = loadDataDivided(ifSubDir=True)
    rbf_scores, linear_scores = runSelectKBest(X_train, X_test, y_train, y_test, comp_range)
    draw(comp_range, rbf_scores, 'rbf')
    draw(comp_range, linear_scores, 'linear')

if __name__ == '__main__':
    main()

import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC
import sys
sys.path.append("..")
from processData import loadDataDivided
import model

class PCAThread(threading.Thread):
    def __init__(self, X_train, X_test, y_train, y_test, comp_range, Kernel):
        threading.Thread.__init__(self)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.comp_range = comp_range
        self.Kernel = Kernel
        self.C = model.getBestParam(Kernel)

    def run(self):
        scores = []
        for n_comp in self.comp_range:
            print("\nn_comp=%d\n"%(n_comp))
            transformer = KernelPCA(n_components=n_comp, kernel=self.Kernel)
            transformer.fit(self.X_train)
            X_train_proj = transformer.transform(self.X_train)
            X_test_proj = transformer.transform(self.X_test)
            if n_comp == 2:
                np.save('X_train_proj_2d_' + self.Kernel, X_train_proj)
                np.save('X_test_proj_2d_' + self.Kernel, X_test_proj)
            score = model.runSVM(X_train_proj, X_test_proj, self.y_train, self.y_test, self.C, self.Kernel)
            scores.append(score.mean())
        np.save('PCA_scores_' + self.Kernel, scores)

def draw(comp_range, scores, kernel):
    bestIdx = np.argmax(scores)
    bestNComp = comp_range[bestIdx]
    bestAcc = scores[bestIdx]
    with open('res_PCA_' + kernel + '.txt', 'w') as f:
        for i in range(len(comp_range)):
            f.write(kernel + ": n_comp = %f, acc = %f\n"%(comp_range[i], scores[i]))
        f.write(kernel + ": Best n_comp = %f\n"%(bestNComp))
        f.write(kernel + ": acc = %f\n"%(bestAcc))

    plt.figure()
    plt.plot(comp_range, scores, 'bo-', linewidth=2)
    plt.title('PCA with ' + kernel + ' kernel')
    plt.xlabel('n_components')
    plt.ylabel('Accuracy')
    plt.savefig('PCA_' + kernel + '.jpg')

def main():
    comp_range = [2, 5, 10, 20, 50, 100, 200, 500, 750, 1000, 1200, 1500, 1750, 2000]
    X_train, X_test, y_train, y_test = loadDataDivided(ifSubDir=True)
    t1 = PCAThread(X_train, X_test, y_train, y_test, comp_range, 'rbf')
    t2 = PCAThread(X_train, X_test, y_train, y_test, comp_range, 'linear')

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    rbf_scores = np.load('PCA_scores_rbf.npy')
    linear_scores = np.load('PCA_scores_linear.npy')
    draw(comp_range, rbf_scores, 'rbf')
    draw(comp_range, linear_scores, 'linear')

if __name__ == '__main__':
    main()

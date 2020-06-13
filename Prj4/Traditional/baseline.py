import sys
sys.path.append('..')
from utils import dataloader, SVM

def baseline():
    pairs = [('Art', 'RealWorld'), ('Clipart', 'RealWorld'), ('Product', 'RealWorld')]
    for p in pairs:
        print("%s->%s", (p[0], p[1]))
        X_train, y_train, X_test, y_test = dataloader.loadData(p[0], p[1])
        score = SVM.SVM(X_train, X_test, y_train, y_test)
        with open('baseline.txt', 'a') as f:
            f.write('%s->%s with acc=%f\n' % (p[0], p[1], score))
        print()

baseline()
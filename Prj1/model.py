import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.svm import SVC

def runSVM(X_train, X_test, y_train, y_test, C, kernel):
    model = SVC(C=C, kernel=kernel, gamma='auto', verbose=False)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return score
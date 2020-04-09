import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from processData import loadDataDivided

class PCAThread(threading.Thread):
    def __init__(self):
        super().__init__(self)

    def run(self):
        pass



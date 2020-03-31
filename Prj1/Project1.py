import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

x_file_name = "Animals_with_Attributes2/Features/ResNet101/AwA2-features.txt"
y_file_name = "Animals_with_Attributes2/Features/ResNet101/AwA2-labels.txt"
col_name = ['feature' + str(i) for i in range(2048)]
x_data = pd.read_csv(x_file_name, sep=' ', names=col_name)
y_data = pd.read_csv(y_file_name, names=['label'])
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, random_state=101)
x_train = x_train.values
y_train = y_train.values.ravel()
x_test = x_test.values
y_test = y_test.values
model = SVC(C=1.0, kernel='rbf', gamma='auto', verbose=True)
model.fit(x_train, y_train)
result = model.predict(x_test)
tmp = [int(result[i] == y_test[i]) for i in range(len(result))]
acc = sum(tmp) / len(tmp)
print("Accuracy: %f"%acc)
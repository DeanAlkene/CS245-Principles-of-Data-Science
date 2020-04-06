import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

X_file_name = "Animals_with_Attributes2/Features/ResNet101/AwA2-features.txt"
y_file_name = "Animals_with_Attributes2/Features/ResNet101/AwA2-labels.txt"
col_name = ['feature' + str(i) for i in range(2048)]
# X_data = pd.read_csv(X_file_name, sep=' ', nrows=2000, names=col_name)
# y_data = pd.read_csv(y_file_name, nrows=2000, names=['label'])
X_data = pd.read_csv(X_file_name, sep=' ', names=col_name)
y_data = pd.read_csv(y_file_name, names=['label'])

def fineTuning(X, y, k=5):
    Cs = [0.1 * i for i in range(1, 21)]
    cv_scores = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    for c in Cs:
        print("\nCross-Validation: C=%0.2f\n"%(c))
        model = SVC(C=c, kernel='rbf', gamma='auto', verbose=False)
        score = cross_val_score(model, X_train, y_train, cv=k, scoring='accuracy')
        cv_scores.append(score.mean())
    bestC = Cs[cv_scores.index(max(cv_scores))]
    plt.plot(Cs, cv_scores, 'bo-', linewidth=2)
    plt.title('Parameter Tuning')
    plt.xlabel('C')
    plt.xticks([0.1 * i for i in range(0, 22)])
    plt.ylabel('Accuracy')
    plt.savefig('TuningParam.jpg')

    bestModel = SVC(C=bestC, kernel='rbf', gamma='auto', verbose=False)
    bestModel.fit(X_train, y_train)
    bestAcc = bestModel.score(X_test, y_test)
    with open('res.txt', 'w') as f:
        f.write("The best C = %0.2f\n"%(bestC))
        f.write("Acc = %f"%(bestAcc))

def main():
    fineTuning(X_data, y_data)

if __name__ == '__main__':
    main()

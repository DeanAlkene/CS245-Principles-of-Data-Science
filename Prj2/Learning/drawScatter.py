import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

def draw(suffix):
    X_test = pd.DataFrame(np.load('X_test_' + suffix + '.npy'), columns=['x', 'y'])
    y_test = pd.DataFrame(np.load('y_test.npy'), columns=['label'])

    transformer = TSNE(n_jobs=8)
    X_test_2d = transformer.fit_transform(X_test)

    test_categories = np.unique(y_test)
    test_colors = [plt.cm.tab10(i/float(len(test_categories)-1)) for i in range(len(test_categories))]
    test_2d = pd.concat([X_test_2d, y_test], axis=1)

    plt.figure()
    for i, label in enumerate(test_categories):
        plt.scatter(test_2d.loc[test_2d.label==label].x, test_2d.loc[test_2d.label==label].y, s=2, cmap=test_colors[i], alpha=0.5)
    plt.title('X')
    plt.savefig('X_test_scatter_' + suffix)

def main():
    draw('LDA')
    draw('LMNN_4')
    draw('PCA_MMC')
    draw('LDA_MMC')

if __name__ == '__main__':
    main()

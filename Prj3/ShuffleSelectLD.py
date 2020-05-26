import numpy as np

f_class_dict = np.load('f_class_dict.npy').item()

def shuffleSelectorClass(className, tot, prop):
    pool = np.load(className + '_10001.npy')
    for i in (10002, tot + 1):
        tmp = np.load(className + '_' + str(i) + '.npy') # 2d np array
        np.concatenate((pool, tmp))
    rand_array = np.arange(pool.shape[0])
    np.random.shuffle(rand_array)
    return pool[rand_array[0: len(pool) / prop]] # 2d np array

def shuffleSelector(prop):
    ld = [] # list of selected features of each class in 2d np array
    for className, totalNum in f_class_dict.items():
        ld.append(shuffleSelectorClass(className, totalNum))
    return np.vstack(ld)

def main():
    lds = shuffleSelector(10)
    np.save('LD_for_clustering', lds)
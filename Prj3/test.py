import numpy as np
import threading

f_class_dict = np.load('f_class_dict.npy', allow_pickle=True).item()
dict_list = np.load('f_class_dict_mul.npy', allow_pickle=True)

class BOWThread(threading.Thread):
    def __init__(self, class_dict):  # model
        threading.Thread.__init__(self)
        self.class_dict = class_dict
    
    def run(self):
        feature = []
        for className, totalNum in self.class_dict.items():
            print("SS at %s" % (className))
            for idx in range(10001, totalNum + 1):
                feature.append(className + '_' + str(idx))
        self.val = np.vstack(feature)
    
    def get(self):
        return self.val

def divideDict():
    dict_list = [{} for i in range(8)]
    i = 0
    for className, totalNum in f_class_dict.items():
        dict_list[i // 7][className] = totalNum
        i += 1
    np.save('f_class_dict_mul', dict_list)

def main():
    feature = [None for i in range(8)]
    threadPool = [BOWThread(dict_list[i]) for i in range(8)]
    for i in range(8):
        threadPool[i].start()
    for i in range(8):
        threadPool[i].join()
    for i in range(8):
        feature[i] = threadPool[i].get()
    n = np.vstack(feature)

main()

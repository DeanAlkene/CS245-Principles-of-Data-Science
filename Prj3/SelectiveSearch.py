import numpy as np
import pickle
import time
from skimage import io, transform
import selective_search
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

IMG_PATH = 'AwA2-data/JPEGImages/'
PRP_PATH = 'AwA2-data/Proposals/'

def gray2rgb():
    src = cv2.imread("collie_10718.jpg", cv2.IMREAD_COLOR)
    cv2.imwrite("collie_10718_rgb.jpg", src)

def SelectiveSearchImg(className, imgName):
    image = io.imread(IMG_PATH + className + '/' + imgName + '.jpg')

    image = transform.resize(image, (224, 224))
    boxes = selective_search.selective_search(image, mode='single')
    boxes_filter = selective_search.box_filter(boxes, min_size=30, topN=20)
    image = np.asarray(image)
    proposals = []
    for box in boxes_filter:
        w, h = box[2] - box[0], box[3] - box[1]
        if w < 150 and h < 150:
            proposals.append(image[box[0] : box[2], box[1] : box[3], :])
    return proposals
    # np.savez_compressed(PRP_PATH + imgName, **{str(i) : box for i, box in enumerate(proposals)}) # for save

def main():
    # f_class_dict = {}
    # with open('AwA2-data/AwA2-filenames.txt') as f_name_list:
    #     line = f_name_list.readline()
    #     curClass = 'antelope'
    #     curNum = '10001'
    #     while line:
    #         lastClass = curClass
    #         lastNum = curNum
    #         tmp = line.split('_')
    #         curClass = tmp[0]
    #         curNum = tmp[1].split('.')[0]
    #         if curClass != lastClass:
    #             f_class_dict[lastClass] = eval(lastNum)
    #         line = f_name_list.readline()
    # f_class_dict['zebra'] = 11170
    
    # np.save('f_class_dict.npy', f_class_dict)
    f_class_dict = np.load('f_class_dict.npy', allow_pickle=True).item() #for load dict
    for className, totalNum in f_class_dict.items():
        print("SS at %s" % (className))
        for idx in range(10001, totalNum + 1):
            SelectiveSearchImg(className, className + '_' + str(idx))
            
if __name__ == '__main__':
    main()



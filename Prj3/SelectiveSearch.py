import numpy as np
import time
from skimage import io, transform
import selective_search
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

IMG_PATH = 'AwA2-data/JPEGImages/'
PRP_PATH = 'AwA2-data/Proposals/'

def SelectiveSearchImg(className, imgName):
    image = io.imread(IMG_PATH + className + '/' + imgName + '.jpg')

    image = transform.resize(image, (224, 224))
    boxes = selective_search.selective_search(image, mode='single')
    boxes_filter = selective_search.box_filter(boxes, min_size=30, topN=20)
    image = np.asarray(image)
    proposals = []
    for i, box in enumerate(boxes_filter):
        w, h = box[2] - box[0], box[3] - box[1]
        if w < 150 and h < 150:
            proposals.append(image[box[0] : box[2], box[1] : box[3], :])
    proposals = np.asarray(proposals)
    np.save(PRP_PATH + imgName, proposals)

def main():
    f_class_dict = {}
    with open('AwA2-data/AwA2-filenames.txt') as f_name_list:
        line = f_name_list.readline()
        curClass = 'antelope'
        curNum = '10001'
        while line:
            lastClass = curClass
            lastNum = curNum
            tmp = line.split('_')
            curClass = tmp[0]
            curNum = tmp[1].split('.')[0]
            if curClass != lastClass:
                f_class_dict[lastClass] = eval(lastNum)
            line = f_name_list.readline()
    f_class_dict['zebra'] = 11170
    
    for className, totalNum in f_class_dict.items():
        print("SS at %s" % (className))
        for idx in range(10001, totalNum + 1):
            SelectiveSearchImg(className, className + '_' + str(idx))
            
if __name__ == '__main__':
    main()

import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from torch import nn, optim
from SelectiveSearch import SelectiveSearchImg

IMG_PATH = '../AwA2-data/JPEGImages/'
LD_PATH = '../AwA2-data/DL_LD/'

data_tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
model = models.resnet152(pretrained=False)
model.load_state_dict(torch.load('./model_res152.pkl'))
exact_list = ['avgpool']
exactor = FeatureExtractor(model, exact_list)

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs

def extract(className, imgName):
    props = SelectiveSearchImg(className, imgName)
    feature_list = []
    for img in props:
        with torch.no_grad():
            img = Variable(img)
        feature = exactor(data_tf(img))[0]
        feature = feature.resize(feature.shape[0], feature.shape[1])
        feature_list.append(feature.detach().cpu().numpy())
        features = np.row_stack(feature_list)
    return feature_list
    
def main():
    
    f_class_dict = np.load('../f_class_dict.npy', allow_pickle=True).item()  #for load dict
    des = extract('antelope', 'antelope_10001')
    print(des.shape)
    # for className, totalNum in f_class_dict.items():
    #     print("SS at %s" % (className))
    #     for idx in range(10001, totalNum + 1):
    #         des = extract(className, className + '_' + str(idx))
    #         np.save(LD_PATH + className + '/' + className + '_' + str(idx), des)

if __name__ == '__main__':
    main()
import torchvision
from torch import nn
from torch.nn import init
from models.utils import pooling
import torch   
import copy     
import torch.nn.functional as F
import random
from models.classifier import NormalizedClassifier
class Pooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=2048, out_channels=4, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.softmax(x)
        return x
    
class TTK(nn.Module):
    def __init__(self, H, W, C):
        self.linear = nn.Linear(C, 16)
        self.linear2 = nn.Linear(H*W, 1024)
        init.normal_(self.linear.weight.data, std=0.001)
        init.constant_(self.linear.bias.data, 0.0)
        init.normal_(self.linear2.weight.data, std=0.001)
        init.constant_(self.linear2.bias.data, 0.0)

class BatchCrop(nn.Module):
    def __init__(self, ratio):
        super(BatchCrop, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rw = int(self.ratio * w)
            start = random.randint(0, h-1)
            if start + rw > h:
                select = list(range(0, start+rw-h)) + list(range(start, h))
            else:
                select = list(range(start, start+rw))
            mask = x.new_zeros(x.size())
            mask[:, :, select, :] = 1
            x = x * mask
        return x

class ResNet50(nn.Module):
    def __init__(self, config, num_clothes,**kwargs):
        super().__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet50.layer4[0].conv2.stride=(1, 1)
            resnet50.layer4[0].downsample[0].stride=(1, 1)
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        self.layer4c = copy.deepcopy(self.layer4)
        if config.MODEL.POOLING.NAME == 'avg':
            self.globalpooling = nn.AdaptiveAvgPool2d(1)
        elif config.MODEL.POOLING.NAME == 'max':
            self.globalpooling = nn.AdaptiveMaxPool2d(1)
        elif config.MODEL.POOLING.NAME == 'gem':
            self.globalpooling = pooling.GeMPooling(p=config.MODEL.POOLING.P)
        elif config.MODEL.POOLING.NAME == 'maxavg':
            self.globalpooling = pooling.MaxAvgPooling()
        else:
            raise KeyError("Invalid pooling: '{}'".format(config.MODEL.POOLING.NAME))
        self.bn = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)  
        self.bn2 = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn2.weight.data, 1.0, 0.02)
        init.constant_(self.bn2.bias.data, 0.0)  
        self.CamClassifier = nn.Linear(config.MODEL.FEATURE_DIM, 13)
        init.normal_(self.CamClassifier.weight.data, std=0.001)
        init.constant_(self.CamClassifier.bias.data, 0.0) 
        self.CloClassifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_clothes) 
        self.IntClassifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=config.MODEL.NUM_INT_CLASSES) 
    def forward(self, x,mode='train'):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  
        base_f = self.layer4(x)
        if mode!='test':
            base_f1,_ = x.split(int(x.size(0)/2), dim=0)
            base_f2 = self.layer4c(base_f1)
            base_f=self.globalpooling(base_f)
            base_f = base_f.view(base_f.size(0), -1)
            ID_Features = self.bn(base_f)
            INT_Features=self.globalpooling(base_f2)
            INT_Features = INT_Features.view(INT_Features.size(0), -1)
            INT_Features = self.bn2(INT_Features)
            camera_score = self.CamClassifier(INT_Features)   
            clothes_score = self.CloClassifier(INT_Features) 
            int_score = self.IntClassifier(INT_Features)
            # f2 = self.gan(INT_Features.detach())
            return base_f, ID_Features, INT_Features, camera_score, clothes_score, int_score
        ID_Features=self.globalpooling(base_f)
        ID_Features = ID_Features.view(ID_Features.size(0), -1)    
        ID_Features = self.bn(ID_Features)
        base_f1 = self.layer4c(x)
        INT_Features=self.globalpooling(base_f1)
        INT_Features = INT_Features.view(INT_Features.size(0), -1)
        INT_Features = self.bn2(INT_Features)
        score = self.CamClassifier(INT_Features)
        _, preds = torch.max(score.data, 1)
        
        return preds, ID_Features


    
    def get_atten_map(self, feature_maps, gt_labels):
        b, c, h, w = feature_maps.shape        #1,2048,7,7
        output_cam = torch.zeros((b,h,w))
        for i in range(b):
            cam = self.fc_weights[gt_labels[i]]@feature_maps[i].reshape((c, h*w))  
                #(1, 2048) * (2048, 7*7) -> (1, 7*7) 
            cam = cam.reshape(h, w)
            cam_img = (cam - cam.min()) / (cam.max() - cam.min())  #Normalize
            binary_mask = cam_img.ge(0.9)
            binary_mask = ~binary_mask
            binary_cam = binary_mask.float()
            output_cam[i]=binary_cam.cuda()
        atten_map=output_cam.unsqueeze(1).cuda()
        return atten_map
   

class DISIFLF(nn.Module):
    def __init__(self, config,**kwargs):
        super().__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet50.layer4[0].conv2.stride=(1, 1)
            resnet50.layer4[0].downsample[0].stride=(1, 1)
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        if config.MODEL.POOLING.NAME == 'avg':
            self.globalpooling = nn.AdaptiveAvgPool2d(1)
        elif config.MODEL.POOLING.NAME == 'max':
            self.globalpooling = nn.AdaptiveMaxPool2d(1)
        elif config.MODEL.POOLING.NAME == 'gem':
            self.globalpooling = pooling.GeMPooling(p=config.MODEL.POOLING.P)
        elif config.MODEL.POOLING.NAME == 'maxavg':
            self.globalpooling = pooling.MaxAvgPooling()
        else:
            raise KeyError("Invalid pooling: '{}'".format(config.MODEL.POOLING.NAME))
        self.bn = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)  
        self.bn2 = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn2.weight.data, 1.0, 0.02)
        init.constant_(self.bn2.bias.data, 0.0)   
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  
        base_f = self.layer4(x)
        base_f=self.globalpooling(base_f)
        base_f = base_f.view(base_f.size(0), -1)
        ID_Features = self.bn(base_f)
        return base_f, ID_Features



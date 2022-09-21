import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.models._utils as _utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from itertools import product as product
from math import ceil
import os
import torch.utils.data as data
from PIL import Image, ImageFilter 
import os, cv2
import numpy as np
import random
from scipy.stats import norm
from scipy.spatial.transform import Rotation
from math import floor
import math

def get_meanface(meanface_file, num_nb):
    with open(meanface_file) as f:
        meanface = f.readlines()[0]
        
    meanface = meanface.strip().split()
    meanface = [float(x) for x in meanface]
    meanface = np.array(meanface).reshape(-1, 2)
    # each landmark predicts num_nb neighbors
    meanface_indices = []
    for i in range(meanface.shape[0]):
        pt = meanface[i,:]
        dists = np.sum(np.power(pt-meanface, 2), axis=1)
        indices = np.argsort(dists)
        meanface_indices.append(indices[1:1+num_nb])
    
    # each landmark predicted by X neighbors, X varies
    meanface_indices_reversed = {}
    for i in range(meanface.shape[0]):
        meanface_indices_reversed[i] = [[],[]]
    for i in range(meanface.shape[0]):
        for j in range(num_nb):
            meanface_indices_reversed[meanface_indices[i][j]][0].append(i)
            meanface_indices_reversed[meanface_indices[i][j]][1].append(j)
    
    max_len = 0
    for i in range(meanface.shape[0]):
        tmp_len = len(meanface_indices_reversed[i][0])
        if tmp_len > max_len:
            max_len = tmp_len
    
    # tricks, make them have equal length for efficient computation
    for i in range(meanface.shape[0]):
        tmp_len = len(meanface_indices_reversed[i][0])
        meanface_indices_reversed[i][0] += meanface_indices_reversed[i][0]*10
        meanface_indices_reversed[i][1] += meanface_indices_reversed[i][1]*10
        meanface_indices_reversed[i][0] = meanface_indices_reversed[i][0][:max_len]
        meanface_indices_reversed[i][1] = meanface_indices_reversed[i][1][:max_len]

    # make the indices 1-dim
    reverse_index1 = []
    reverse_index2 = []
    for i in range(meanface.shape[0]):
        reverse_index1 += meanface_indices_reversed[i][0]
        reverse_index2 += meanface_indices_reversed[i][1]
    return meanface_indices, reverse_index1, reverse_index2, max_len

class PIPNet(nn.Module):
    def __init__(self, mbnet, num_nb, num_lms=98, input_size=256, net_stride=32,reverse_ind1=None,reverse_ind2=None,max_len=None):
        super(PIPNet, self).__init__()
        self.num_nb = num_nb
        self.num_lms = num_lms
        self.input_size = input_size
        self.net_stride = net_stride
        self.reverse_ind1=reverse_ind1
        self.reverse_ind2=reverse_ind2
        self.max_len = max_len
        self.features = mbnet.features[:11]
        self.nextfeatures=nn.Sequential(nn.Conv2d(64,64,kernel_size=(9,9),stride=(1,1),bias=False),
                                        nn.BatchNorm2d(64,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
                                        nn.ReLU6(inplace=True))
        self.sigmoid = nn.Sigmoid()

        self.cls_layer = nn.Conv2d(64, num_lms, kernel_size=1, stride=1, padding=0)
        self.x_layer = nn.Conv2d(64, num_lms, kernel_size=1, stride=1, padding=0)
        self.y_layer = nn.Conv2d(64, num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_x_layer = nn.Conv2d(64, num_nb*num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_y_layer = nn.Conv2d(64, num_nb*num_lms, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x = self.features(x)
        x= self.nextfeatures(x)
        x1 = self.cls_layer(x)
        x2 = self.x_layer(x)
        x3 = self.y_layer(x)
        x4 = self.nb_x_layer(x)
        x5 = self.nb_y_layer(x)
        outputs_cls, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y=x1,x2,x3,x4,x5
        tmp_batch, tmp_channel, tmp_height, tmp_width = x1.size()

        outputs_cls = outputs_cls.view(tmp_batch*tmp_channel, -1)
        max_ids = torch.argmax(outputs_cls, 1)
        max_cls = torch.max(outputs_cls, 1)[0]
        max_ids = max_ids.view(-1, 1)
        max_ids_nb = max_ids.repeat(1, 10).view(-1, 1)

        outputs_x = outputs_x.view(tmp_batch*tmp_channel, -1)
        outputs_x_select = torch.gather(outputs_x, 1, max_ids)
        outputs_x_select = outputs_x_select.squeeze(1)
        outputs_y = outputs_y.view(tmp_batch*tmp_channel, -1)
        outputs_y_select = torch.gather(outputs_y, 1, max_ids)
        outputs_y_select = outputs_y_select.squeeze(1)

        outputs_nb_x = outputs_nb_x.view(tmp_batch*10*tmp_channel, -1)
        outputs_nb_x_select = torch.gather(outputs_nb_x, 1, max_ids_nb)
        outputs_nb_x_select = outputs_nb_x_select.squeeze(1).view(-1, 10)
        outputs_nb_y = outputs_nb_y.view(tmp_batch*10*tmp_channel, -1)
        outputs_nb_y_select = torch.gather(outputs_nb_y, 1, max_ids_nb)
        outputs_nb_y_select = outputs_nb_y_select.squeeze(1).view(-1, 10)

        tmp_x = (max_ids%tmp_width).view(-1,1).float()+outputs_x_select.view(-1,1)
        tmp_y = (max_ids//tmp_width).view(-1,1).float()+outputs_y_select.view(-1,1)
        tmp_x /= 1.0 * 256 / 32
        tmp_y /= 1.0 * 256 / 32

        tmp_nb_x = (max_ids%tmp_width).view(-1,1).float()+outputs_nb_x_select
        tmp_nb_y = (max_ids//tmp_width).view(-1,1).float()+outputs_nb_y_select
        tmp_nb_x = tmp_nb_x.view(-1, 10)
        tmp_nb_y = tmp_nb_y.view(-1, 10)
        tmp_nb_x /= 1.0 * 256 / 32
        tmp_nb_y /= 1.0 * 256 / 32
        lms_pred = torch.cat((tmp_x, tmp_y), dim=1).flatten()
        tmp_nb_x = tmp_nb_x[self.reverse_ind1, self.reverse_ind2].view(98, self.max_len)
        tmp_nb_y = tmp_nb_y[self.reverse_ind1, self.reverse_ind2].view(98, self.max_len)
        tmp_x = torch.mean(torch.cat((tmp_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
        tmp_y = torch.mean(torch.cat((tmp_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
        lms_pred_merge*=256
        return lms_pred_merge
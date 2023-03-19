import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

import cv2
import os
import numpy as np

from time import time

from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool

#path_t1_8_7 = './carpalTunnel/8/T1/6.jpg'
#path_t2_8_7 = './carpalTunnel/8/T2/6.jpg'
#path_ct_8_7 = './carpalTunnel/8/CT/6.jpg'
#path_ft_8_7 = './carpalTunnel/8/FT/6.jpg'
#path_mn_8_7 = './carpalTunnel/8/MN/6.jpg'

t1 = cv2.imread('./carpalTunnel/8/T1/8.jpg',2)
t2 = cv2.imread('./carpalTunnel/8/T2/8.jpg',2)
ct = cv2.imread('./carpalTunnel/8/CT/8.jpg',2)
ft = cv2.imread('./carpalTunnel/8/FT/8.jpg',2)
mn = cv2.imread('./carpalTunnel/8/MN/8.jpg',2)

#print(ct.shape)
inputdata = []

inputdata.append(t1)
inputdata.append(t2)
inputdata = np.array(inputdata)
inputdata = np.expand_dims(inputdata,0)
#inputdata = inputdata[:,:,128:384,128:384]
inputdata = inputdata / 255

label = []

label.append(ct)
label.append(ft)
label.append(mn)

label = np.array(label)
label = np.transpose(label,(1,2,0))
print(np.array(label).shape)
with torch.no_grad():
    device = torch.device("cuda")
    model = DinkNet34()
    model.cuda()
    model.load_state_dict(torch.load('model_dlinknet50.pth'))
    model.eval()
    inputdata = torch.tensor(inputdata)
    inputdata = inputdata.type(torch.FloatTensor).cuda()
    output = model(inputdata)
    output = output.cpu()
    output = np.array(output)
    output =np.squeeze(output)
    output = np.transpose(output,(1,2,0)).round()
    #print(output.shape)
    cv2.imshow('moduloutput',output)
    cv2.imshow('label',label)
    cv2.waitKey(0)

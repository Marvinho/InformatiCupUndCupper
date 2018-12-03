# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 18:44:21 2018

@author: MRVN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader#, random_split
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import requests


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3)
        self.conv4 = nn.Conv2d(64, 128, kernel_size = 3)
        self.conv_dropout = nn.Dropout2d()
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(21632, 512)
        self.fc2 = nn.Linear(512, 1)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.softsign(x)
        x = self.conv2(x)
        x = F.softsign(x)
        x = F.max_pool2d(x, 2)
#        print(x.shape)
        x = self.conv3(x)
        x = F.softsign(x)
        x = self.conv4(x)
        x = F.softsign(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        #exit()
        #x = self.dropout(x)
        #x = F.softsign(x)
        x = self.fc1(x)
        x = F.softsign(x)
        x = self.fc2(x)
        return F.sigmoid(x)


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def loss(self, pics, x):
        url = "https://phinau.de/trasi"
        key = {"key" : "raekieh3ZofooPhaequoh9oonge8eiya"}
        dirs = os.listdir("./AdvTraining/Results" )
        confidences = []
        i = 0
        criterion = nn.L1Loss()
        image_size = (64,64)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        device = torch.device("cpu")
        #print(x[0])


        while i < 5: 
            y = pics[i]     #remove batch dimension # B X C H X W ==> C X H X W
           # y = y.mul(torch.FloatTensor(std).to(device).view(3,1,1))
            #y = y.add(torch.FloatTensor(mean).to(device).view(3,1,1))
            y = y.detach().to("cpu").numpy()#reverse of normalization op- "unnormalize"
            y = np.transpose( y , (1,2,0))   # C X H X W  ==>   H X W X C
            #y = np.clip(y, 0, 1)
            y = Image.fromarray(np.uint8(y*255), "RGB")
            y.save("./AdvTraining/Results/adv_img_{}.png".format(i))
            i = i + 1
        
        for file in dirs:
            files = {"image": open("./AdvTraining/Results/{}".format(file), "rb")}
            r = requests.post(url, data = key, files = files)
            #print(r.json())
            answer = r.json()
            confidences.append(answer[0]['confidence'])
            #print(r.json())
        
        print(confidences)
        results = torch.FloatTensor(confidences)
        results = results.unsqueeze(1)
        #print(results)
        #print(x)
        #print(criterion(x, results))
        #exit()
        time.sleep(6)
        return criterion(x, results)

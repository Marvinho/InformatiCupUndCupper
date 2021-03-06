# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 18:44:21 2018

@author: MRVN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
#    pretrained_model = "saved_model_state_CNN_final.pth"
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 5)
        self.conv3 = nn.Conv2d(32, 32, kernel_size = 3)
        self.conv_dropout = nn.Dropout2d()
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(32*5*5, 256)
        self.fc2 = nn.Linear(256, 43)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
#        print(x.shape)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
#        print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x)


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    
    def loadModel(self, pretrained_model):
        
        print("loading the model...")  
        self.load_state_dict(torch.load(pretrained_model, map_location=lambda storage, loc: storage))
        self.eval()
        print("loaded the model.")
        
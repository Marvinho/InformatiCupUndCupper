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

class InformatiCupLoss(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        url = "https://phinau.de/trasi"
        key = {"key" : "raekieh3ZofooPhaequoh9oonge8eiya"}
        dirs = os.listdir("./AdvTraining/Results" )
        confidences = []
        i = 0
        criterion = nn.CrossEntropyLoss()
        #print(x[0])


        while i < 5: 
            y = x[i]     #remove batch dimension # B X C H X W ==> C X H X W
            y = y.mul(torch.FloatTensor(std).to(device).view(3,1,1))
            y = y.add(torch.FloatTensor(mean).to(device).view(3,1,1))
            y = y.detach().to("cpu").numpy()#reverse of normalization op- "unnormalize"
            y = np.transpose( y , (1,2,0))   # C X H X W  ==>   H X W X C
            #y = np.clip(y, 0, 1)
            y = Image.fromarray(np.uint8(y*255), "RGB")
            y.save("./AdvTraining/Results/adv_img_{}.png".format(i))
            i = i + 1
    
        for file in dirs:
            files = {"image": open("./AdvTraining/Results//{}".format(file), "rb")}
            r = requests.post(url, data = key, files = files)
            #print(r.json())
            answer = r.json()
            confidences.append(answer[0]['confidence'])
            #print(r.json())
        results = torch.FloatTensor(confidences)
        time.sleep(5)
        return criterion(input, results)

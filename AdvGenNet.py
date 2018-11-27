# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 11:31:28 2018

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
import InformatiCupLoss
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import requests
import DistilledCNN



class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(64*64, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 64*64)
    
    
    def forward(self, x):
        x = x.view(5,3,64*64)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
#        print(x.shape)
        x = self.fc3(x)
        x = F.leaky_relu(x)
        x = self.fc4(x)
#        print(x.shape)
        x = F.tanh(x)
        x = x.view(5,3, 64, 64)
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features




#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Ye-Aulde-PC-Fix
device = torch.device("cpu")
print(device)

batch_size = 5
image_size = (64,64)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

traindatapath = "./AdvTraining/"
testdatapath = "./AdvTraining/Images"

data_transforms = {
        "train": transforms.Compose([transforms.Resize(size = image_size), 
                            transforms.ToTensor(), 
                            transforms.Normalize(mean, std)]),
        "test": transforms.Compose([transforms.Resize(size = image_size), 
                            transforms.ToTensor(), 
                            transforms.Normalize(mean, std)])
    }

#print(data_transforms["train"])

train_data = ImageFolder(root = traindatapath, transform = data_transforms["train"])
test_data = train_data

train_size = int(len(train_data))
#test_size = len(full_data) - train_size
#train_data, test_data = random_split(full_data, [train_size, test_size])

#print(len(train_data))

trainloader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)

testloader = DataLoader(dataset = test_data, batch_size = batch_size)

#class_names = full_data.classes

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(trainloader))

# Make a grid from batch
out = utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])


model1 = Net()
model1 = model1.to(device)
model2 = DistilledCNN.Net()
model2 = model2.to(device)
criterion1 = InformatiCupLoss.apply
criterion2 = nn.L1Loss()

optimizer1 = optim.SGD(model1.parameters(), lr=0.01, momentum=0.9)
optimizer2 = optim.SGD(model2.parameters(), lr=0.01, momentum=0.9)
num_epochs = 61
target = torch.FloatTensor([[1],[1],[1],[1],[1]])
target = target.squeeze()



def training(model = model, optimizer = optimizer, num_epochs = num_epochs):
    
    print("start training...")
    since = time.time()
    model.train()
    
    for epoch in range(num_epochs):
        timestart_epoch = time.time()
        print()
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        running_loss = 0.0
        running_corrects = 0.0
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = Variable(inputs, requires_grad= True)
            #print("put inputs and labels on gpu...")
      
            # zero the parameter gradients
            optimizer1.zero_grad()
            optimizer2.zero_grad()
    
            # forward + backward + optimize
            output1 = model1(inputs)
            output2 = model2(output1)

            
            loss2 = criterion1(output2)
            loss1 = criterion2(output2) 
            loss2.backward()
            optimizer2.step()
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()


            # print statistics
            running_loss1 += output2.item() * inputs.size(0)
            running_loss2 += loss.item() * inputs.size(0)
            #running_corrects += torch.sum(predicted.data == labels.data)

        epoch_loss1 = running_loss1 / train_size
        epoch_loss2 = running_loss2 / train_size
        #epoch_accuracy = (running_corrects.item() / train_size) * 100
        print("Loss1: {:.4f} Loss2: {:.4f}".format(epoch_loss1, epoch_loss2))
        
        time_elapsed = time.time() - timestart_epoch
        print("finished epoch in {:.0f}m {:.0f}s.".format(time_elapsed // 60, time_elapsed % 60))
        
        
    
    time_elapsed = time.time() - since
    print("finished training in {:.0f}m {:.0f}s.".format(time_elapsed // 60, time_elapsed % 60))
    print("saving the model...")
    torch.save(model.state_dict(), "saved_model_state_AdvGenNet_{}.pth".format(epoch))
    print("saved the model.")



def testing(epoch, model = model):
    print("start testing...")
    model.eval()
    with torch.no_grad():
        
        running_loss = 0.0
        running_corrects = 0.0
        
        for data in testloader:
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

    
            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            loss = criterion(outputs, labels)
    
            # print statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(predicted.data == labels.data)
            

        epoch_loss = running_loss / test_size
        epoch_accuracy = (running_corrects.item() / test_size) * 100
        print("Loss: {:.4f} Accuracy: {:.4f}%".format(epoch_loss, epoch_accuracy))
    
#    print('Accuracy of the network on the test images: {}'.format(epoch_accuracy))
    print("finished testing.")
    print("saving the model...")
    torch.save(model.state_dict(), "saved_model_state_AdvGenNet_{}.pth".format(epoch))
    print("saved the model.")


if __name__ == "__main__":
    training()
    #testing(epoch = num_epochs)
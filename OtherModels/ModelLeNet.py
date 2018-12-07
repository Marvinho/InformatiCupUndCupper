# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 11:31:28 2018

@author: MRVN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, utils, models
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 720)
        self.fc2 = nn.Linear(720, 340)
        self.fc3 = nn.Linear(340, 43)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
#        print(x.shape)
        x = x.view(-1, 16 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 64
image_size = (64,64)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

traindatapath = "../GTSRB/Final_Training/Images"
testdatapath = "../GTSRB/Final_Test/Images"

data_transforms = {
        "train": transforms.Compose([transforms.Resize(size = image_size), 
                            transforms.ToTensor(), 
                            transforms.Normalize(mean, std)]),
        "test": transforms.Compose([transforms.Resize(size = image_size), 
                            transforms.ToTensor(), 
                            transforms.Normalize(mean, std)])
    }

#print(data_transforms["train"])

full_data = ImageFolder(root = traindatapath, transform = data_transforms["train"])


train_size = int(0.8 * len(full_data))
test_size = len(full_data) - train_size
train_data, test_data = random_split(full_data, [train_size, test_size])

#print(len(train_data))

trainloader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)

testloader = DataLoader(dataset = test_data, batch_size = batch_size)

class_names = full_data.classes

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


## Get a batch of training data
#inputs, classes = next(iter(trainloader))
#
## Make a grid from batch
#out = utils.make_grid(inputs)
#
#imshow(out, title=[class_names[x] for x in classes])


model = Net()
#print(model)
#print(model.classifier[6])
#print(model.classifier[6].in_features)
#num_ftrs = model.classifier[6].in_features
#model.classifier[6].out_features = 43
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)
num_epochs = 21

best_accuracy = 0.00

def training(model = model, criterion = criterion, optimizer = optimizer, num_epochs = num_epochs):
    

    
#    best_model_weights = copy.deepcopy(model.state_dict())
    since = time.time()
    model.train()
    
    for epoch in range(num_epochs):
        timestart_epoch = time.time()
        print("---------- Training Phase ----------")
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        running_loss = 0.0
        running_corrects = 0.0
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            #print("put inputs and labels on gpu...")
      
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(predicted.data == labels.data)

        epoch_loss = running_loss / train_size
        epoch_accuracy = (running_corrects.item() / train_size) * 100
        print("training Loss: {:.4f} Accuracy: {:.4f}%".format(epoch_loss, epoch_accuracy))
        
        time_elapsed = time.time() - timestart_epoch
        print("finished epoch in {:.0f}m {:.0f}s.".format(time_elapsed // 60, time_elapsed % 60))
        
        if(epoch > 2):
            print()
            testing(epoch)
            print()
    
    time_elapsed = time.time() - since
    print("finished training in {:.0f}m {:.0f}s.".format(time_elapsed // 60, time_elapsed % 60))



def testing(epoch, model = model, criterion = criterion):
    print("---------- Testing Phase ----------")
    global best_accuracy
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
        print("validation Loss: {:.4f}, Accuracy: {:.4f}%".format(epoch_loss, epoch_accuracy))
    
#    print('Accuracy of the network on the test images: {}'.format(epoch_accuracy))
    print("finished testing.")
    if(epoch_accuracy > best_accuracy):
        print("old best accuracy: {:.4f}, new best accuracy: {:.4f}".format(best_accuracy, epoch_accuracy))
        best_accuracy = epoch_accuracy        
        print("saving the model...")
        torch.save(model.state_dict(), "saved_model_state_lenet_epoch_{}_accuracy_{:.2f}.pth"
                   .format(epoch, best_accuracy))
        print("saved the model.")


if __name__ == "__main__":
    training()
#    testing(epoch = num_epochs)
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 11:31:28 2018

@author: MRVN
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import time
import modelcnn







device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 32
image_size = (64,64)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

traindatapath = "../../GTSRB/Final_Training/Images"
testdatapath = "../../GTSRB/Final_Test/Images"

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


# Get a batch of training data
inputs, classes = next(iter(trainloader))

# Make a grid from batch
out = utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])


model = modelcnn.Net()
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epochs = 61

def training(model = model, criterion = criterion, optimizer = optimizer, num_epochs = num_epochs):
    
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
        print("Loss: {:.4f} Accuracy: {:.4f}%".format(epoch_loss, epoch_accuracy))
        
        time_elapsed = time.time() - timestart_epoch
        print("finished epoch in {:.0f}m {:.0f}s.".format(time_elapsed // 60, time_elapsed % 60))
        
        if(epoch%4 == 0 and epoch > 0):
            print()
            testing(epoch)
            print()
    
    time_elapsed = time.time() - since
    print("finished training in {:.0f}m {:.0f}s.".format(time_elapsed // 60, time_elapsed % 60))



def testing(epoch, model = model, criterion = criterion):
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
    torch.save(model.state_dict(), "saved_model_state_CNN_{}.pth".format(epoch))
    print("saved the model.")


if __name__ == "__main__":
    training()
    testing(epoch = num_epochs)
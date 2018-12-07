# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 16:12:37 2018

@author: marvi
"""
import torch
from torchvision import transforms
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import shutil
import modelcnn
import generateimage
import requests

image_size = (64,64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

preprocess = transforms.Compose([transforms.Resize(size = image_size), 
                            transforms.ToTensor()])
    
testdatapath = "./Images/"
try:
    test_data = ImageFolder(root = testdatapath, transform = preprocess)
except:
    print("YOU NEED IMAGES IN ./IMAGES/ORIGINALS")
    print("CREATING RANDOM IMAGE...")
    generateimage.createImage(random = True)
    test_data = ImageFolder(root = testdatapath, transform = preprocess)

testloader = DataLoader(dataset = test_data)


class AdvImage():

    def __init__(self):
        pass


    def transformTensorToImage(x_adv):
        x_adv = x_adv.squeeze(0)     #remove batch dimension # B X C H X W ==> C X H X W
        x_adv = np.transpose(x_adv, (1,2,0))   # C X H X W  ==>   H X W X C
        x_adv = np.clip(x_adv, 0, 1)
        im = Image.fromarray(np.uint8(x_adv*255), "RGB")
        im = im.resize((64,64))
#        plt.imshow(im)
#        plt.show()
        return im

        
    def saveAdvImage(im, epsilon, num_step, alpha, y_target_label):        
        date_string = time.strftime("%Y-%m-%d-%H_%M")
        image_path = "./adversarials/adverimg_{}_eps{}_iter{}_alpha{}_label{}.png".format(date_string, 
                                              epsilon, num_step, alpha, y_target_label)
        print("saving image at {}...".format(image_path))
        im.save(image_path)
        return image_path
        
            
    def createIterativeAdversarial(image, y_target_label, output, 
                                   epsilon=0.25, alpha=0.025, num_steps=31):
        y_target = torch.tensor([y_target_label], requires_grad=False)
        print("targetlabel for the attack: {}".format(y_target.item()))
        y_target = y_target.to(device)
                
        for num_step in range(num_steps):
            zero_gradients(image)
            output = model.forward(image)
            
            loss = torch.nn.CrossEntropyLoss()                     
            loss_cal2 = loss(output, y_target)
            loss_cal2.backward(retain_graph=True)
            
            x_grad = alpha * torch.sign(image.grad.data)
            adv_temp = image.data - x_grad
            total_grad = adv_temp - image
            total_grad = torch.clamp(total_grad, -epsilon, epsilon)
            x_adversarial = image + total_grad
            image.data = x_adversarial
            
        im = AdvImage.transformTensorToImage(image.data)
        image_path = AdvImage.saveAdvImage(im, epsilon, num_step, alpha, y_target_label)
        top_confidence = AdvImage.testAdvImageOnBlackbox(image_path)
        print("Blackbox confidence: {}".format(top_confidence))
        AdvImage.deleteImage(top_confidence, image_path)

    
    def moveUsedImage():
        source = "./Images/originals/"
        destination = "./usedImages/"               
        files = os.listdir(source)        
        for file in files:
            shutil.move(os.path.join(source, file), destination)    
    

    def testAdvImageOnBlackbox(image_path):    
        url = "https://phinau.de/trasi"
        key = {"key" : "raekieh3ZofooPhaequoh9oonge8eiya"}
        
        with open("{}".format(image_path), "rb") as f:
            files = {"image" : f}
            r = requests.post(url, data = key, files = files)
            if(r.status_code != 200):
                print("too many requests or sth else went wrong")
    #        print(r.status_code)
            confidences = r.json()
            top_confidence = confidences[0]["confidence"]
        print("Precision: {}".format(top_confidence))
        return top_confidence

    
    def deleteImage(top_confidence, image_path):
        if(top_confidence < 0.9):
            try:
                print("deleting Image...")
                os.remove(image_path)
            except OSError as e:  ## if failed, report it back to the user ##
                print ("Error: %s - %s." % (e.filename, e.strerror))

    
if __name__ == "__main__":
    
    model = modelcnn.Net()    
    model.loadModel(pretrained_model = "saved_model_state_CNN_final.pth")
    model = model.to(device)

    for y_target_label in range(6):
        for data in testloader:            
            image, labels = data
            image, labels = image.to(device), labels.to(device)
            image.requires_grad = True
            output = model.forward(image)
            
            AdvImage.createIterativeAdversarial(image, y_target_label, output)
    
    AdvImage.moveUsedImage()
    print("finished.")
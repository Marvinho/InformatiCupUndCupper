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
import warnings

class AdvGenerator():
    image_size = (64,64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if (torch.cuda.get_device_capability(0) < (5,0)):
            device = torch.device("cpu")
    print(device)
    labels = np.arange(43)
    preprocess = transforms.Compose([transforms.Resize(size = image_size), 
                                transforms.ToTensor()])   
    testdatapath = "./Images/"
    created_adversarial_list = []   
    
    
    def __init__(self):
        pass
    
    def loadData(self):
        try:
            test_data = ImageFolder(root = self.testdatapath, transform = self.preprocess)
        except:
            print("YOU NEED IMAGES IN ./IMAGES/ORIGINALS")
            print("CREATING RANDOM IMAGE...")
            generateimage.createImage(color = "Random")
            test_data = ImageFolder(root = self.testdatapath, transform = self.preprocess)
        
        testloader = DataLoader(dataset = test_data)
        return testloader

    def transformTensorToImage(self, x_adv):
        x_adv = x_adv.squeeze(0).cpu()     #remove batch dimension # B X C H X W ==> C X H X W
        x_adv = np.transpose(x_adv, (1,2,0))   # C X H X W  ==>   H X W X C
        x_adv = np.clip(x_adv, 0, 1)
        image = Image.fromarray(np.uint8(x_adv*255), "RGB")
        image = image.resize((64,64))
#        plt.imshow(im)
#        plt.show()
        return image

        
    def saveImage(self, image, epsilon, num_step, alpha, target_label):        
        date_string = time.strftime("%Y-%m-%d-%H_%M")
        image_path = "./adversarials/adverimg_{}_eps{}_iter{}_alpha{}_label{}.png".format(date_string, 
                                              epsilon, num_step, alpha, target_label)
        print("saving image at {}...".format(image_path))
        if not os.path.exists("./adversarials/"):
            os.makedirs("./adversarials/")
        image.save(image_path)
        return image_path
        
            
    def createIterativeAdversarial(self, image, target_label, output, 
                                   epsilon, alpha, num_steps, model):
        y_target = torch.tensor([target_label], requires_grad=False)
        print("targetlabel for the attack: {}".format(y_target.item()))
        y_target = y_target.to(self.device)
                
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
            
        im = self.transformTensorToImage(image.data)
        image_path = self.saveImage(im, epsilon, num_step, alpha, target_label)
        top_confidence = self.testOnBlackbox(image_path, target_label)
        print("Blackbox confidence: {}".format(top_confidence))
        self.deleteImage(top_confidence, image_path)

    
    def moveUsedImage(self):
        date_string = time.strftime("%Y-%m-%d-%H_%M_%S")
        source = "./Images/originals/"
        destination = "./usedImages/UsedImage{}.png".format(date_string)
        if not os.path.exists("./usedImages/"):
            os.makedirs("./usedImages/")               
        files = os.listdir(source)        
        for file in files:
            shutil.move(os.path.join(source, file), destination)    
    

    def testOnBlackbox(self, image_path, target_label):    
        url = "https://phinau.de/trasi"
        key = {"key" : "raekieh3ZofooPhaequoh9oonge8eiya"}
        
        with open(image_path, "rb") as f:
            files = {"image" : f}
            r = requests.post(url, data = key, files = files)
            if(r.status_code != 200):
                print("too many requests or sth else went wrong")
    #        print(r.status_code)
            confidences = r.json()
            top_confidence = confidences[0]["confidence"]
            top_class = confidences[0]["class"]
        self.created_adversarial_list.append((target_label, 
                                              top_class, top_confidence))
        return top_confidence

    
    def deleteImage(self, top_confidence, image_path):
        if(top_confidence < 0.9):
            try:
                print("deleting Image...")
                os.remove(image_path)
            except OSError as e:  ## if failed, report it back to the user ##
                print ("Error: %s - %s." % (e.filename, e.strerror))

    
    def generateAdv(self, num_steps, epsilon, alpha, target):
        device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
        if (torch.cuda.get_device_capability(0) < (5,0)):
            device = torch.device("cpu")
        model = modelcnn.Net()   
        model.loadModel(pretrained_model = "saved_model_state_CNN_final.pth")
        adv = AdvGenerator()
        model = model.to(device)
        testloader = adv.loadData()
        if (target>=0 and target<43):
            adv.labels = np.array([target])
        for target_label in adv.labels:
            for data in testloader:            
                image, labels = data
                image, labels = image.to(device), labels.to(device)
                image.requires_grad = True
                output = model.forward(image)

                adv.createIterativeAdversarial(image, target_label.item(), 
                                               output, epsilon, alpha, 
                                               num_steps, model)
        adv.moveUsedImage()        
    
if __name__ == "__main__":
    
    model = modelcnn.Net()    
    model.loadModel(pretrained_model = "saved_model_state_CNN_final.pth")
    adv = AdvGenerator()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model = model.to(device)
        
    testloader = adv.loadData()
    for target_label in adv.labels:
        for data in testloader:            
            image, labels = data
            image, labels = image.to(device), labels.to(device)
            image.requires_grad = True
            output = model.forward(image)
                
            adv.createIterativeAdversarial(image, target_label.item(), output, epsilon, alpha, num_steps, model)
        
    adv.moveUsedImage()
    print("finished.")
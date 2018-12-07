# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 11:20:37 2018

@author: MRVN
"""
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import shutil
import modelcnn
import generateimage




model = modelcnn.Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

image_size = (64,64)
#mean = [0.485, 0.456, 0.406]
#std = [0.229, 0.224, 0.225]

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



def showPlot(probs):
    
    fig = plt.figure(figsize = (15,9))
    fig.suptitle('Probabilities')
    x_axe = x_axe = np.arange(0,43, step = 1)
    plt.bar(x_axe, probs.detach().squeeze().numpy(), width = 0.5, tick_label = x_axe)    
#    print(output_probs)
    

def predictImage(data):
    
    image, labels = data
    image, labels = image.to(device), labels.to(device)
    image.requires_grad = True
    
    output = model.forward(image)
    x_pred = torch.max(output.data, 1)[1][0]   #get an index(class number) of a largest element   
    output_probs = F.softmax(output, dim=1)
#    showPlot(output_probs)
    x_pred_prob =  torch.max(output_probs.data, 1)[0][0]
    
#    print("prediction: {} confidence of: {:.2f}%"
#          .format(x_pred.item(), x_pred_prob.item()*100))
    
    return image, output, x_pred, x_pred_prob

    
def createIterativeAdversarial(image, y_target_label, output, x_pred, x_pred_prob):
    
    image_temp = image.clone()
    y_target = torch.tensor([y_target_label], requires_grad=False)
    print("targetlabel for the attack: {}".format(y_target.item()))
    y_target = y_target.to(device)
             
    epsilons = [0.25]
    num_steps = 26
    alphas = [0.025]
    
    for alpha in alphas:
        for epsilon in epsilons:
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
                
                if(num_step == 25):
                    output_adv = model.forward(Variable(image))
                    x_adv_pred = torch.max(output_adv.data, 1)[1][0]
                    op_adv_probs = F.softmax(output_adv, dim=1)
                    x_adv_pred_prob =  torch.max(op_adv_probs.data, 1)[0][0]
                    visualize(image_temp, image.data, total_grad, epsilon, num_step, 
                              alpha, x_pred, x_adv_pred, 
                              x_pred_prob, x_adv_pred_prob, y_target_label) 
        
    
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.squeeze(0)
    print(inp.size())
    inp = inp.detach().to(device).numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated



def visualize(x, x_adv, x_grad, epsilon, iteration, alpha, clean_pred, adv_pred, clean_prob, adv_prob, y_target_label):
    
    x = x.squeeze(0)     #remove batch dimension # B X C H X W ==> C X H X W
#    x = x.mul(torch.FloatTensor(std).to(device).view(3,1,1))
#    x = x.add(torch.FloatTensor(mean).to(device).view(3,1,1))
    x = x.detach().to("cpu").numpy()#reverse of normalization op- "unnormalize"
    x = np.transpose( x , (1,2,0))   # C X H X W  ==>   H X W X C
    x = np.clip(x, 0, 1)

    x_adv = x_adv.squeeze(0)
#    x_adv = x_adv.mul(torch.FloatTensor(std).to(device).view(3,1,1))
#    x_adv = x_adv.add(torch.FloatTensor(mean).to(device).view(3,1,1)).to("cpu").numpy()#reverse of normalization op
    x_adv = np.transpose( x_adv , (1,2,0))   # C X H X W  ==>   H X W X C
    x_adv = np.clip(x_adv, 0, 1)
    
    im = Image.fromarray(np.uint8(x_adv*255), "RGB")
    im = im.resize((64,64))
#    plt.imshow(im)
#    plt.show()
    date_string = time.strftime("%Y-%m-%d-%H_%M")
    image_path = "./adversarials/adverimg_{}_eps{}_iter{}_alpha{}_label{}.png".format(date_string, epsilon, iteration, alpha, y_target_label)
    print("saving image at {}...".format(image_path))
    im.save(image_path)
    
    x_grad = x_grad.squeeze(0).detach().to("cpu").numpy()
    x_grad = np.transpose(x_grad, (1,2,0))
    x_grad = np.clip(x_grad, 0, 1)
    
    figure, ax = plt.subplots(1,2, figsize=(5,5))
    ax[0].imshow(x)
    ax[0].set_title('Clean Example', fontsize=8)
    
    
#    ax[1].imshow(x_grad)
#    ax[1].set_title('Perturbation', fontsize=15)
#    ax[1].set_yticklabels([])
#    ax[1].set_xticklabels([])
#    ax[1].set_xticks([])
#    ax[1].set_yticks([])

    
    ax[1].imshow(x_adv)
    ax[1].set_title('Adversarial Example', fontsize=8)
    
    ax[0].axis('off')
    ax[1].axis('off')

#    ax[0].text(1.1,0.5, "+{}*".format(round(epsilon,3)), size=10, ha="center", 
#             transform=ax[0].transAxes)
    
    ax[0].text(0.5,-0.13, "Prediction: {}\n Probability: {:.4f}".format(clean_pred, clean_prob), 
              size=8, ha="center", 
              transform=ax[0].transAxes)
    
#    ax[1].text(1.1,0.5, " = ", size=15, ha="center", transform=ax[1].transAxes)

    ax[1].text(0.5,-0.13, "Prediction: {}\n Probability: {:.4f}".format(adv_pred, adv_prob), 
              size=8, ha="center", 
              transform=ax[1].transAxes)
    

    plt.show()
    
    
def moveUsedImages():
    source = "./Images/originals/"
    destination = "./usedImages/"
    
    
    files = os.listdir(source)
    
    for file in files:
            shutil.move(os.path.join(source, file), destination)

 
if __name__ == "__main__":
    
    model.loadModel(pretrained_model = "saved_model_state_CNN_final.pth")
    model = model.to(device)
    for i in range(3):
        for data in testloader:            
            image, output, x_pred, x_pred_prob = predictImage(data)
            createIterativeAdversarial(image, i, output, x_pred, x_pred_prob)
    moveUsedImages()
    print("finished.")
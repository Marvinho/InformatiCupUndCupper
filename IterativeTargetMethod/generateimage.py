# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 18:41:40 2018

@author: MRVN
"""

from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
import shutil
import os
from tkinter.filedialog import askopenfilename


def createImage(color):
    
    w, h = 64, 64      # w * h
    print("creating image...")
    if(color == "Random"):        
        test_image = np.random.randint(256, size = (w, h, 3), dtype = np.uint8)
        test_image = Image.fromarray(test_image, 'RGB')
    else:   
        test_image = Image.new("RGB", size = (w, h), color = color)
    
    plt.imshow(test_image)
    plt.show()    
    print("saving image...")
    date_string = time.strftime("%Y-%m-%d-%H_%M_%S")
    test_image.save("./Images/originals/org_img_{}.png".format(date_string))

def copyImage(src, dst = "./Images/originals/baseImage.png"):
    shutil.copyfile(src, dst)
    
def openDir(foldername):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, foldername)
    os.startfile(filename)

def chooseImage(src, dst = "./Images/originals/baseImage.png"):
    shutil.copyfile(src, dst)
    
if __name__ == "__main__":
    createImage(random = False, color = "magenta")

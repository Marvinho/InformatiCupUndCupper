# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 18:41:40 2018

@author: MRVN
"""

from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt


def createImage(random, color = None):
    
    w, h = 64, 64      # w * h
    print("creating image...")
    if(random == True):        
        test_image = np.random.randint(256, size = (w, h, 3), dtype = np.uint8)
        test_image = Image.fromarray(test_image, 'RGB')
    else:
        if(color == None):
            color = 0    
        test_image = Image.new("RGB", size = (w, h), color = color)
    
    plt.imshow(test_image)
    plt.show()    
    print("saving image...")
    date_string = time.strftime("%Y-%m-%d-%H_%M_%S")
    test_image.save("./Images/originals/org_img_{}.png".format(date_string))


if __name__ == "__main__":
    createImage(random = False, color = "magenta")

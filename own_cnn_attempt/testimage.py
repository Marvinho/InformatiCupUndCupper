# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 12:43:57 2018

@author: MRVN
"""
import os
import requests
import time

all_top_confidences = []
   
def testImagesOnNetwork():
    # tests all images that are in the "adversarials" folder
    url = "https://phinau.de/trasi"
    key = {"key" : "raekieh3ZofooPhaequoh9oonge8eiya"}
    dirs = os.listdir("./adversarials" )
    
    for file in dirs:
#        print(file)
        files = {"image": open("./adversarials/{}".format(file), "rb")}
        r = requests.post(url, data = key, files = files)      
        confidences = r.json()
        
        top_confidence = confidences[0]["confidence"]
        top_class = confidences[0]["class"]
        all_top_confidences.append([file, top_class, top_confidence])
#        time.sleep(1)
    print(all_top_confidences)
    print()
    print(sorted(all_top_confidences, key=lambda confidence: confidence[2], reverse = True)[:5])


        
if __name__ == "__main__":
    testImagesOnNetwork()
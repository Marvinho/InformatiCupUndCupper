# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 12:43:57 2018

@author: MRVN
"""
import os
import requests
#import time


   
def testImagesOnNetwork():
    
    # tests all images that are in the "adversarials" folder
    all_top_confidences = []
    url = "https://phinau.de/trasi"
    key = {"key" : "raekieh3ZofooPhaequoh9oonge8eiya"}
    dirs = os.listdir("./adversarials" )
    
    for file in dirs:

        with open("./adversarials/{}".format(file), "rb") as f:
            
            files = {"image": f}
            r = requests.post(url, data = key, files = files)
#            print(r.status_code)
            if(r.status_code != 200):
                print("too many requests or sth else went wrong")
                break
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
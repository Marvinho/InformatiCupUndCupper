import matplotlib as mpl
mpl.use('Agg')
mpl.use('PS')   # generate postscript output by default

import torch
import data
from util import *
import torch.autograd
import torch.optim
from Network import Network
from const import *
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import os
import time
import torch.cuda
import numpy
import random
numpy.set_printoptions(threshold=numpy.nan)

if len(sys.argv) < 3:
    raise ValueError("Wrong number of arguments. Usage:\npython3 train.py PathToData OutputPath [PretrainedModel]")

dataPath = sys.argv[1]

outPath = sys.argv[2]
print("Saving results in " + outPath )
if not os.path.exists(outPath):
    os.makedirs(outPath)

trainData = data.loadSimData( dataPath, 10000 )
testData = data.loadSimData( dataPath, 1000, len(trainData) )
trainData = data.removeLargeDeformations( trainData, 0.1)
testData = data.removeLargeDeformations( testData, 0.1)
#trainData = data.augment( trainData )
#testData = data.augment( testData )
trainData = data.toTorch( trainData, cuda=True )
testData = data.toTorch( testData, cuda=True )
learningRate = 1e-4
#learningRate = 1e-7
#learningRate = 1e-5     # Seems to be good for Augmented, 10000 samples
#learningRate = 2e-5
#learningRate = 5e-5
#learningRate = 5e-6     # Seems to be good for SmallerForces, 10000 samples
print( "Number of training samples: {:d} x 8".format(len(trainData)) )
print( "Number of test samples: {:d}".format(len(testData)) )
modelPath = None
if len(sys.argv) > 3:
    modelPath = sys.argv[3]

trainErrsIndividual = [[],[],[]]
trainErrsTotal = []
testErrsIndividual = [[],[],[]]
testErrsTotal = []

def plotErrors( trainErrs, testErrs ):
    for i in range(0,3):
        trainErrsIndividual[i].append( trainErrs[i] )
        testErrsIndividual[i].append( testErrs[i] )
    trainErrsTotal.append( trainErrs[3] )
    testErrsTotal.append( testErrs[3] )

    plt.clf()
    
    plt.plot( trainErrsTotal, '-', label = "train total", color = (0.5,0,0.8) )
    for i in range(0,3):
        plt.plot( trainErrsIndividual[i], '-', label = "train {:d}".format(i), color = (0,0, 1-i/6) )

    plt.plot( testErrsTotal, '-', label = "test total", color = (0.5,0.8,0) )
    for i in range(0,3):
        plt.plot( testErrsIndividual[i], '-', label = "test {:d}".format(i), color = (0, 1-i/6,0) )

    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig( outPath + "/errors" )

net = Network()
if modelPath is not None:
    net.load_state_dict( torch.load(modelPath) )

net.cuda()

criterion0 = nn.MSELoss()
criterion1 = nn.MSELoss()
criterion2 = nn.MSELoss()
criterion0.cuda()
criterion1.cuda()
criterion2.cuda()

avgPool0to1 = torch.nn.AvgPool3d(2, stride=2)
avgPool0to2 = torch.nn.AvgPool3d(4, stride=4)

def test():
    net.eval()

    lossSum = numpy.array([0.0,0.0,0.0,0.0])

    for i in range( 0, len(testData) ):

        #d = data.sampleToTorch( trainData[shuffleIndex[i]], cuda = True )
        d = testData[i]

        # Forward pass and loss calculation:
        out0, out1, out2 = net.forward( d["mesh"], d["force"] )
            
        l0 = criterion0.forward( out0, d["target"] )
        l1 = criterion1.forward( out1, avgPool0to1( d["target"] ) )
        l2 = criterion2.forward( out2, avgPool0to2( d["target"] ) )
        #loss = (1-0.5*secondaryLossFactor)*l0 + 0.5*secondaryLossFactor*(l1 + l2)
        loss = l0 + 0.5*(l1 + l2)
        
        # Individual errors:
        lossSum[0] += l0.data[0]
        lossSum[1] += l1.data[0]
        lossSum[2] += l2.data[0]
        # Total err:
        lossSum[3] += loss.data[0]

    avgTestErrs = lossSum/len(testData)

    return avgTestErrs

#optimizer = torch.optim.Adam( net.parameters(), lr = learningRate, weight_decay=5e-9 )
#optimizer = torch.optim.Adam( net.parameters(), lr = learningRate, weight_decay=1e-7 )
optimizer = torch.optim.Adam( net.parameters(), lr = learningRate )
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.995 )
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.985 )
for epoch in range( 0, 999999999 ):

    lossSum = numpy.array([0.0,0.0,0.0,0.0])

    shuffleIndex = torch.randperm( len(trainData) )

    net.train()

    #secondaryLossFactor = max( (10-epoch)/10, 0 )
    #secondaryLossFactor = 1/(epoch+1)

    for i in range( 0, len(trainData) ):
        printProgressBar( i+1, len(trainData), "Epoch {}:".format( epoch ), decimals=2 )

        #d = data.sampleToTorch( trainData[shuffleIndex[i]], cuda = True )
        augmented = data.augmentSample( trainData[shuffleIndex[i]] )
        for d in augmented:
            optimizer.zero_grad()

            # Forward pass and loss calculation:
            out0, out1, out2 = net.forward( d["mesh"], d["force"] )
            
            l0 = criterion0.forward( out0, d["target"] )
            l1 = criterion1.forward( out1, avgPool0to1( d["target"] ) )
            l2 = criterion2.forward( out2, avgPool0to2( d["target"] ) )
            #loss = (1-0.5*secondaryLossFactor)*l0 + 0.5*secondaryLossFactor*(l1 + l2)
            loss = l0 + 0.5*(l1 + l2)
            
            # Individual errors:
            lossSum[0] = lossSum[0] + l0.data[0]
            lossSum[1] = lossSum[1] + l1.data[0]
            lossSum[2] = lossSum[2] + l2.data[0]
            # Total err:
            lossSum[3] = lossSum[3] + loss.data[0]

            loss.backward()
            optimizer.step()

    avgTrainErrs = lossSum/(len(trainData)*8)

    avgTestErrs = test()

    print( epoch, " Loss: {:e} {:e} {:e} \tTotal: {:e}".format(
                    avgTrainErrs[0], avgTrainErrs[1], avgTrainErrs[2], avgTrainErrs[3] ),
            "\n\tTest: {:e} {:e} {:e} \tTotal: {:e}".format(
                    avgTestErrs[0], avgTestErrs[1], avgTestErrs[2], avgTestErrs[3] ),
            "\n\tLearning Rate: {:.2e}".format( optimizer.param_groups[0]['lr'] ) )

    plotErrors( avgTrainErrs, avgTestErrs )

    scheduler.step()    # Decrease Learning Rate

    #if epoch % 5 == 0:
    torch.save(net.state_dict(), outPath + "/model" )



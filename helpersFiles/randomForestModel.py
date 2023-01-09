import cv2 as cv 
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import os
from sklearn.ensemble import RandomForestClassifier
from classifySinus import getCorrelate
from skimage.measure import label, regionprops, regionprops_table
import pandas as pd
import re
from modelHelper import*


# [0, 2, 1, 3]




def classify2D(pathToFolder, axis):
    trainY = np.repeat([2, 2, 3, 2, 1, 2, 0, 1, 1, 2, 1, 3], NUM_OF_SLICES)

    # create train set
    pathToTrain = pathToFolder+'/train'
    trainX = getFeatures2D(pathToTrain, axis)

    # create test set
    pathToTest= pathToFolder+'/test'
    testX = getFeatures2D(pathToTest, axis)

    RF = RandomForestClassifier()
    RF.fit(trainX, trainY)
    testY = RF.predict(testX)

    testPredictions = []

    for i in range(len(testY)//NUM_OF_SLICES):
        testPredictions.append(np.bincount(testY[i*NUM_OF_SLICES:i*NUM_OF_SLICES+NUM_OF_SLICES]).argmax())
    
    print(f'axis {axis}: ', testPredictions)
    print([classes[x] for x in testPredictions])



def classify3D(pathToFolder):
    trainY = np.repeat([2, 2, 3, 2, 1, 2, 0, 1, 1, 2, 1, 3], 1)

    # create train set
    pathToTrain = pathToFolder+'/train'
    trainX = getFeatures3D(pathToTrain)

    # create test set
    pathToTest= pathToFolder+'/test'
    testX = getFeatures3D(pathToTest)

    RF = RandomForestClassifier()
    RF.fit(trainX, trainY)
    testY = RF.predict(testX)
    
    print(testY)
    print([classes[x] for x in testY])


classify2D('/Users/elilevinkopf/Documents/Ex23A/FinalProject/validForRF', axis='A')
classify2D('/Users/elilevinkopf/Documents/Ex23A/FinalProject/validForRF', axis='C')

# classify3D('/Users/elilevinkopf/Documents/Ex23A/FinalProject/validForRF')
    
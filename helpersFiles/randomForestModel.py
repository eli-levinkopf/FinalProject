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




def classify(pathToFolder):
    # trainY = np.array([2, 2, 3, 2, 1, 2, 0, 1, 1, 2, 1, 3])
    trainY = np.array([2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 3, 3, 3])
    # trainY = np.array([2, 2, 2, 2, 3, 3, 2, 2, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1, 2, 2, 1, 1, 3, 3])


    # create train set
    pathToTrain = pathToFolder+'/train'
    trainX = getFeatures(pathToTrain, True)

    # create test set
    pathToTest= pathToFolder+'/test'
    testX = getFeatures(pathToTest, False)

    RF = RandomForestClassifier()
    RF.fit(trainX, trainY)
    testY = RF.predict(testX)
    
    print(testY)
    print([classes[x] for x in testY])



classify('/Users/elilevinkopf/Documents/Ex23A/FinalProject/validForRF')
# sinusSeg=  nib.load('/Users/elilevinkopf/Documents/Ex23A/FinalProject/sinusSeg/case#3.nii').get_fdata()
# createMoreSampels(sinusSeg)
    
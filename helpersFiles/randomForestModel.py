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


# [0, 2, 1, 3]

classes = {0: 'both healthy', 1: 'both sick', 2: 'left sick right healthy', 3: 'left healthy right sick'}

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def featureExtraction(sinusSegPath):
    sinusSeg = nib.load(sinusSegPath).get_fdata()
    # cx, cy, cz = ndi.center_of_mass(sinusSeg)
    # countNoneZeros = np.count_nonzero(sinusSeg)


def getFeaxtures(pathToFolder):
    lst = []
    for filename in sorted(os.listdir(pathToFolder), key=natural_keys):
        if filename == '.DS_Store':
            continue
        sinusSeg = nib.load(pathToFolder+ '/' + filename).get_fdata()
        label_img = label(sinusSeg)
        props = regionprops_table(label_img, properties=('centroid',
                                                 'axis_major_length',
                                                 'axis_minor_length'))
        props = pd.DataFrame(props)
        lst.append(np.concatenate((sinusSeg[::16, ::16, ::16].flatten(), props.to_numpy()[0])))
    return np.array(lst)


def classify(pathToFolder):
    trainY = np.array([2, 2, 3, 2, 1, 2, 0, 1, 1, 2, 1, 3])

    # create train set
    pathToTrain = pathToFolder+'/train'
    trainX = getFeaxtures(pathToTrain)

    # create test set
    pathToTest= pathToFolder+'/test'
    testX = getFeaxtures(pathToTest)

    RF = RandomForestClassifier()
    RF.fit(trainX, trainY)
    testY = RF.predict(testX)
    
    print(testY)
    print([classes[x] for x in testY])



classify('/Users/elilevinkopf/Documents/Ex23A/FinalProject/validForRF')
    
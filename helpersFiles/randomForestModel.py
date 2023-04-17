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


# test: [0, 2, 1, 3]
# train: [2, 2, 3, 2, 1, 2, 0, 1, 1, 2, 1, 3]

# test1 = [1, 2, 2, 3, 0, 1, 3, 1, 1]




def classify2D(pathToFolder, axis):
    # trainY = np.repeat([2, 2, 3, 2, 1, 2, 0, 1, 1, 2, 1, 3], NUM_OF_SLICES)
    trainY = np.repeat([2, 3, 2, 3, 2, 3, 3, 3, 0, 3, 1, 2, 3, 3, 1, 1, 1, 2, 2, 1, 2, 3, 2, 2, 3], NUM_OF_SLICES)

    # create train set
    pathToTrain = pathToFolder+'/train1'
    trainX = getFeatures2D(pathToTrain, axis)

    # create test set
    pathToTest= pathToFolder+'/test1'
    testX = getFeatures2D(pathToTest, axis)

    RF = RandomForestClassifier()
    RF.fit(trainX, trainY)
    testY = RF.predict(testX)


    testPredictions = []

    for i in range(len(testY)//NUM_OF_SLICES):
        testPredictions.append(np.bincount(testY[i*NUM_OF_SLICES:i*NUM_OF_SLICES+NUM_OF_SLICES]).argmax())
    
    # labelBinarizer = LabelBinarizer().fit(np.array([0,1, 2, 3]))
    # testYOneHot = labelBinarizer.transform(np.array(testPredictions)) 
    # classId = np.flatnonzero(labelBinarizer.classes_ == 0)[0]

    # RocCurveDisplay.from_predictions(
    # testYOneHot[:, classId],
    # np.array(testPredictions)[:, classId],
    # name="0 vs the rest",
    # color="darkorange",
    # )
    # plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    # plt.axis("square")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("One-vs-Rest ROC curves:\0 vs the rest")
    # plt.legend()
    # plt.show()

    print(f'axis {axis}: ', testPredictions)
    print([classes[x] for x in testPredictions])

    # plotRocCurve(RF.predict_proba(testX), f'random Forest 2D{axis}')
    print(f1_score(y_true=np.array([0, 2, 1, 3]), y_pred=testPredictions, average='micro'))


# {0: 'both healthy', 1: 'both sick', 2: 'left sick right healthy', 3: 'left healthy right sick'}
# [1, 2, 2, 3, 0, 1, 3, 1, 1]
def classify3D(pathToFolder):
    # trainY = np.repeat([2, 2, 3, 2, 1, 2, 0, 1, 1, 2, 1, 3], 1)
    trainY = [2, 3, 2, 3, 2, 3, 3, 3, 0, 3, 1, 2, 3, 3, 1, 1, 1, 2, 2, 1, 2, 3, 2, 2, 3]

    # create train set
    pathToTrain = pathToFolder+'/train1'
    trainX = getCountFeatures3D(pathToTrain)

    # create test set
    pathToTest= pathToFolder+'/test1'
    testX = getCountFeatures3D(pathToTest)

    RF = RandomForestClassifier()
    RF.fit(trainX, trainY)
    testY = RF.predict(testX)
    
    print(testY)
    print([classes[x] for x in testY])
    # plotRocCurve(testY, 'random Forest 3D')
    # print(f1_score(y_true=np.array([0, 2, 1, 3]), y_pred=testY, average='micro'))


# classify2D('/Users/elilevinkopf/Documents/Ex23A/FinalProject/validForRF', axis='A')
# classify2D('/Users/elilevinkopf/Documents/Ex23A/FinalProject/validForRF', axis='C')

classify3D('/Users/elilevinkopf/Documents/Ex23A/FinalProject/validForRF')
    
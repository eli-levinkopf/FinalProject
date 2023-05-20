
import cv2 as cv 
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import os
from classifySinus import getCorrelate, getCorrelate2D
from skimage.measure import label, regionprops, regionprops_table
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import re
from pandas_profiling import ProfileReport
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, f1_score
import matplotlib.pyplot as plt

NUM_OF_SLICES = 25

properties = [                                      'centroid',
                                                    'axis_major_length',
                                                    # 'axis_minor_length',
                                                    'area',
                                                    # 'area_convex',
                                                    # 'area_filled',
                                                    # 'area_bbox',
                                                    # 'equivalent_diameter_area',
                                                    # 'euler_number',
                                                    # 'feret_diameter_max',
                                                    'orientation',
                                                    ]
properties1 = ['centroid_x',
                                                    'centroid_y',
                                                    'axis_major_length',
                                                    'axis_minor_length',
                                                    'area',
                                                    'area_convex',
                                                    'area_filled',
                                                    'area_bbox',
                                                    'equivalent_diameter_area',
                                                    'euler_number',
                                                    'feret_diameter_max',
                                                    'orientation',
                                                     'correlation']



classes = {0: 'both healthy', 1: 'both sick', 2: 'left sick right healthy', 3: 'left healthy right sick'}


def create2DSampelsA(sinusSeg):
    sampels = []
    for i in range(NUM_OF_SLICES):
        sampels.append(sinusSeg[:, :, i])
    return sampels


def create2DSampelsC(sinusSeg):
    sampels = []
    for i in range(NUM_OF_SLICES):
        sampels.append(sinusSeg[:, i+50, :])
    return sampels


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def getFeatures2D(pathToFolder, axis):
    featuresMat = []
    for filename in sorted(os.listdir(pathToFolder), key=natural_keys):
        if filename == '.DS_Store':
            continue
        sinusSeg = nib.load(pathToFolder+ '/' + filename).get_fdata()
        
        if axis == 'A':
            subSampels = create2DSampelsA(sinusSeg)
        elif axis == 'C':
            subSampels = create2DSampelsC(sinusSeg)

        for sample in subSampels:
            # label_img = label(sample)
            # props = regionprops_table(label_img, properties=properties)
            # props = pd.DataFrame(props)
            
            # if props.empty:
            #     props = np.zeros(len(properties)+1)
            # else:
            #     props = props.to_numpy()[0]

            # features = np.append(props, getCorrelate2D(sample)[0])
            # featuresMat.append(features)
            
            featuresMat.append(sample.flatten())

    return np.array(featuresMat)


def getFeatures3D(pathToFolder):
    featuresMat = []
    for filename in sorted(os.listdir(pathToFolder), key=natural_keys):
        if filename == '.DS_Store':
            continue
        sinusSeg = nib.load(pathToFolder+ '/' + filename).get_fdata()
        label_img = label(sinusSeg)
        props = regionprops_table(label_img, properties=('centroid',
                                                # 'axis_major_length',
                                                # 'axis_minor_length',
                                                'area',
                                                'equivalent_diameter_area',
                                                ))
        props = pd.DataFrame(props)
        if props.to_numpy().size == 0:
            props = np.zeros((3))
        else:
            props = props.to_numpy()[0]
            props = np.delete(props, np.s_[1:3])
        features = np.append(props, getCorrelate(sinusSeg)[0])
        featuresMat.append(np.nan_to_num(features))

    return np.array(featuresMat)

def getCountFeatures3D(pathToFolder):
    featuresMat = []
    for filename in sorted(os.listdir(pathToFolder), key=natural_keys):
        if filename == '.DS_Store':
            continue
        sinusSeg = nib.load(pathToFolder+ '/' + filename).get_fdata()
        corr, rightSeg, leftSeg = getCorrelate(sinusSeg)
        featuresMat.append([corr, np.count_nonzero(rightSeg), np.count_nonzero(leftSeg)])
    return np.array(featuresMat)


def plotRocCurve(yTest, modelName):
    yTrue = np.array([0, 2, 1, 3])
    yTestOneClass = np.zeros(yTest.size)
    yTestOneClass[yTest == 0] = 1
    yTrueOneClass = np.zeros(yTrue.size)
    yTrueOneClass[yTrue == 0] = 1

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(yTrueOneClass, yTestOneClass, pos_label=0)
    roc_auc = auc(fpr, tpr)

    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    print(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve {modelName}')
    plt.legend(loc="lower right")
    plt.show()


# sinusSeg = nib.load('/Users/elilevinkopf/Documents/Ex23A/FinalProject/downSampledScans/case#3.nii.gz').get_fdata()
# create2DSampels(sinusSeg)

# mat = getFeatures('/Users/elilevinkopf/Documents/Ex23A/FinalProject/validForRF/train')
# df = pd.DataFrame(mat, columns=properties1)
# df['label'] = np.repeat([2, 2, 3, 2, 1, 2, 0, 1, 1, 2, 1, 3], NUM_OF_SLICES)
# profile = ProfileReport(df)
# profile.to_file("output.html")
# df.to_csv('/Users/elilevinkopf/Documents/Ex23A/FinalProject/featuresMat.csv')
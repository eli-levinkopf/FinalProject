
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
from pandas_profiling import ProfileReport



classes = {0: 'both healthy', 1: 'both sick', 2: 'left sick right healthy', 3: 'left healthy right sick'}


def createMoreSampels(sinusSeg):
    sampels = []
    shape = sinusSeg.shape
    # for i in range(3):
    #     if i == 0:
    #         sampels.append(sinusSeg[:, :int(0.4*shape[1]), :])
    #     elif i == 1:
    #         sampels.append(sinusSeg[:, int(0.4*shape[1]):int(0.6*shape[1]), :])
    #     else:
    #         sampels.append(sinusSeg[:, int(0.6*shape[1]):, :])

    return [sinusSeg, np.flip(sinusSeg, axis=1), np.flip(sinusSeg, axis=2)]



def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def featureExtraction(sinusSegPath):
    sinusSeg = nib.load(sinusSegPath).get_fdata()
    # cx, cy, cz = ndi.center_of_mass(sinusSeg)
    # countNoneZeros = np.count_nonzero(sinusSeg)


def getFeatures(pathToFolder, train=True):
    featuresMat = []
    for filename in sorted(os.listdir(pathToFolder), key=natural_keys):
        if filename == '.DS_Store':
            continue
        sinusSeg = nib.load(pathToFolder+ '/' + filename).get_fdata()
        if train: 
            subSampels = createMoreSampels(sinusSeg)
        else:
            subSampels = [sinusSeg]
        for sample in subSampels:
            label_img = label(sample)
            props = regionprops_table(label_img, properties=('centroid',
                                                    'axis_major_length',
                                                    'axis_minor_length',
                                                    'area',
                                                    'equivalent_diameter_area',
                                                    ))
            
            props = pd.DataFrame(props)
            props = props.to_numpy()[0]
            props = np.delete(props, np.s_[1:3])
            features = np.append(props, getCorrelate(sample)[0])
            featuresMat.append(features)
            # featuresMat.append((sample[::16, ::16, ::16].flatten()))
    return np.array(featuresMat)



# mat = getFeatures('/Users/elilevinkopf/Documents/Ex23A/FinalProject/validForRF/train')
# df = pd.DataFrame(mat, columns=['centroid_x',
#                                                     'centroid_y',
#                                                     'centroid_z',
#                                                     'axis_major_length',
#                                                     'axis_minor_length',
#                                                     'area',
#                                                     'area_convex',
#                                                     'area_filled',
#                                                     'area_bbox',
#                                                     'equivalent_diameter_area',
#                                                     'euler_number',
#                                                     'feret_diameter_max', 'correlation'])
# df['label'] = [2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 3, 3, 3]
# profile = ProfileReport(df)
# profile.to_file("output.html")
# df.to_csv('/Users/elilevinkopf/Documents/Ex23A/FinalProject/featuresMat.csv')

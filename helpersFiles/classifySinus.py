from scipy.signal import correlate
import nibabel as nib
import numpy as np
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize




SYMMETRY_TH = 0.55
TWO_HEALTHY_TH = 25000
SINGLE_HEALTHY_SINUS_TH = 3000


def sinusClassification(sinusSegPath):
    sinusSeg = nib.load(sinusSegPath).get_fdata()
    corr, rightSeg, leftSeg = getCorrelate(sinusSeg)
    if corr > SYMMETRY_TH: # in this case we assume this is a symmetric segmentation
        if np.count_nonzero(sinusSeg) > TWO_HEALTHY_TH: # both sinuses are healthy
            print("0: right sinus: healthy, left sinus: healthy")
            return 0
        else: # both sinuses are sick
            print("1: right sinus: sick, left sinus: sick")
            return 1

    else: # in this case we assume asymmetric segmentation
        if np.count_nonzero(sinusSeg) <= SINGLE_HEALTHY_SINUS_TH: 
            print("1: right sinus: sick, left sinus: sick")
            return 1

        else: # in this case we assume that only one of the sinuses is healthy.
            if np.count_nonzero(leftSeg) <= np.count_nonzero(rightSeg):
                print("2: right sinus: healthy, left sinus: sick")
                return 2
            else:
                print("3: right sinus: sick, left sinus: healthy")
                return 3


def getCorrelate(sinusSeg):
    rightSeg = sinusSeg[: int(sinusSeg.shape[0]/2), :, :].astype(int)
    leftSeg = sinusSeg[int(sinusSeg.shape[0]/2):, :, :].astype(int)
    if rightSeg.shape[0] != leftSeg.shape[0]:
        leftSeg = leftSeg[:-1, :, :]


    leftSeg = np.flip(leftSeg, axis=0)
    norm_a = np.linalg.norm(leftSeg)
    leftIsEmpty, rightIsEmpty = None, None
    if np.count_nonzero(leftSeg) == 0:
        norm_a = 1
        leftIsEmpty = True
    leftSegNorm = leftSeg / norm_a
    norm_b = np.linalg.norm(rightSeg)
    if np.count_nonzero(rightSeg) == 0:
        norm_b = 1
        rightIsEmpty  =True
    rightSegNorm = rightSeg / norm_b
    if leftIsEmpty and rightIsEmpty:
        return 1, rightSeg, leftSeg
    if leftIsEmpty or rightIsEmpty:
        return 0, rightSeg, leftSeg
    corr = np.correlate(leftSegNorm.flatten(), rightSegNorm.flatten())[0]
    # corr = np.corrcoef(leftSegNorm.flatten(), rightSegNorm.flatten())[0, 1]

    return corr, rightSeg,leftSeg


def getCorrelate2D(sinusSeg):
    rightSeg = sinusSeg[: int(sinusSeg.shape[0]/2), :].astype(int)
    leftSeg = sinusSeg[int(sinusSeg.shape[0]/2):, :].astype(int)
    if rightSeg.shape[0] != leftSeg.shape[0]:
        leftSeg = leftSeg[:-1, :]

    leftSeg = np.flip(leftSeg, axis=0)
    norm_a = np.linalg.norm(leftSeg)
    leftSegNorm = leftSeg / norm_a
    norm_b = np.linalg.norm(rightSeg)
    rightSegNorm = rightSeg / norm_b
    corr = np.correlate(leftSegNorm.flatten(), rightSegNorm.flatten())[0]
    corr = 0 if np.isnan(corr) else corr

    return corr, rightSeg,leftSeg



def splitSegmantation():
    """
    split a segmentation to leftSeg and rightSeg and save them in oneSideSegmantations floder.
    """
    for i in range(35):
        path = f'/Users/elilevinkopf/Documents/Ex23A/FinalProject/sinusSegmantation/case#{i}.nii'
        if os.path.isfile(path):
            sinusSeg = nib.load(path).get_fdata()
            rightSeg = sinusSeg[: int(sinusSeg.shape[0]/2), :, :].astype(int)
            leftSeg = sinusSeg[int(sinusSeg.shape[0]/2):, :, :].astype(int)
            nib.save(nib.Nifti1Image(rightSeg, None), f'/Users/elilevinkopf/Documents/Ex23A/FinalProject/oneSideSegmantations/case#{i}_right.nii.gz')
            nib.save(nib.Nifti1Image(leftSeg, None), f'/Users/elilevinkopf/Documents/Ex23A/FinalProject/oneSideSegmantations/case#{i}_left.nii.gz')

# from sklearn.metrics import roc_curve, auc, RocCurveDisplay, f1_score
# print(f1_score(y_true=np.array([0, 2, 1, 3]), y_pred=np.array([0, 3, 1, 3]), average='micro'))

# import re
# def atoi(text):
#     return int(text) if text.isdigit() else text

# def natural_keys(text):
#     return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# res = []
# # Loop over all files in specified folder
# folder = '/Users/elilevinkopf/Documents/Ex23A/FinalProject/Perfect segmentations/sinus segmentations'
# for filename in sorted(os.listdir(folder), key=natural_keys):
#     # Check if file is a .nii.gz file
#     if filename.endswith('.nii.gz'):
#         print(filename.split('/')[-1].split('.')[0])
#         res.append(sinusClassification(folder +'/' + filename))

# res = np.array(res)
# # classes = {0: 'both healthy', 1: 'both sick', 2: 'left sick right healthy', 3: 'left healthy right sick'}
# yTrue = np.array([2, 3, 2, 3, 2, 0, 0, 3, 0, 3, 1, 2, 0, 3, 1, 1, 1, 2, 2, 1, 1, 2, 2, 3, 0, 1, 3, 2, 3, 2, 2, 1, 3, 1])
# print(res == yTrue)

# yTestOneClass = np.zeros(yTest.size)
# yTestOneClass[yTest == 0] = 1
# yTrueOneClass = np.zeros(yTrue.size)
# yTrueOneClass[yTrue == 0] = 1

# # Compute ROC curve and ROC area
# fpr, tpr, _ = roc_curve(yTrueOneClass, yTestOneClass)
# roc_auc = auc(fpr, tpr)

# # Plot of a ROC curve for a specific class
# plt.figure()
# plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
# print(fpr, tpr)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title(f'ROC curve classify by TH')
# plt.legend(loc="lower right")
# plt.show()


# splitSegmantation()



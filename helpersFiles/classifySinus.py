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
        else: # both sinuses are sick
            print("1: right sinus: sick, left sinus: sick")

    else: # in this case we assume asymmetric segmentation
        if np.count_nonzero(sinusSeg) <= SINGLE_HEALTHY_SINUS_TH: 
            print("1: right sinus: sick, left sinus: sick")

        else: # in this case we assume that only one of the sinuses is healthy.
            if np.count_nonzero(leftSeg) <= np.count_nonzero(rightSeg):
                print("2: right sinus: healthy, left sinus: sick")
            else:
                print("3: right sinus: sick, left sinus: healthy")


def getCorrelate(sinusSeg):
    rightSeg = sinusSeg[: int(sinusSeg.shape[0]/2), :, :].astype(int)
    leftSeg = sinusSeg[int(sinusSeg.shape[0]/2):, :, :].astype(int)
    if rightSeg.shape[0] != leftSeg.shape[0]:
        leftSeg = leftSeg[:-1, :, :]

    leftSeg = np.flip(leftSeg, axis=0)
    norm_a = np.linalg.norm(leftSeg)
    leftSegNorm = leftSeg / norm_a
    norm_b = np.linalg.norm(rightSeg)
    rightSegNorm = rightSeg / norm_b
    corr = np.correlate(leftSegNorm.flatten(), rightSegNorm.flatten())[0]

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


# for i in [16, 24, 25, 33]:
#     path = f'/Users/elilevinkopf/Documents/Ex23A/FinalProject/sinusSeg/case#{i}.nii'
#     if os.path.isfile(path):
#         print(f'case#{i}')
#         sinusClassification(path)
# yTrue = np.array([0, 2, 1, 3])
# yTest = np.array([0, 3, 1, 3])
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



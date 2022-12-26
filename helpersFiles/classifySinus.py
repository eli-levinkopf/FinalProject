from scipy.signal import correlate
import nibabel as nib
import numpy as np
import os

SYMMETRY_TH = 0.55
TWO_HEALTHY_TH = 25000
SINGLE_HEALTHY_SINUS_TH = 3000


def sinusClassification(sinusSegPath):
    sinusSeg = nib.load(sinusSegPath).get_fdata()
    corr, rightSeg, leftSeg = getCorrelate(sinusSeg)

    if corr > SYMMETRY_TH: # in this case we assume this is a symmetric segmentation
        if np.count_nonzero(sinusSeg) > TWO_HEALTHY_TH: # both sinuses are healthy
            print("right sinus: healthy, left sinus: healthy")
        else: # both sinuses are sick
            print("right sinus: sick, left sinus: sick")

    else: # in this case we assume asymmetric segmentation
        if np.count_nonzero(sinusSeg) <= SINGLE_HEALTHY_SINUS_TH: 
            print("right sinus: sick, left sinus: sick")

        else: # in this case we assume that only one of the sinuses is healthy.
            if np.count_nonzero(leftSeg) <= np.count_nonzero(rightSeg):
                print("right sinus: healthy, left sinus: sick")
            else:
                print("right sinus: sick, left sinus: healthy")


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


for i in range(17, 18):
    path = f'/Users/elilevinkopf/Documents/Ex23A/FinalProject/sinusSegmantation/case#{i}.nii'
    if os.path.isfile(path):
        print(f'case#{i}')
        sinusClassification(path)

# splitSegmantation()

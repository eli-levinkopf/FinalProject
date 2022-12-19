from scipy.signal import correlate
import nibabel as nib
import numpy as np
import os

SYMMETRY_TH = 0.35
TWO_HEALTHY_TH = 25000
SINGLE_HEALTHY_SINUS_TH = 3000


def sinusClassification(sinusSegPath):
    sinusSeg = nib.load(sinusSegPath).get_fdata()
    rightSeg = sinusSeg[: int(sinusSeg.shape[0]/2), :, :].astype(int)
    leftSeg = sinusSeg[int(sinusSeg.shape[0]/2):, :, :].astype(int)
    if rightSeg.shape[0] != leftSeg.shape[0]:
        leftSeg = leftSeg[:-1, :, :]

    leftSeg = np.flip(leftSeg, axis=0)
    intesection = rightSeg & leftSeg
    union = rightSeg | leftSeg
    res = np.count_nonzero(intesection)/np.count_nonzero(union)
    print(res)

    if res > SYMMETRY_TH: # in this case we assume that the seg is a symetric
        if np.count_nonzero(sinusSeg) > TWO_HEALTHY_TH: # 2 sinuses are healthy
            print("right sinus: healthy, left sinus: healthy")
        else: # 2 sinuses are sick
            print("right sinus: sick, left sinus: sick")

    else: # in this case we assume that the seg is not a symetric
        if np.count_nonzero(sinusSeg) <= SINGLE_HEALTHY_SINUS_TH: 
            print("right sinus: sick, left sinus: sick")

        else: # in this case we assume that only one sinus is healthy.
            if np.count_nonzero(leftSeg) <= np.count_nonzero(rightSeg):
                print("right sinus: healthy, left sinus: sick")
            else:
                print("right sinus: sick, left sinus: healthy")


for i in range(35):
    path = f'/Users/elilevinkopf/Documents/Ex23A/FinalProject/sinusSegmantation/case#{i}.nii'
    if os.path.isfile(path):
        print(f'case#{i}')
        sinusClassification(path)

from skimage.filters import threshold_local
import nibabel as nib
from skimage.measure import label
import numpy as np
from skimage.morphology import area_closing, area_opening, binary_erosion, dilation, square, binary_dilation, star, ball
import os




def loadFile(niftiFilePath, caseIdx):
    niftiFile = nib.load(niftiFilePath)
    downSampledImg = niftiFile.get_fdata()[::4, ::4, ::4]
    downSampledFile = nib.Nifti1Image(downSampledImg, None)
    nib.save(downSampledFile, f'/Users/elilevinkopf/Documents/Ex23A/FinalProject/downSampledFiles/{caseIdx}.nii.gz')
    return downSampledFile

def postProcessing(imgToProcess):
    processedImg = area_closing(imgToProcess, area_threshold=64, connectivity=2)
    processedImg = area_opening(imgToProcess, area_threshold=64, connectivity=2)
    processedImg = binary_erosion(imgToProcess)
    return processedImg

def sinusSeg(pathToNifti):
    caseIdx = pathToNifti.split('/')[-1].split('.')[0]
    niftiFile = loadFile(pathToNifti, caseIdx)
    ctData= niftiFile.get_fdata()
    # thresholds = threshold_local(ctData, block_size=301)
    sinusSegmentation = ctData < -900

    sinusSegmentation[:, int(3*ctData.shape[1]/4): , :] = 0
    # sinusSegmentation[:, int(4*ctData.shape[1]/5): , :] = 0
    largestConnectedComponent = label(sinusSegmentation, connectivity=2)
    bins = np.bincount(largestConnectedComponent.flat, weights=sinusSegmentation.flat)
    bins[bins.argmax()] = -np.inf
    largestCC = (largestConnectedComponent == bins.argmax()).astype(float)

    nib.save(nib.Nifti1Image(largestCC, None), f'/Users/elilevinkopf/Documents/Ex23A/FinalProject/sinusSegmantation/{caseIdx}') 


for i in range(24, 25):
    path = f'/Users/elilevinkopf/Documents/Ex23A/FinalProject/ctScanNiftiFiles/case#{i}.nii.gz'
    if os.path.isfile(path):
        sinusSeg(path)

# sinusSeg('/Users/elilevinkopf/Documents/Ex23A/FinalProject/ctScanNiftiFiles/case#5.nii.gz')
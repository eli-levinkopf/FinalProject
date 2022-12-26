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
    # niftiFile = loadFile(pathToNifti, caseIdx)
    niftiFile = nib.load(pathToNifti)
    ctData= niftiFile.get_fdata()
    # thresholds = threshold_local(ctData, block_size=301)
    sinusSegmentation = ctData < -900

    sinusSegmentation[:, int(3*ctData.shape[1]/4): , :] = 0
    # sinusSegmentation[:, int(4*ctData.shape[1]/5): , :] = 0
    largestConnectedComponent = label(sinusSegmentation, connectivity=2)
    bins = np.bincount(largestConnectedComponent.flat, weights=sinusSegmentation.flat)
    bins[bins.argmax()] = -np.inf
    largestCC = (largestConnectedComponent == bins.argmax()).astype(float)

    nib.save(nib.Nifti1Image(largestCC, None), f'/Users/elilevinkopf/Documents/Ex23A/FinalProject/sinusSeg/{caseIdx}') 



def resizeScan():
    """
    resize the scan to uniform size (116, 116, 76) and save it to downSampledScans folder.
    """
    for i in range(35):
        path = f'/Users/elilevinkopf/Documents/Ex23A/FinalProject/downSampledFiles/case#{i}.nii.gz'
        if os.path.isfile(path):
            ctScan = nib.load(path).get_fdata()
            shape = ctScan.shape
            if (shape[0] < 116 or shape[1] < 116 or shape[2] < 76):
                 continue
            ctScan = ctScan[int((shape[0] - 116)/2):int(-np.ceil((shape[0] - 116)/2)), int((shape[1] - 116)/2):int(-np.ceil((shape[1] - 116)/2)), int((shape[2] - 76)/2):int(-np.ceil((shape[2] - 76)/2))]
            nib.save(nib.Nifti1Image(ctScan, None), f'/Users/elilevinkopf/Documents/Ex23A/FinalProject/downSampledScans/case#{i}.nii.gz')



# resizeScan()

for i in range(35):
    path = f'/Users/elilevinkopf/Documents/Ex23A/FinalProject/downSampledScans/case#{i}.nii.gz'
    if os.path.isfile(path):
        sinusSeg(path)


# sinusSeg('/Users/elilevinkopf/Documents/Ex23A/FinalProject/ctScanNiftiFiles/case#5.nii.gz')\


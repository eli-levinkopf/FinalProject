from skimage.filters import threshold_local
import nibabel as nib
from skimage.measure import label
import numpy as np
from skimage.morphology import area_closing, area_opening, binary_erosion, dilation, square, binary_dilation, star, ball
import os



def loadAndDownSampledFile(niftiFilePath):
    """
    This function loads a NIfTI image file, downsamples the image by a factor of 4 
    in all three dimensions, and returns the downsampled image as a NIfTI file.

    :param niftiFilePath: The file path of the NIfTI image file to be loaded and downsampled.
    :type niftiFilePath: str
    :return: A downsampled NIfTI image file.
    :rtype: nibabel.nifti1.Nifti1Image
    """
    niftiFile = nib.load(niftiFilePath)
    downSampledImg = niftiFile.get_fdata()[::4, ::4, ::4]
    return nib.Nifti1Image(downSampledImg, None)


def postProcessing(imgToProcess):
    """
    This function performs post-processing on a binary image using area closing, area opening, and binary erosion.

    :param imgToProcess: The binary image to be processed.
    :type imgToProcess: numpy.ndarray
    :return: The processed binary image.
    :rtype: numpy.ndarray
    """
    processedImg = area_closing(imgToProcess, area_threshold=64, connectivity=2)
    processedImg = area_opening(imgToProcess, area_threshold=64, connectivity=2)
    processedImg = binary_erosion(imgToProcess)
    return processedImg


def sinusSegmentation(pathToNifti):
    """
    This function performs segmentation on a given NIfTI image file to identify the sinuses.

    :param pathToNifti: The file path of the NIfTI image file.
    :type pathToNifti: str
    :return: None
    """
    # extract case index from file path
    caseIdx = pathToNifti.split('/')[-1].split('.')[0]
    # resize the scan to a uniform format
    niftiFile = resizeScan(pathToNifti)
    ctData= niftiFile.get_fdata()
    # thresholds = threshold_local(ctData, block_size=301)
    # perform segmentation to identify sinuses
    sinusSegmentation = ctData < -900
    sinusSegmentation[:, int(3*ctData.shape[1]/4): , :] = 0
    # identify largest connected component of segmented image
    largestConnectedComponent = label(sinusSegmentation, connectivity=2)
    bins = np.bincount(largestConnectedComponent.flat, weights=sinusSegmentation.flat)
    bins[bins.argmax()] = -np.inf
    largestCC = (largestConnectedComponent == bins.argmax()).astype(float)
    # save largest connected component as new NIfTI image file
    nib.save(nib.Nifti1Image(largestCC, None), f'/Users/elilevinkopf/Documents/Ex23A/FinalProject/sinusSeg/{caseIdx}') 


def resizeScan(pathToNifti):
    """
    Resizes a 3D CT scan to a uniform size of (116, 116, 76) and returns it as a NIfTI file object.

    :param pathToNifti: The path to the input NIfTI file.
    :type pathToNifti: str
    :return: A NIfTI file object containing the resized CT scan.
    :rtype: nibabel.nifti1.Nifti1Image
    """
    ctScan = loadAndDownSampledFile(pathToNifti).get_fdata()
    shape = ctScan.shape
    # Check if the shape of the downsampled scan is smaller than the target shape. If so, return without resizing the scan.
    if (shape[0] < 116 or shape[1] < 116 or shape[2] < 76):
            return
    ctScan = ctScan[int((shape[0] - 116)/2):int(-np.ceil((shape[0] - 116)/2)), int((shape[1] - 116)/2):
                    int(-np.ceil((shape[1] - 116)/2)), int((shape[2] - 76)/2):int(-np.ceil((shape[2] - 76)/2))]
    # nib.save(nib.Nifti1Image(ctScan, None), f'/Users/elilevinkopf/Documents/Ex23A/FinalProject/downSampledScans/case#{i}.nii.gz')
    return nib.Nifti1Image(ctScan, None)




for i in range(33, 34):
    path = f'/Users/elilevinkopf/Documents/Ex23A/FinalProject/ctScanNiftiFiles/case#{i}.nii.gz'
    if os.path.isfile(path):
        sinusSegmentation(path)




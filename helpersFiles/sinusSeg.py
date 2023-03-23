from skimage.filters import threshold_local
import nibabel as nib
from skimage.measure import label
import numpy as np
from skimage.morphology import area_closing, area_opening, binary_erosion, dilation, square, binary_dilation, star, ball
import os
import random


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
    processedImg = area_closing(imgToProcess, area_threshold=256, connectivity=2)
    processedImg = area_opening(imgToProcess, area_threshold=256, connectivity=2)
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
    # niftiFile = resizeScan(pathToNifti)
    niftiFile = nib.load(pathToNifti)
    ctData = niftiFile.get_fdata()
    # thresholds = threshold_local(ctData, block_size=301)
    # perform segmentation to identify sinuses
    sinusSegmentation = ctData < -800
    # sinusSegmentation[:, int(3*ctData.shape[1]/4): , :] = 0
    # identify largest connected component of segmented image
    largestConnectedComponent = label(sinusSegmentation, connectivity=2)
    bins = np.bincount(largestConnectedComponent.flat, weights=sinusSegmentation.flat)
    # bins[bins.argmax()] = -np.inf
    largestCC = (largestConnectedComponent == bins.argmax()).astype(float)

    largestCC[: int(0.25*largestCC.shape[0]), :, :] = 0
    largestCC[int(0.75*largestCC.shape[0]):, :, :] = 0
    largestCC[:, :int(0.4*largestCC.shape[1]), :] = 0
    largestCC[:, :, int(0.7*largestCC.shape[2]):] = 0
    largestCC[:, int(0.7*largestCC.shape[1]):, :] = 0
    
    # largestCC = postProcessing(largestCC)
    # save largest connected component as new NIfTI image file
    nib.save(nib.Nifti1Image(largestCC, None), f'/Users/elilevinkopf/Documents/Ex23A/FinalProject/{caseIdx}.nii.gz') 


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


# def reSize(input):
#     # (S, C, A)
#     # (468, 468, 407)
#     # case13, 18, 24, 32, 35 (600, 600, 312)
#     random.seed(1)
#     slicesToRemoveS = random.sample(range(600), 132)
#     slicesToRemoveC = random.sample(range(600), 132)
#     newScan = np.zeros((468, 468, 312))
#     newScan = np.delete(input, slicesToRemoveS, axis=0)
#     newScan = np.delete(newScan, slicesToRemoveC, axis=1)
    
#     matrixToAdd = np.zeros((468, 468, 47))
#     matrixToAdd = np.dstack((matrixToAdd, newScan))
#     matrixToAdd = np.dstack((matrixToAdd, np.zeros((468, 468, 48))))

#     # matrixToAddD = np.zeros((468, 468, 31))
#     # matrixToAddD = np.dstack((matrixToAddD, matrixToAdd))
#     # matrixToAddD = np.dstack((matrixToAddD, np.zeros((468, 468, 32))))
#     # return matrixToAddD
#     return matrixToAdd

# def preProcessing():
#     scan = nib.load('/Users/elilevinkopf/Documents/Ex23A/FinalProject/ctScanNiftiFiles/case#35.nii.gz').get_fdata()
#     seg = nib.load('/Users/elilevinkopf/Documents/Ex23A/FinalProject/Perfect segmentations/case#35.nii.gz').get_fdata()

#     newScan = reSize(scan)
#     newSeg = reSize(seg)
    
#     nib.save(nib.Nifti1Image(newScan, None), f'/Users/elilevinkopf/Documents/Ex23A/FinalProject/ctScanNiftiFiles/case#35.nii.gz')
#     nib.save(nib.Nifti1Image(newSeg, None), f'/Users/elilevinkopf/Documents/Ex23A/FinalProject/Perfect segmentations/case#35.nii.gz')


def reshape3DScan(scan: np.ndarray, targetShape: tuple) -> np.ndarray:
    """
    Reshapes a 3D numpy array to a target shape by adding or removing slices along each axis.

    If an axis of `scan` is larger than the corresponding value in `targetShape`, slices are removed uniformly along that axis.
    If an axis of `scan` is smaller than the corresponding value in `targetShape`, zeros are added to both sides of that axis.
    After reshaping, the data along the y-axis is flipped like a mirror.

    Parameters
    ----------
    scan : np.ndarray
        A 3D numpy array to be reshaped.
    targetShape : tuple
        A tuple of three integers representing the desired shape of the reshaped array.

    Returns
    -------
    np.ndarray
        A new 3D numpy array with shape `targetShape` and data from `scan` that has been reshaped and flipped along the y-axis.
    """
    
    # Create a copy of scan to avoid modifying the original array
    reshapedScan = scan.copy()
    
    # Loop over each axis
    for axis in range(3):
        # Calculate difference between target shape and current shape along this axis
        diff = targetShape[axis] - scan.shape[axis]
        
        # If difference is positive, add zeros to both sides of this axis
        if diff > 0:
            padWidth = diff // 2
            padTuple = [(0, 0)] * axis + [(padWidth, diff - padWidth)] + [(0, 0)] * (2 - axis)
            reshapedScan = np.pad(reshapedScan, padTuple)
        
        # If difference is negative, remove slices uniformly along this axis
        elif diff < 0:
            removeIndices = np.round(np.linspace(0, scan.shape[axis] - 1, abs(diff))).astype(int)
            indexObj = [slice(None)] * axis + [np.delete(np.arange(scan.shape[axis]), removeIndices)] + [slice(None)] * (2 - axis)
            reshapedScan = reshapedScan[tuple(indexObj)]
    
     # Flip data along y-axis like a mirror before returning it 
    reshapedScan[...] = np.flip(reshapedScan,axis=1)

    return reshapedScan


def preProcessing(folderPath: str, targetShape: tuple):
    """
    Applies the `reshape3DScan` function to all .nii.gz files in a specified folder.

    This function loads each .nii.gz file in the specified folder as a 3D numpy array using the nibabel library.
    It then applies the `reshape3DScan` function to this array with the specified `targetShape`. 
    The resulting reshaped array is saved back to disk, overwriting the original .nii.gz file.

    Parameters
    ----------
    folderPath : str
        The path to the folder containing the .nii.gz files to be processed.
    targetShape : tuple
        A tuple of three integers representing the desired shape of the reshaped arrays.

    Returns
    -------
    None
    """
    
    # Loop over all files in specified folder
    for filename in os.listdir(folderPath):
        # Check if file is a .nii.gz file
        if filename.endswith('.nii.gz'):
            # Load .nii.gz file as 3D numpy array using nibabel library
            filePath = os.path.join(folderPath, filename)
            scan = nib.load(filePath).get_fdata()
            
            # Apply reshape3DMatrix function to scan with specified targetShape
            reshapedScan = reshape3DScan(scan, targetShape)
            
            # Save reshaped scan back to disk, overwriting original .nii.gz file 
            nib.save(nib.Nifti1Image(reshapedScan, None), filePath)


# for i in range(44, 51):
#     path = f'/Users/elilevinkopf/Documents/Ex23A/FinalProject/ctScanNiftiFiles/case#{i}.nii.gz'
#     if os.path.isfile(path):
        # sinusSegmentation(path)
        # print(f'case{i}', nib.load(path).get_fdata().shape)


preProcessing(folderPath='/Users/elilevinkopf/Documents/Ex23A/FinalProject/untitled folder', targetShape=(468, 468, 407))
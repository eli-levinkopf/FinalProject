from email.mime import image
from tkinter import N
import nibabel as nib
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import area_closing, area_opening, remove_small_objects
from skimage.transform import resize
from matplotlib import pyplot as plt
import math
from multiprocessing import Pool, Array, Value
from functools import partial
from scipy import ndimage


# TODO: OOP

def segmentationByTH(niftiFile, imgData, Imin, Imax):

    #getting a pointer to the data
    # imgData = np.array(niftiFile.get_fdata())
    bones = (imgData >= Imin) & (imgData <= Imax)
    noBones = ~bones

    #Turning boolean to
    imgData[noBones] = 0
    imgData[bones] = 1

    # TODO: return 0/1 and save the files.
    return nib.Nifti1Image(imgData, niftiFile.affine)

def postProcessing(imgToProcess):
    processedImg = area_closing(imgToProcess, area_threshold=256, connectivity=2)
    processedImg = area_opening(imgToProcess, area_threshold=256, connectivity=2)
    return processedImg

def init(a_array, v):
    global array
    array = a_array


def helper(Imin, niftiFile, imgData):
    Imax = 1300 
    candidateFile = segmentationByTH(niftiFile, imgData, Imin, Imax)
    _, count = label(candidateFile.dataobj, connectivity=2, return_num=True)
    idx = int((Imin-150)/14)
    print(idx)
    nib.save(candidateFile, f"/Users/elilevinkopf/Documents/MIP/ex2/candidateFiles/{idx}.nii.gz")
    array[idx] = count


def skeletonTHFinder(niftiFile):
    fileName = niftiFile.split('/')[-1].split('.')[0]
    niftiFile = nib.load(niftiFile)
    # shape = niftiFile.get_fdata().shape

    # imgData = resize(niftiFile.get_fdata(), (1/(shape[0]/160), 1/(shape[1]/160) ,1/(shape[2]/130)), order=1, preserve_range=True)
    # imgData = niftiFile.get_fdata()

    # niftiFile = nib.Nifti1Image(imgData, niftiFile.affine)
    # nib.save(niftiFile, '/Users/elilevinkopf/Documents/MIP/ex2/Case5.0_CT.nii.gz')

    connectivityComponents = Array('i', [0] * 25)
    v = Value('i', 3)
    with Pool(4, initializer=init ,initargs=(connectivityComponents, v)) as p:
        p.map(partial(helper, niftiFile=niftiFile, imgData=imgData), list(range(150, 500, 14)))
    idx = np.argmin(connectivityComponents)
    minCountFile = nib.load(f"/Users/elilevinkopf/Documents/MIP/ex2/candidateFiles/{idx}.nii.gz")
    processedImg = postProcessing(minCountFile.get_fdata())
    processedFile = nib.Nifti1Image(processedImg, minCountFile.affine)

    # shape = processedImg.shape
    # processedImg = resize(processedImg, (shape[0]*3, shape[1]*3, shape[2]*3), order=1, preserve_range=True)
    # processedFile = nib.Nifti1Image(processedImg, minCountFile.affine)


    nib.save(processedFile, f"/Users/elilevinkopf/Documents/MIP/ex2/{fileName}_SkeletonSegmentation.nii.gz")
    plt.xlabel("Imin threshold")
    plt.ylabel("Number of connectivity components")
    plt.plot(list(range(150, 500, 14)), connectivityComponents)
    plt.show()





def AortaSegmentation(niftiFile, L1SegNiftiFile):
    pass

skeletonTHFinder('/Users/elilevinkopf/Documents/MIP/ex2/Case5_CT.nii.gz')




import nibabel as nib
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import area_closing, area_opening, remove_small_objects
from matplotlib import pyplot as plt
import math
from skimage.transform import resize
from nilearn.image import resample_img
from sklearn.metrics import f1_score



# TODO: OOP

def segmentationByTH(niftiFile, Imin, Imax, niftiImg=None):
    # niftiFile = nib.load(niftiFile)
    #getting a pointer to the data
    if niftiFile is None:
        imgData = np.array(niftiFile.get_fdata())
    else:
        imgData = niftiImg
    bones = (imgData >= Imin) & (imgData <= Imax)
    noBones = ~bones

    #Turning boolean to
    imgData[noBones] = 0
    imgData[bones] = 1

    # TODO: return 0/1 and save the files.
    return nib.Nifti1Image(imgData, niftiFile.affine)

def postProcessing(imgToProcess):
    # processedImg = area_closing(imgToProcess, area_threshold=256, connectivity=2)
    # processedImg = area_opening(imgToProcess, area_threshold=140, connectivity=2)
    processedImg = area_closing(imgToProcess, area_threshold=40, connectivity=2)
    processedImg = area_opening(imgToProcess, area_threshold=40, connectivity=2)
    return processedImg

def skeletonTHFinder(niftiFilePath):
    fileName = niftiFilePath.split('/')[-1].split('.')[0]
    originalShapeFile = nib.load(niftiFilePath)
    downsampledFile = resample_img(originalShapeFile, target_affine=np.eye(3)*2., interpolation='nearest')
    print(downsampledFile.shape)
    print(type(downsampledFile))
    nib.save(downsampledFile, f'/Users/elilevinkopf/Documents/MIP/ex2/{fileName}.0_CT.nii.gz')

    Imax = 1300
    connectivityComponents = []
    minCount = math.inf
    minCountFile = None
    for Imin in range(150, 500, 14):
        candidateFile = segmentationByTH(downsampledFile, Imin, Imax)
        _, count = label(candidateFile.dataobj, connectivity=2, return_num=True)
        connectivityComponents.append(count)
        minCountFile = candidateFile if minCount > count else minCountFile
        minCount = count if minCount > count else minCount
        print(Imin)

    processedImg = postProcessing(minCountFile.dataobj)
    processedFile = nib.Nifti1Image(processedImg, candidateFile.affine)
    # upsampledProcessedFile = resample_img(processedFile, target_affine=np.eye(3)*0.5, interpolation='nearest')
    # print(upsampledProcessedFile.shape)
    nib.save(processedFile, f"/Users/elilevinkopf/Documents/MIP/ex2/{fileName}_SkeletonSegmentation.nii.gz")

    plt.xlabel("Imin threshold")
    plt.ylabel("Number of connectivity components")
    plt.plot(list(range(150, 500, 14)), connectivityComponents)
    plt.show()
    return minCount

def AortaSegmentation(niftiFilePath, L1SegNiftiFile, aortaFilePath):
    L1File = nib.load(L1SegNiftiFile)
    niftiFile = nib.load(niftiFilePath)
    aortaFile = nib.load(aortaFilePath)
    niftiImg = niftiFile.get_fdata()[::6, ::6, ::6]
    downsampledImgL1 = L1File.get_fdata()[::6, ::6, ::6]
    aortaImg = aortaFile.get_fdata()[::6, ::6, ::6]
    # niftiImg = downsampledFile.get_fdata()
    # aortaImg = downsampledFileAorta.get_fdata()
    downsampledFileAorta = nib.Nifti1Image(aortaImg, None)
    downsampledFile = nib.Nifti1Image(niftiImg, None)

    # niftiImg = resize(niftiFile.get_fdata(), (100, 100, 150), preserve_range=True)
    # downsampledFile = nib.Nifti1Image(niftiImg, None)
    # aortaImg = resize(aortaFile.get_fdata(), (100, 100, 150), preserve_range=True)
    # downsampledFileAorta = nib.Nifti1Image(aortaImg, None)
    # downsampledFileL1 = resize(L1File.get_fdata(), (100, 100, 150), preserve_range=True)


    rotateL1Img = np.roll(downsampledImgL1, shift=(-6, -14) , axis=(0, 1))
    nib.save(nib.Nifti1Image(rotateL1Img, None), "/Users/elilevinkopf/Documents/MIP/ex2/tmpFile.nii.gz")
    
    nib.save(downsampledFileAorta, "/Users/elilevinkopf/Documents/MIP/ex2/Case1downAorta.nii.gz")
    # rotateFile = nib.Nifti1Image(rotateL1Img, downsampledFileL1.affine)
    # nib.save(rotateFile, "/Users/elilevinkopf/Documents/MIP/ex2/rotaeL1Case1.nii.gz")

    niftiImg[rotateL1Img == 0] = 0
    aortaImg[rotateL1Img == 0] = 0
    # FileToSave = nib.Nifti1Image(aortaImg, downsampledFileAorta.affine)
    # nib.save(FileToSave, "/Users/elilevinkopf/Documents/MIP/ex2/sliceAortaCase1.nii.gz")


    # newNifti = nib.Nifti1Image(niftiImg, niftiFile.affine)
    # nib.save(newNifti, "/Users/elilevinkopf/Documents/MIP/ex2/tmpCase1.nii.gz")

    connectivityComponents = []
    Imax = 200
    minCount = math.inf
    minCountFile = None
    # for Imin in range(50, 63, 14):
    candidateFile = segmentationByTH(downsampledFile, 110, Imax, niftiImg)
    _, count = label(candidateFile.dataobj, connectivity=1, return_num=True)
    connectivityComponents.append(count)
    minCountFile = candidateFile if minCount > count else minCountFile
    minCount = count if minCount > count else minCount
    print(count)
    nib.save(minCountFile, "/Users/elilevinkopf/Documents/MIP/ex2/tmpFile.nii.gz")

    processedImg = postProcessing(minCountFile.dataobj)
    processedFile = nib.Nifti1Image(processedImg, candidateFile.affine)
    nib.save(processedFile, "/Users/elilevinkopf/Documents/MIP/ex2/aortaSegCase1.nii.gz")
    
    print(f1_score(aortaImg.flatten(), processedImg.flatten()))


# skeletonTHFinder('/Users/elilevinkopf/Documents/MIP/ex2/Case1_CT.nii.gz')
AortaSegmentation('/Users/elilevinkopf/Documents/MIP/ex2/Case1_CT.nii.gz', '/Users/elilevinkopf/Documents/MIP/ex2/Case1_L1.nii.gz',
'/Users/elilevinkopf/Documents/MIP/ex2/Case1_Aorta.nii.gz')


import nibabel as nib
import numpy as np
from skimage.measure import label
from skimage.morphology import area_closing, area_opening, binary_erosion, dilation, square, binary_dilation, star, ball
from sklearn.metrics import f1_score
from nilearn.image import resample_img
from scipy.ndimage import generate_binary_structure
# import torchio as tio
from skimage.transform import resize


def IsolateBody(niftiFilePath): 
    fileName = niftiFilePath.split('/')[-1].split('.')[0]
    downSampledFile = loadFile(niftiFilePath, fileName)
    noGrayPixles = segmentationByTH(downSampledFile)
    noNoise = postProcessing(noGrayPixles.dataobj)
    largestConnectedComponent, x = label(noNoise, connectivity=2, return_num=True)
    largestCC = largestConnectedComponent == np.argmax(np.bincount(largestConnectedComponent.flat, weights=noNoise.flat))
    new_cc = np.zeros(largestCC.shape)
    new_cc[largestCC == True] = 1 
    niftiFile = nib.Nifti1Image(new_cc, None)
    nib.save(niftiFile, f'/Users/elilevinkopf/Documents/MIP/ex3/{fileName}_bodySegmentation.nii.gz')


def postProcessing(imgToProcess):
    processedImg = area_closing(imgToProcess, area_threshold=64, connectivity=2)
    processedImg = area_opening(imgToProcess, area_threshold=64, connectivity=2)
    processedImg = binary_erosion(imgToProcess)
    return processedImg
    

def segmentationByTH(niftiFile):
    Imin = -500
    Imax = 2000
    imgData = np.array(niftiFile.get_fdata())
    gray = (imgData >= Imin) & (imgData <= Imax)
    noGray = ~gray

    #Turning boolean to
    imgData[noGray] = 0
    imgData[gray] = 1
    # return imgData
    return nib.Nifti1Image(imgData, niftiFile.affine)


def loadFile(niftiFilePath, fileName):
    niftiFile = nib.load(niftiFilePath)

    # zeros = np.zeros((150, 185, 185))
    # zerosNifti = nib.Nifti1Image(zeros, np.eye(4))
    # downSampledFile = resample_img(niftiFile, target_affine=zerosNifti.affine, target_shape=(150, 185, 185), interpolation='nearest')
    # nib.save(downSampledFile, f'/Users/elilevinkopf/Documents/MIP/ex3/{fileName}_resample_CT.nii.gz')
    # return downSampledFile

    downSampledImg = niftiFile.get_fdata()[::6, ::6, ::6]
    downSampledFile = nib.Nifti1Image(downSampledImg, None)
    nib.save(downSampledFile, f'/Users/elilevinkopf/Documents/MIP/ex3/{fileName}_resample_CT.nii.gz')
    return downSampledFile



def IsolateBS(niftiFilePath):
    fileName = niftiFilePath.split('/')[-1].split('.')[0]
    bodySegmentation = nib.load(niftiFilePath)
    bodySegmentation = bodySegmentation.get_fdata()
    lungs = np.ones(bodySegmentation.shape)
    lungs[bodySegmentation == 1] = 0
    
    largestConnectedComponent = label(lungs, connectivity=2)
    largestCC = largestConnectedComponent == np.argmax(np.bincount(largestConnectedComponent.flat, weights=lungs.flat))
    lungs[largestCC == True] = 0
    lungs = area_opening(lungs, area_threshold=20000, connectivity=2)

    idxMaxima = lungs.sum(axis=(0,1)).argmax()
    idxArray = np.nonzero(lungs)
    idxMinima = idxArray[2][-1]
    lungs[:, :, idxMinima] = 1
    lungs[:, :, idxMaxima] = 1

    lungsNifti = nib.Nifti1Image(lungs, None)
    nib.save(lungsNifti, f"/Users/elilevinkopf/Documents/MIP/ex3/{fileName}_lungs.nii.gz")
    return idxMaxima, idxMinima

def ThreeDBand(bodySegmentationPath):
    fileName = bodySegmentationPath.split('/')[-1].split('.')[0]
    idxMaxima, idxMinima = IsolateBS(bodySegmentationPath)
    niftiFile = nib.load(f"/Users/elilevinkopf/Documents/MIP/ex3/{fileName}_lungs.nii.gz")
    lungsSegmentation = niftiFile.get_fdata()
    copyLungs = lungsSegmentation.copy()
    copyLungs[:, :, idxMinima] = 0
    copyLungs[:, :, idxMaxima] = 0
    lungsSegmentation[:,:,idxMaxima:-1] = 0
    lungsSegmentation[:,:,0:idxMinima] = 0
    lungsSegmentation[:, :, idxMinima] = 0

    # lungsSegmentation[copyLungs==0] = 0

    # notLungs = np.ones(lungsSegmentation.shape)
    # notLungs[lungsSegmentation == 1] = 0
    # notLungs[:,:,idxMaxima:-1] = 0
    # notLungs[:,:,0:idxMinima] = 0

    # largestConnectedComponent = label(lungsSegmentation, connectivity=2)
    # largestCC = largestConnectedComponent == np.argmax(np.bincount(largestConnectedComponent.flat, weights=lungs.flat))
    # newCC = np.zeros(largestCC.shape)
    # newCC[bodySegmentation == True] = 1

    lungsNifti = nib.Nifti1Image(lungsSegmentation, None)
    nib.save(lungsNifti, f"/Users/elilevinkopf/Documents/MIP/ex3/{fileName}_slice_around_lungs.nii.gz")


def MergedROI(pathToLungsSlice, pathToAortaSeg, pathToSkeletonSeg):
    lungsSlice = nib.load(pathToLungsSlice).get_fdata()
    aortaSeg = nib.load(pathToAortaSeg).get_fdata()
    skeletonSeg = nib.load(pathToSkeletonSeg).get_fdata()

    idxArray = np.nonzero(lungsSlice)
    idxMaximaAorta = idxArray[2].min()
    aortaSeg[:,:,idxMaximaAorta:-1] = 0

    idxArray = np.nonzero(aortaSeg)
    idxMinimaAorta = idxArray[2].min()
    skeletonSeg[:,:,idxMaximaAorta:skeletonSeg.shape[2]] = 0
    skeletonSeg[:,:,0:idxMinimaAorta] = 0

    skeletonSeg[:, 0, idxMinimaAorta:idxMaximaAorta+1] = 1
    skeletonSeg[:, skeletonSeg.shape[1]-1, idxMinimaAorta:idxMaximaAorta] = 1
    skeletonSeg[0, :, idxMinimaAorta:idxMaximaAorta+1] = 1
    skeletonSeg[skeletonSeg.shape[0]-1, :, idxMinimaAorta:idxMaximaAorta] = 1

    skeletonSegSlice = nib.Nifti1Image(skeletonSeg, None)
    aortaSlice = nib.Nifti1Image(aortaSeg, None)

    nib.save(aortaSlice, "/Users/elilevinkopf/Documents/MIP/ex3/aortaSlice.nii.gz")
    nib.save(skeletonSegSlice, "/Users/elilevinkopf/Documents/MIP/ex3/skeletonSegSegSlice.nii.gz")


def liverROI(pathToCTScan ,pathToAortaSeg, pathToBodySegmentation):
    CTScan = nib.load(pathToCTScan).get_fdata()
    aortaSeg = nib.load(pathToAortaSeg).get_fdata()
    bodySegmentation = nib.load(pathToBodySegmentation).get_fdata()

    liverROI = np.roll(aortaSeg, shift=(25, 10) , axis=(0,2))  

    # booleanCTScan = (CTScan >= -100) & (CTScan <= 200)
    # booleanLiverROI = liverROI==1
    # booleanfinalROI = booleanCTScan & booleanLiverROI

    # finalROI = np.zeros(booleanfinalROI.shape)
    # finalROI[booleanfinalROI == True] = 1

    liverROINifti = nib.Nifti1Image(liverROI, None)
    nib.save(liverROINifti, "/Users/elilevinkopf/Documents/MIP/ex3/liverROI.nii.gz")


def findSeeds (pathToROI):
    ROI = nib.load(pathToROI).get_fdata()
    idxArray = np.nonzero(ROI)
    seedsMetrix = np.zeros(ROI.shape)
    for i in range(0, len(idxArray[0]), len(idxArray[0]) // 200):
        seedsMetrix[idxArray[0][i], idxArray[1][i], idxArray[2][i]] = 1
    return seedsMetrix 


def multipleSeedsRG(pathToCTScan, pathToROI):
    CTScanImg = nib.load(pathToCTScan).get_fdata() 
    seedsMetrix = nib.load(pathToROI).get_fdata()

    seedsMetrix = binary_dilation(image=seedsMetrix, footprint=generate_binary_structure(CTScanImg.ndim, 3))

    liverSegMetrix = np.zeros(seedsMetrix.shape)
    liverSegMetrix[seedsMetrix==True] = 1
    liverSeg = nib.Nifti1Image(liverSegMetrix, None)
    nib.save(liverSeg, "/Users/elilevinkopf/Documents/MIP/ex3/liverSeg.nii.gz")


# IsolateBody("/Users/elilevinkopf/Documents/MIP/ex3/Case1_CT.nii.gz")
# IsolateBS("/Users/elilevinkopf/Documents/MIP/ex3/Case1_CT_SkeletonSegmentation.nii.gz")

# ThreeDBand("/Users/elilevinkopf/Documents/MIP/ex3/Case1_CT_SkeletonSegmentation.nii.gz")
# loadFile("/Users/elilevinkopf/Documents/MIP/ex3/Case1_Aorta.nii.gz", "Case1_Aorta")
# MergedROI("/Users/elilevinkopf/Documents/MIP/ex3/Case1_CT_SkeletonSegmentation_slice_around_lungs.nii.gz", "/Users/elilevinkopf/Documents/MIP/ex3/Case1_Aorta_resample_CT.nii.gz", "/Users/elilevinkopf/Documents/MIP/ex3/Case1_CT_SkeletonSegmentation.nii.gz")
# liverROI("/Users/elilevinkopf/Documents/MIP/ex3/Case1_CT_resample_CT.nii.gz", "/Users/elilevinkopf/Documents/MIP/ex3/aortaSegCase1.nii.gz", "/Users/elilevinkopf/Documents/MIP/ex3/Case1_CT_bodySegmentation.nii.gz")

multipleSeedsRG("/Users/elilevinkopf/Documents/MIP/ex3/Case1_CT_resample_CT.nii.gz", "/Users/elilevinkopf/Documents/MIP/ex3/liverROI.nii.gz")
# loadFile("/Users/elilevink opf/Documents/MIP/ex3/Case1_CT.nii.gz", "Case1_CT")
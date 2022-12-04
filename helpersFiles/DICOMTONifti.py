import dicom2nifti
import dicom2nifti.settings as settings
import os
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from pydicom import dcmread
from pydicom.data import get_testdata_file
import nibabel as nib
import numpy as np

def convertDicom2Nifti(pathToFolder):
    for i in range(11, 12):
        dirSize = 0
        if len(os.listdir(pathToFolder + f"/case #{i}")) == 0:
            continue
        folder_list = (os.listdir(pathToFolder + f"/case #{i}/DICOM/00000001/"))
        for folder in folder_list:
            folder_to_convert = pathToFolder + f"/case #{i}/DICOM/00000001/"+folder
            if os.path.isdir(folder_to_convert) and len(os.listdir(folder_to_convert)) > dirSize:
                dirSize = len(os.listdir(folder_to_convert))
                dicom2nifti.convert_directory(folder_to_convert, f"/Users/elilevinkopf/Documents/Ex23A/FinalProject/ctScanNiftiFiles/case#{i}")



convertDicom2Nifti("/Users/elilevinkopf/Documents/Ex23A/FinalProject/ALL SCANS")


# dicom2nifti.dicom_series_to_nifti("/Users/elilevinkopf/Downloads/SinusCT/SINUS1/DICOM/00000001/00000001", "/Users/elilevinkopf/Downloads/ctScanNiftiFiles/DentalCT/dental_case_1")
# dicom2nifti.dicom_series_to_nifti("/Users/elilevinkopf/Downloads/DentalCT/DENTAL1/DICOM/00000001/00000002", "/Users/elilevinkopf/Downloads/ctScanNiftiFiles/DentalCT/dental_case_1")


# dicom = get_testdata_file("/Users/elilevinkopf/Downloads/DentalCT/DENTAL1/DICOM/00000001/00000002/00000000.dcm")
# ds = dcmread(dicom)
# data = ds.PixelData
# niftiFile = nib.Nifti1Image(data, None)
# nib.save(niftiFile, f'/Users/elilevinkopf/Downloads/ctScanNiftiFiles/DentalCT/Dental1.nii.gz')

# dicom = pydicom.read_file("/Users/elilevinkopf/Downloads/DentalCT/DENTAL1/DICOM/00000001/00000002/00000000")
# data = dicom.pixel_array
# new_data = np.zeros((data.shape[1], data.shape[0], data.shape[2]))

# new_data = np.swapaxes(data, 0, 2)
# new_data = np.rot90(new_data, axes=(2, 0))

# niftiFile = nib.Nifti1Image(new_data, None)
# nib.save(niftiFile, f'/Users/elilevinkopf/Downloads/ctScanNiftiFiles/DentalCT/Dental1.nii.gz')

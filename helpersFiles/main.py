import argparse
import os
import nibabel as nib

import helpers.preProcess as preProcess
import helpers.detection as detection

# Command to run the inference using nnUNet_predict tool
INFERENCE_COMNAND = 'nnUNet_predict -i nnUNet_raw_data_base/nnUNet_raw_data/Task509_sinus_bone_segmantation/imagesTs -o output_task509 -t Task509_sinus_bone_segmantation -m 3d_fullres -f all'

def main():
    # Set up the argument parser for command-line arguments
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='The program get a CT scan and returns (x,y,z) coordinates of anomalies points (if exists)')
    parser.add_argument('-f', '--file', required=True)
    args = parser.parse_args()
    # Check if the specified file is valid and has the correct extension
    if not os.path.isfile(args.file) or not args.file.endswith('nii.gz'):
        raise Exception ('Invalid file')
    
    scan = nib.load(args.file).get_fdata()
    # Normalize and reshape the 3D scan
    normalized_scan = preProcess.reshape3DScan(scan)
    nib.save(nib.Nifti1Image(normalized_scan, None), args.file)
    # Run the inference using the nnUNet_predict tool
    os.system(INFERENCE_COMNAND)
    # Detect anomalies in the output file generated by the inference
    detection.detect_anomalies('output_task509/sinus_bone_001.nii.gz')


if _name_ == '_main_':
    main()
import numpy as np


def reshape3DScan(scan: np.ndarray, targetShape: tuple=(468, 468, 407)) -> np.ndarray:
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
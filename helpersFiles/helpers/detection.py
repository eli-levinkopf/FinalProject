import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import area_closing
import matplotlib.pyplot as plt 
import os

def process_segmentation(segmentation_3D_path):
    """
    This function processes a 3D segmentation matrix by projecting it to a 2D matrix, filling the holes in the 2D matrix,
    and returning a matrix with the holes marked with ones.
    
    :param segmentation_path: Path to the 3D segmentation matrix
    :return: Tuple containing the filled regions and the rounded centers of mass
    """
    segmentation_3D = nib.load(segmentation_3D_path).get_fdata()
    indices_of_z_segmentation = np.argwhere(segmentation_3D == 1)
    if (np.max(indices_of_z_segmentation[:,2]) > 140): # TODO: Think about this threshold
        # Fill the lower part of the segmentation with zeros
        segmentation_3D[:, :, :segmentation_3D.shape[1]//5] = 0
    # Project the 3D matrix to a 2D matrix
    segmentation_2D = np.sum(segmentation_3D, axis=2) != 0
    # Fill the holes in the 2D matrix
    filled_matrix = ndi.binary_fill_holes(segmentation_2D).astype(int)

    # Subtract to get a matrix with the holes marked with ones
    inverse_2D_segmentation = filled_matrix - segmentation_2D

    # TODO: delete these lines
    # plt.imshow(filled_matrix, cmap='gray')
    # plt.show()

    # plt.imshow(inverse_2D_segmentation, cmap='gray')
    # plt.show()

    return inverse_2D_segmentation, segmentation_3D



def find_dental_anomalies_edges(inverse_2D_segmentation):
    """
    Find the edges of 'islands' of ones in a 2D binary matrix.
    
    :param matrix: 2D list of integers representing the binary matrix
    :return: List of tuples representing the coordinates of the edges of 'islands' of ones
    """
    # Convert input matrix to a numpy array
    inverse_2D_segmentation = np.array(inverse_2D_segmentation)
    edges = []

    # Find cells with value 1 that are adjacent to cells with value 0
    edge_mask = np.logical_and(inverse_2D_segmentation == 1, np.logical_or.reduce([np.roll(inverse_2D_segmentation, shift, axis) == 0 \
                                 for axis, shift in [(0, -1), (0, 1), (1, -1), (1, 1)]]))
    
    # Get the row and column indices of the edges
    row_indices, col_indices = np.where(edge_mask)
    
    # Add the edges to the list
    for i, j in zip(row_indices, col_indices):
        for k in range(-3, 4):
            edges.append((i + k, j))
        for k in range(-3, 4):
            edges.append((i, j + k))

    # TODO: delete these lines
    # x = [edge[1] for edge in edges]
    # y = [edge[0] for edge in edges]
    # plt.scatter(x, y, s=0.1 ,c='red')
    # plt.show()

    return edges


def detect_dental_anomalies1(points, original_segmentation):
    """
    Find the centers of mass of dental anomalies points in a 3D segmentation.
    
    :param points: List of lists representing the x and y coordinates of points in the 2D plane
    :param original_segmentation: 3D numpy array representing the original segmentation (segmentation of the
    sinus bone with holes in the dental anomalies points).
    :return: List of lists representing the centers of mass of the dental anomalies points in the 3D segmentation (x, y, z).
    """
    if (len(points) == 0):
        return []
    # Create a 2D result matrix
    result_matrix = np.zeros((original_segmentation.shape[0], original_segmentation.shape[1]))
    
    # Set the values at the given points to 1
    points = np.array(points)
    result_matrix[points[:, 0], points[:, 1]] = 1
    
    # Label the connected components in the 2D result matrix
    labels_2D_matrix, num_labels = ndi.label(result_matrix)
    
    # Create a 3D result matrix
    result_3D_matrix = np.zeros_like(original_segmentation)
    
    # Find the z coordinate for each label
    z_coordinates = np.zeros(num_labels)
    for i in range(1, num_labels + 1):
        mask = labels_2D_matrix == i
        masked_original_segmentation = np.sum(original_segmentation * mask[..., np.newaxis], axis=(0, 1))
        z_coordinate = np.sum(masked_original_segmentation * np.arange(masked_original_segmentation.size)) / np.sum(masked_original_segmentation)
        z_coordinates[i - 1] = z_coordinate
    
    # Set the values in the 3D result matrix to 1 at the given coordinates for each label
    for i in range(1, num_labels + 1):
        mask = labels_2D_matrix == i
        result_3D_matrix[mask, int(z_coordinates[i - 1])] = 1
    
    # Label the connected components in the 3D result matrix
    labels, num_labels = ndi.label(result_3D_matrix)
    # Get the size of every connected components in the 3D result matrix
    label_sizes = np.bincount(labels.ravel())

    # Find the center of mass for each label
    centers_of_mass = []
    for i in range(1, num_labels + 1):
        if label_sizes[i] < 100: # drop elements that are smaller than 100
            continue
        center_of_mass = ndi.center_of_mass(labels == i)
        centers_of_mass.append(center_of_mass)
    
    return np.rint(centers_of_mass).astype(int)


def detect_anomalies(segmentation_3D_path):
    inverse_2D_segmentation, segmentation_3D = process_segmentation(segmentation_3D_path)
    edges = find_dental_anomalies_edges(inverse_2D_segmentation)
    anomalies = detect_dental_anomalies1(edges, segmentation_3D)
    return anomalies
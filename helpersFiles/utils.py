import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import os
from skimage.morphology import area_closing
import cv2
import matplotlib.pyplot as plt


def defaultSegmentation(folderPath):
    # Loop over all files in specified folder
    for filename in os.listdir(folderPath):
        # Check if file is a .nii.gz file
        if filename.endswith('.nii.gz'):
            # Load .nii.gz file as 3D numpy array using nibabel library
            filePath = os.path.join(folderPath, filename)
            scan = nib.load(filePath).get_fdata()
            scan[scan == 2] = 0
            nib.save(nib.Nifti1Image(scan, None), filePath)


def count_unique_values(path_to_folder):
    files = sorted(os.listdir(path_to_folder))
    for ounput_file, case in zip(files, [3, 5, 6, 10, 17, 18, 19, 25, 44]):
        if ounput_file.endswith('.nii.gz'):
            data_output = nib.load(os.path.join(path_to_folder, ounput_file)).get_fdata()
            data_true = nib.load(os.path.join(path_to_folder,
                        f'/Users/elilevinkopf/Documents/Ex23A/FinalProject/Perfect segmentations/penetration segmentations/case#{case}.nii.gz')).get_fdata()
            unique_values_output = np.unique(data_output[data_output != 0])
            unique_values_true = np.unique(data_true[data_true != 0])
            print(f'{ounput_file}: {len(unique_values_output)} unique values')
            print(f'case#{case}_true: {len(unique_values_true)} unique values')


PATH_TO_TRUE_SEGMENTATION = '/Users/elilevinkopf/Documents/Ex23A/FinalProject/Perfect segmentations/penetration segmentations/all_classes_one_label/case#'

def count_intersections(path_to_folder):
    """
    This function counts the number of intersections between connected components of the true segmentation and the output segmentation (3D matrices).
    
    :param path: The path to the directory containing the output segmentations.
    :type path: str
    """
    # Get a sorted list of all files in the specified directory
    files = sorted(os.listdir(path_to_folder))

    total_features_true, total_intersections, total_features_output = 0, 0, 0
    
    # Iterate over pairs of files and case numbers
    for output_file, case in zip(files, [3, 5, 6, 10, 17, 18, 19, 25, 44]):
        # Skip files that do not have the expected file extension
        if not output_file.endswith('.nii.gz'):
            continue
        
        # Load the data from the input files
        data_true = nib.load(os.path.join(path_to_folder, f'{PATH_TO_TRUE_SEGMENTATION}{case}.nii.gz')).get_fdata()
        data_output = nib.load(os.path.join(path_to_folder, output_file)).get_fdata()
        
        # Find the connected components in each matrix
        labeled_true, num_features_true = ndi.label(data_true)
        labeled_output, num_features_output = ndi.label(data_output)

        total_features_true += num_features_true
        total_features_output += num_features_output
        
        # Print the number of connected components in each matrix
        print(f'num of connectivity components in true segmentation case #{case}: {num_features_true}')
        print(f'num of connectivity components in output segmentation case #{case}: {num_features_output}')
        
        # Count the number of intersections between connected components
        intersections = 0
        for i in range(1, num_features_true + 1):
            for j in range(1, num_features_output + 1):
                if np.any(np.logical_and(labeled_true == i, labeled_output == j)):
                    intersections += 1
        total_intersections += intersections
        
        # Print the number of intersections for this pair of matrices
        print(f'num of intersections in case#{case}: {intersections}')
    print(f'Precision TP/(TP + FP): {total_intersections/(total_features_output)}')
    print(f'Recall: {(total_intersections/total_features_true)*100}')




# img = nib.load('/Users/elilevinkopf/Downloads/res.nii.gz').get_fdata()

# m = img.get_fdata()
# m_closed = area_closing(m, area_threshold=1024, connectivity=2)
# nib.save(nib.Nifti1Image(m_closed, None), '/Users/elilevinkopf/Documents/Ex23A/FinalProject/test_output/task 509_sinus_bone/tmp044.nii.gz')


# count_unique_values('/Users/elilevinkopf/Documents/Ex23A/FinalProject/test_output/task 502_penetration')
# count_intersections('/Users/elilevinkopf/Documents/Ex23A/FinalProject/test_output/task 508_all_classes_one_label')
# defaultSegmentation('/Users/elilevinkopf/Documents/Ex23A/FinalProject/sinus_bone_segmantation')



def process_segmentation(segmentation_3D_path):
    """
    This function processes a 3D segmentation matrix by projecting it to a 2D matrix, filling the holes in the 2D matrix,
    and returning a matrix with the holes marked with ones.
    
    :param segmentation_path: Path to the 3D segmentation matrix
    :return: Tuple containing the filled regions and the rounded centers of mass
    """
    segmentation_3D = nib.load(segmentation_3D_path).get_fdata()
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
    inverse_2D_segmentation = np.array(inverse_2D_segmentation)
    rows, cols = inverse_2D_segmentation.shape
    edges = []
    for i in range(rows):
        for j in range(cols):
            if inverse_2D_segmentation[i, j] == 1:
                if (i == 0 or inverse_2D_segmentation[i-1, j] == 0) or (j == 0 or inverse_2D_segmentation[i, j-1] == 0) or\
                    (i == rows-1 or inverse_2D_segmentation[i+1, j] == 0) or (j == cols-1 or inverse_2D_segmentation[i, j+1] == 0):
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


def find_dental_anomalies_edges1(inverse_2D_segmentation):
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



def detect_dental_anomalies(edges, original_segmentation):
    result_matrix = np.zeros((original_segmentation.shape[0], original_segmentation.shape[1]))
    for x, y in edges:
        result_matrix[x, y] = 1
    labels_2D_matrix, num_labels = ndi.label(result_matrix)
    result_3D_matrix = np.zeros_like(original_segmentation)

    for i in range(1, num_labels + 1):
        z_coordinate = 0
        num_of_z = 0
        for x, y in edges:
            for z in range(original_segmentation.shape[2]):
                if labels_2D_matrix[x, y] == i and original_segmentation[x, y, z] == 1:
                    z_coordinate += z
                    num_of_z += 1
        z_coordinate /= num_of_z
        for x,y in edges:
            if labels_2D_matrix[x, y] == i:
                result_3D_matrix[x, y, int(z_coordinate)] = 1

    labels, num_labels = ndi.label(result_3D_matrix)
    label_sizes = np.bincount(labels.ravel())

    centers_of_mass = []
    # Find the center of mass for each label
    for i in range(1, num_labels + 1):
        if label_sizes[i] < 100: # drop elements that are smaller than 100
            continue
        center_of_mass = ndi.center_of_mass(labels == i)
        centers_of_mass.append(center_of_mass)

    # TODO: delete this line
    nib.save(nib.Nifti1Image(result_3D_matrix, None), '/Users/elilevinkopf/Documents/Ex23A/FinalProject/test_output/task 509_sinus_bone/res.nii.gz')

    return np.rint(centers_of_mass).astype(int)


def detect_dental_anomalies1(points, original_segmentation):
    """
    Find the centers of mass of dental anomalies points in a 3D segmentation.
    
    :param points: List of tuples representing the x and y coordinates of points in the 2D plane
    :param original_segmentation: 3D numpy array representing the original segmentation (segmentation of the
    sinus bone with holes in the dental anomalies points).
    :return: List of tuples representing the centers of mass of the dental anomalies points in the 3D segmentation (x, y, z).
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
    
    # TODO: delete this line
    nib.save(nib.Nifti1Image(result_3D_matrix, None), '/Users/elilevinkopf/Documents/Ex23A/FinalProject/test_output/task 509_sinus_bone/res.nii.gz')
    
    return np.rint(centers_of_mass).astype(int)



inverse_2D_segmentation, segmentation_3D = process_segmentation('/Users/elilevinkopf/Documents/Ex23A/FinalProject/test_output/task 509_sinus_bone/sinus_bone_044.nii.gz')
edges = find_dental_anomalies_edges1(inverse_2D_segmentation)
print(detect_dental_anomalies1(edges, segmentation_3D))


# def find_appropriate_points(points, matrix):
#     result = []
#     result_matrix = np.zeros_like(matrix)
#     z_coordinate = 0
#     num_of_z = 0
#     for x, y in points:
#         for z in range(matrix.shape[2]):
#             if matrix[x, y, z] == 1:
#                 z_coordinate += z
#                 num_of_z += 1
#                 # result.append((x, y, z))
#                 # result_matrix[x, y, z] = 1
#                 # result_matrix[x+1, y, z] = 1
#                 # result_matrix[x, y+1, z] = 1
#                 # result_matrix[x-1, y, z] = 1
#                 # result_matrix[x, y-1, z] = 1
#                 # result_matrix[x+2, y, z] = 1
#                 # result_matrix[x, y+2, z] = 1
#                 # result_matrix[x-2, y, z] = 1
#                 # result_matrix[x, y-2, z] = 1
#     z_coordinate /= num_of_z
#     for x,y in points:
#         result_matrix[x, y, int(z_coordinate)] = 1
#         # result_matrix[x+1, y, int(z_coordinate)] = 1
#         # result_matrix[x+2, y, int(z_coordinate)] = 1
#         # result_matrix[x+3, y, int(z_coordinate)] = 1
#         # result_matrix[x-1, y, int(z_coordinate)] = 1
#         # result_matrix[x-2, y, int(z_coordinate)] = 1
#         # result_matrix[x-3, y, int(z_coordinate)] = 1
#         # result_matrix[x, y+1, int(z_coordinate)] = 1
#         # result_matrix[x, y+2, int(z_coordinate)] = 1
#         # result_matrix[x, y+3, int(z_coordinate)] = 1
#         # result_matrix[x, y-1, int(z_coordinate)] = 1
#         # result_matrix[x, y-2, int(z_coordinate)] = 1
#         # result_matrix[x, y-3, int(z_coordinate)] = 1


#     labels, num_labels = ndimage.label(result_matrix)
#     print(num_labels)

# def find_edges_no_loops(matrix):
#     """
#     Find the edges of 'islands' of ones in a 2D binary matrix.
    
#     :param matrix: 2D list of integers representing the binary matrix
#     :return: List of tuples representing the coordinates of the edges of 'islands' of ones
#     """
#     # Convert the input matrix to a numpy array
#     matrix = np.array(matrix)
    
#     # Pad the matrix with zeros to avoid index out of bounds errors
#     padded_matrix = np.pad(matrix, ((1, 1), (1, 1)), mode='constant')
    
#     # Find the row and column indices of ones in the original matrix
#     row_indices, col_indices = np.where(padded_matrix[1:-1, 1:-1] == 1)
    
#     # Adjust the indices to account for the padding
#     row_indices += 1
#     col_indices += 1
    
#     # Find the values of the cells above, below, to the left, and to the right of each one
#     top = padded_matrix[row_indices - 1, col_indices] == 0
#     bottom = padded_matrix[row_indices + 1, col_indices] == 0
#     left = padded_matrix[row_indices, col_indices - 1] == 0
#     right = padded_matrix[row_indices, col_indices + 1] == 0
    
#     # Find the ones that are adjacent to a zero (i.e. on the edge of an 'island')
#     edges_mask = top | bottom | left | right
    
#     # Get the coordinates of the edges
#     edges = list(zip(row_indices[edges_mask] - 1, col_indices[edges_mask] - 1))
    
#     # Plot the edges
#     x = [edge[1] for edge in edges]
#     y = [edge[0] for edge in edges]
#     plt.scatter(x, y, s=0.1 ,c='red')
#     plt.show()
    
#     return edges


# def find_edges_no_loops1(matrix):
#     """
#     Find the edges of 'islands' of ones in a 2D binary matrix using numpy operations without explicit loops.

#     :param matrix: 2D list of integers representing the binary matrix
#     :return: List of tuples representing the coordinates of the edges of 'islands' of ones
#     """
#     # Convert the input matrix to a numpy array
#     matrix = np.array(matrix)

#     # Pad the matrix with zeros to avoid index out of bounds errors
#     padded_matrix = np.pad(matrix, ((1, 1), (1, 1)), mode='constant')

#     # Find the row and column indices of zeros in the original matrix
#     row_indices, col_indices = np.where(padded_matrix[1:-1, 1:-1] == 0)

#     # Adjust the indices to account for the padding
#     row_indices += 1
#     col_indices += 1

#     # Find the values of the cells above, below, to the left, and to the right of each zero
#     top = padded_matrix[row_indices - 1, col_indices] == 1
#     bottom = padded_matrix[row_indices + 1, col_indices] == 1
#     left = padded_matrix[row_indices, col_indices - 1] == 1
#     right = padded_matrix[row_indices, col_indices + 1] == 1

#     # Find the zeros that are adjacent to a one (i.e. on the edge of an 'island')
#     edges_mask = top | bottom | left | right

#     # Get the coordinates of the edges
#     edges = list(zip(row_indices[edges_mask] - 1, col_indices[edges_mask] - 1))

#     # Plot the edges
#     x = [edge[1] for edge in edges]
#     y = [edge[0] for edge in edges]
#     plt.scatter(x, y, s=0.1 ,c='red')
#     plt.show()

#     return edges
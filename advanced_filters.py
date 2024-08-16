from skimage.morphology import erosion, dilation, opening, closing, disk
import numpy as np
import cv2
import ast
import pandas as pd
from skimage.measure import label, regionprops
from skimage.morphology import reconstruction, disk
from skimage.morphology import binary_dilation
from skimage import filters, morphology
from skimage.segmentation import active_contour
from skimage import restoration
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from basic_filtering import appliquer_gaussian_filter_au_df, appliquer_savgol_filter_au_df, appliquer_median_filter_au_df, appliquer_bilateral_filter_au_df

""" **1. Morphological Filtering** """

# Function to apply the morphological filter to each image and create a new DataFrame
def appliquer_morphological_filter_au_df(df, operation='opening', selem_size=3):
    IR_FRAME_ROWS = 24
    IR_FRAME_COLUMNS = 32

    # Create a copy of the DataFrame
    df_morph = df.copy()

    # Select the morphological operation to apply
    if operation == 'erosion':
        morph_operation = erosion
    elif operation == 'dilation':
        morph_operation = dilation
    elif operation == 'opening':
        morph_operation = opening
    elif operation == 'closing':
        morph_operation = closing
    else:
        raise ValueError("Unrecognized operation. Choose from 'erosion', 'dilation', 'opening', 'closing'.")

    # Create a structuring element of the given size
    footprint = disk(selem_size)

    # Apply the morphological filter to each image in the 'image array' column
    def appliquer_filter(row):
        image_array = row['image array']

        if isinstance(image_array, str):
            image_array = ast.literal_eval(image_array)

        image_matrix = np.array(image_array).reshape(IR_FRAME_ROWS, IR_FRAME_COLUMNS)
        smoothed_image = morph_operation(image_matrix, footprint=footprint)
        return smoothed_image.flatten()  # Return the flattened image to match the initial format

    # Apply the function to each row of the DataFrame
    df_morph['image array'] = df_morph.apply(appliquer_filter, axis=1)

    return df_morph


""" 
# **2. Global and Local Histogram Equalization**
Histogram Equalization is a technique used to improve the contrast in images by redistributing the intensity values so that the histogram of the output image is roughly uniform. This can be particularly useful for enhancing images with poor contrast.
"""

# Function to apply histogram equalization to each image and create a new DataFrame
def apply_histogram_equalization_to_df(df):
    IR_FRAME_ROWS = 24
    IR_FRAME_COLUMNS = 32

    # Create a copy of the DataFrame
    df_eq = df.copy()

    # Apply histogram equalization to each image in the 'image array' column
    def apply_equalization(row):
        image_array = row['image array']

        # Convert the string to a numpy array, if necessary
        if isinstance(image_array, str):
            image_array = ast.literal_eval(image_array)

        # Reshape the array to 2D
        image_matrix = np.array(image_array).reshape(IR_FRAME_ROWS, IR_FRAME_COLUMNS)

        # Apply histogram equalization
        eq_image = cv2.equalizeHist(image_matrix.astype(np.uint8))

        return eq_image.flatten()  # Return the flattened image to match the original format

    # Apply the function to each row in the DataFrame
    df_eq['image array'] = df_eq.apply(apply_equalization, axis=1)

    return df_eq

# Function to apply CLAHE (Histogram Equalization) to each image and create a new DataFrame
def apply_clahe_histogram_equalization_to_df(df):
    IR_FRAME_ROWS = 24
    IR_FRAME_COLUMNS = 32

    # Create a copy of the DataFrame
    df_eq = df.copy()

    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Apply CLAHE to each image in the 'image array' column
    def apply_equalization(row):
        image_array = row['image array']

        # Convert the string to a numpy array, if necessary
        if isinstance(image_array, str):
            image_array = ast.literal_eval(image_array)

        # Reshape the array to 2D
        image_matrix = np.array(image_array).reshape(IR_FRAME_ROWS, IR_FRAME_COLUMNS)

        # Apply CLAHE (Histogram Equalization)
        eq_image = clahe.apply(image_matrix.astype(np.uint8))

        # Optionally normalize the image after CLAHE
        eq_image = cv2.normalize(eq_image, None, 0, 255, cv2.NORM_MINMAX)

        return eq_image.flatten()  # Return the flattened image to match the original format

    # Apply the function to each row in the DataFrame
    df_eq['image array'] = df_eq.apply(apply_equalization, axis=1)

    return df_eq


""" **3. Connected Component Analysis (CCA)**"""

# Function to apply Connected Component Analysis to each image and create a new DataFrame
def apply_connected_component_analysis_to_df(df):
    IR_FRAME_ROWS = 24
    IR_FRAME_COLUMNS = 32

    # Create a copy of the DataFrame
    df_cca = df.copy()

    # Apply Connected Component Analysis to each image in the 'image array' column
    def apply_cca(row):
        image_array = row['image array']

        # Convert the string to a numpy array, if necessary
        if isinstance(image_array, str):
            image_array = ast.literal_eval(image_array)

        # Reshape the array to 2D
        image_matrix = np.array(image_array).reshape(IR_FRAME_ROWS, IR_FRAME_COLUMNS)

        # Apply thresholding to convert to binary image
        binary_image = image_matrix > 0

        # Apply Connected Component Analysis
        labeled_image = label(binary_image)

        # Optionally, extract properties of the regions
        properties = regionprops(labeled_image)

        # Create a feature vector based on region properties (e.g., area of each region)
        feature_vector = np.array([region.area for region in properties])

        # Ensure feature vector has consistent length with the flattened image
        if len(feature_vector) < IR_FRAME_ROWS * IR_FRAME_COLUMNS:
            feature_vector = np.pad(feature_vector, (0, IR_FRAME_ROWS * IR_FRAME_COLUMNS - len(feature_vector)), 'constant')

        return feature_vector

    # Apply the function to each row in the DataFrame
    df_cca['image array'] = df_cca.apply(apply_cca, axis=1)

    return df_cca


"""# **4. Morphological Reconstruction**"""

# Function to apply morphological reconstruction to each image and create a new DataFrame
def apply_morphological_reconstruction_to_df(df):
    IR_FRAME_ROWS = 24
    IR_FRAME_COLUMNS = 32

    # Create a copy of the DataFrame
    df_reconstructed = df.copy()

    # Apply morphological reconstruction to each image in the 'image array' column
    def apply_reconstruction(row):
        image_array = row['image array']

        # Convert the string to a numpy array if necessary
        if isinstance(image_array, str):
            image_array = ast.literal_eval(image_array)

        # Transform the image into a 2D matrix
        image_matrix = np.array(image_array).reshape(IR_FRAME_ROWS, IR_FRAME_COLUMNS)

        # Apply a simple threshold operation to create a binary image
        binary_image = image_matrix > 0

        # Create a marker for reconstruction. Here, we use a dilation of the binary image.
        seed = binary_dilation(binary_image, disk(1))

        # Apply the morphological reconstruction
        reconstructed_image = reconstruction(seed, binary_image, method='erosion')

        return reconstructed_image.flatten().astype(int)  # Return the flattened image to match the initial format

    # Apply the function to each row of the DataFrame
    df_reconstructed['image array'] = df_reconstructed.apply(apply_reconstruction, axis=1)

    return df_reconstructed


"""# **5. Active Contour Model**"""

# Function to apply the active contour model
def apply_active_contour_model_to_df(df, alpha=0.015, beta=0.015):
    IR_FRAME_ROWS = 24
    IR_FRAME_COLUMNS = 32

    # Create a copy of the DataFrame
    df_active_contour = df.copy()

    # Apply the active contour model to each image in the 'image array' column
    def apply_active_contour(row):
        image_array = row['image array']

        # Convert the string to a numpy array if necessary
        if isinstance(image_array, str):
            image_array = ast.literal_eval(image_array)

        # Transform the image into a 2D matrix
        image_matrix = np.array(image_array).reshape(IR_FRAME_ROWS, IR_FRAME_COLUMNS)

        # Apply a threshold to binarize the image
        threshold = filters.threshold_otsu(image_matrix)
        binary_image = image_matrix > threshold

        # Define an initial contour (here, a circle in the center of the image)
        s = np.linspace(0, 2 * np.pi, 400)
        x = (IR_FRAME_COLUMNS / 2 + (IR_FRAME_COLUMNS / 4) * np.cos(s)).astype(int)
        y = (IR_FRAME_ROWS / 2 + (IR_FRAME_ROWS / 4) * np.sin(s)).astype(int)
        init = np.array([x, y])

        # Apply the active contour model
        snake = active_contour(binary_image, init, alpha=alpha, beta=beta, gamma=0.01)

        # Create an output image with the active contour
        active_contour_image = np.zeros_like(binary_image)
        rr, cc = snake.astype(int)
        active_contour_image[rr, cc] = 1

        return active_contour_image.astype(np.uint8).flatten()  # Ensure the image is in uint8

    # Apply the function to each row of the DataFrame
    df_active_contour['image array'] = df_active_contour.apply(apply_active_contour, axis=1)

    return df_active_contour

# Function to display images
def display_images_in_row2(df):
    IR_FRAME_ROWS = 24
    IR_FRAME_COLUMNS = 32
    # Number of images to display
    num_images = len(df)
    # Create a figure with one subplot per image
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 4, 4))
    # If only one image, 'axes' won't be a list, so convert it to a list
    if num_images == 1:
        axes = [axes]
    for ax, index in zip(axes, df.index):
        image_array = df.iloc[index]['image array']
        # If 'image array' is a string, convert it to a list of integers
        if isinstance(image_array, str):
            image_array = ast.literal_eval(image_array)
        # Convert the list into a 2D matrix
        image_matrix = np.array(image_array).reshape(IR_FRAME_ROWS, IR_FRAME_COLUMNS)
        # Display the image on the current axis
        ax.imshow(image_matrix, cmap='hot')
        ax.set_title(f"Time: {df.iloc[index]['time']}\n"
                     f"peoplecount_rgb: {df.iloc[index]['peoplecount_rgb']}, "
                     f"peoplecount_ir_est: {df.iloc[index]['peoplecount_ir_est']}")
        ax.axis('off')  # Hide the axis
    plt.tight_layout()
    plt.show()


"""# **6. Total Variation (TV) Denoising**"""

# Function to apply Total Variation Denoising to each image and create a new DataFrame
def apply_tv_denoising_to_df(df, weight=0.1):
    IR_FRAME_ROWS = 24
    IR_FRAME_COLUMNS = 32

    # Create a copy of the DataFrame
    df_tv_denoised = df.copy()

    # Apply TV denoising to each image in the 'image array' column
    def apply_denoising(row):
        image_array = row['image array']

        # Convert the string to a numpy array, if necessary
        if isinstance(image_array, str):
            image_array = ast.literal_eval(image_array)

        # Reshape the array to 2D
        image_matrix = np.array(image_array).reshape(IR_FRAME_ROWS, IR_FRAME_COLUMNS)

        # Apply Total Variation Denoising
        denoised_image = restoration.denoise_tv_chambolle(image_matrix, weight=weight)

        return denoised_image.flatten()  # Return the flattened image to match the original format

    # Apply the function to each row in the DataFrame
    df_tv_denoised['image array'] = df_tv_denoised.apply(apply_denoising, axis=1)

    return df_tv_denoised


"""
# CONCLUSION: Combining the successful techniques**

    * Gaussian filter with sigma = 0.5
    * Savgol filter with (window_length=5, polyorder=2) or (window_length=5, polyorder=3) or (window_length=9, polyorder=4)
    * Morphological filter using the closing operation with selem_sizes of 1 or 2
"""

# Function to apply combined filters to each image and create a new DataFrame
def appliquer_combined_filters_au_df(df):

    # Create a copy of the DataFrame
    df_filtered = df.copy()

    #df_filtered = appliquer_gaussian_filter_au_df(df_filtered, sigma=0.9)
    df_filtered = appliquer_savgol_filter_au_df(df_filtered, window_length=5, polyorder=2)
    df_filtered = appliquer_morphological_filter_au_df(df_filtered, operation='closing', selem_size=1)

    return df_filtered
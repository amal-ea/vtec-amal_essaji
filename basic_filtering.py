""" **Definition of basic filtering functions: Gaussian, Bilateral, Savgol, Median**
"""

import numpy as np
import pandas as pd
import ast
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
from scipy.signal import savgol_filter
from skimage.restoration import denoise_bilateral


# Function to apply the Gaussian filter to each image and create a new DataFrame
def appliquer_gaussian_filter_au_df(df, sigma=0.9):
    IR_FRAME_ROWS = 24
    IR_FRAME_COLUMNS = 32

    # Create a copy of the DataFrame
    df_gauss = df.copy()

    # Apply the Gaussian filter to each image in the 'image array' column
    def appliquer_filter(row):
        image_array = row['image array']

        if isinstance(image_array, str):
            image_array = ast.literal_eval(image_array)

        image_matrix = np.array(image_array).reshape(IR_FRAME_ROWS, IR_FRAME_COLUMNS)
        smoothed_image = gaussian_filter(image_matrix, sigma=sigma)
        return smoothed_image.flatten()  # Return the flattened image to match the initial format

    # Apply the function to each row of the DataFrame
    df_gauss['image array'] = df_gauss.apply(appliquer_filter, axis=1)

    return df_gauss

# Function to apply the Savitzky-Golay filter to each image and create a new DataFrame
def appliquer_savgol_filter_au_df(df, window_length=5, polyorder=2):
    IR_FRAME_ROWS = 24
    IR_FRAME_COLUMNS = 32

    # Create a copy of the DataFrame
    df_savgol = df.copy()

    # Apply the Savitzky-Golay filter to each image in the 'image array' column
    def appliquer_filter(row):
        image_array = row['image array']

        if isinstance(image_array, str):
            image_array = ast.literal_eval(image_array)

        image_matrix = np.array(image_array).reshape(IR_FRAME_ROWS, IR_FRAME_COLUMNS)
        smoothed_image = savgol_filter(image_matrix, window_length=window_length, polyorder=polyorder, mode='nearest')
        return smoothed_image.flatten()  # Return the flattened image to match the initial format

    # Apply the function to each row of the DataFrame
    df_savgol['image array'] = df_savgol.apply(appliquer_filter, axis=1)

    return df_savgol

# Function to apply the median filter to each image and create a new DataFrame
def appliquer_median_filter_au_df(df, size=3):
    IR_FRAME_ROWS = 24
    IR_FRAME_COLUMNS = 32

    # Create a copy of the DataFrame
    df_median = df.copy()

    # Apply the median filter to each image in the 'image array' column
    def appliquer_filter(row):
        image_array = row['image array']

        if isinstance(image_array, str):
            image_array = ast.literal_eval(image_array)

        image_matrix = np.array(image_array).reshape(IR_FRAME_ROWS, IR_FRAME_COLUMNS)
        smoothed_image = median_filter(image_matrix, size=size)
        return smoothed_image.flatten()  # Return the flattened image to match the initial format

    # Apply the function to each row of the DataFrame
    df_median['image array'] = df_median.apply(appliquer_filter, axis=1)

    return df_median

# Function to apply the bilateral filter to each image and create a new DataFrame
def appliquer_bilateral_filter_au_df(df, sigma_color=0.05, sigma_spatial=15):
    IR_FRAME_ROWS = 24
    IR_FRAME_COLUMNS = 32

    # Create a copy of the DataFrame
    df_bilateral = df.copy()

    # Apply the bilateral filter to each image in the 'image array' column
    def appliquer_filter(row):
        image_array = row['image array']

        if isinstance(image_array, str):
            image_array = ast.literal_eval(image_array)

        image_matrix = np.array(image_array).reshape(IR_FRAME_ROWS, IR_FRAME_COLUMNS)
        smoothed_image = denoise_bilateral(image_matrix, sigma_color=sigma_color, sigma_spatial=sigma_spatial)
        return smoothed_image.flatten()  # Return the flattened image to match the initial format

    # Apply the function to each row of the DataFrame
    df_bilateral['image array'] = df_bilateral.apply(appliquer_filter, axis=1)

    return df_bilateral

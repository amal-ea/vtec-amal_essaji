# Import the libraries
import pandas as pd
import numpy as np
import cv2
import pandas as pd
import ast
from scipy.stats import pearsonr

from advanced_filters import appliquer_morphological_filter_au_df, apply_histogram_equalization_to_df, apply_clahe_histogram_equalization_to_df, apply_connected_component_analysis_to_df, apply_morphological_reconstruction_to_df, apply_active_contour_model_to_df, display_images_in_row2, apply_tv_denoising_to_df, appliquer_combined_filters_au_df
from basic_filtering import appliquer_gaussian_filter_au_df, appliquer_savgol_filter_au_df, appliquer_median_filter_au_df, appliquer_bilateral_filter_au_df
from xin_algorithm import SO_evaluation, test_estimate_pc_from_frame_blob_method, display_images_in_row, test_and_calculate_metrics


# Load the datasets
df1 = pd.read_csv('/Users/amal/Desktop/VTEC/github/VTEC_A_20230819.csv', delimiter=';')
df2 = pd.read_csv('/Users/amal/Desktop/VTEC/github/VTEC_A_20230911.csv', delimiter=';')
df3 = pd.read_csv('/Users/amal/Desktop/VTEC/github/VTEC_A_20230914.csv', delimiter=';')
df4 = pd.read_csv('/Users/amal/Desktop/VTEC/github/VTEC_A_20230915.csv', delimiter=';')
df5 = pd.read_csv('/Users/amal/Desktop/VTEC/github/VTEC_A_20230919.csv', delimiter=';')
df6 = pd.read_csv('/Users/amal/Desktop/VTEC/github/VTEC_A_20231218.csv', delimiter=';')
df7 = pd.read_csv('/Users/amal/Desktop/VTEC/github/VTEC_A_20231219.csv', delimiter=';')
df8 = pd.read_csv('/Users/amal/Desktop/VTEC/github/VTEC_A_20231220.csv', delimiter=';')
df9 = pd.read_csv('/Users/amal/Desktop/VTEC/github/VTEC_A_20231221.csv', delimiter=';')
df10 = pd.read_csv('/Users/amal/Desktop/VTEC/github/VTEC_A_20231222.csv', delimiter=';')
df11 = pd.read_csv('/Users/amal/Desktop/VTEC/github/VTEC_A_20240125.csv', delimiter=';')
df12 = pd.read_csv('/Users/amal/Desktop/VTEC/github/VTEC_A_20240126.csv', delimiter=';')
df13 = pd.read_csv('/Users/amal/Desktop/VTEC/github/VTEC_A_20240205.csv', delimiter=';')
df14 = pd.read_csv('/Users/amal/Desktop/VTEC/github/VTEC_A_20240206.csv', delimiter=';')


dfs0 = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14]
dfs1 = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14]
dfs2 = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14]
dfs3 = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14]
dfs4 = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14]


"""### **New sub dataset**"""
# Filter rows from df4, df8, and df10 with specific 'time' values
df4_filtered = df4[df4['time'].isin(['2023-09-15 17:33:36+02:00', '2023-09-15 17:35:38+02:00', '2023-09-15 10:19:36+02:00'])]
df10_filtered = df10[df10['time'] == '2023-12-22 08:06:36+01:00']
df8_filtered = df8[df8['time'] == '2023-12-20 17:42:36+01:00']
# Concatenate the filtered DataFrames
filtered_dataframes = [df4_filtered, df10_filtered, df8_filtered]
df_dynamique = pd.concat(filtered_dataframes, ignore_index=True)
# Display the images from the df_dynamique DataFrame
#display_images_in_row(df_dynamique)


"""### **Gaussian filter**"""
df_gauss1 = appliquer_gaussian_filter_au_df(df_dynamique, sigma=0.5)
df_gauss2 = appliquer_gaussian_filter_au_df(df_dynamique, sigma=0.7)
df_gauss3 = appliquer_gaussian_filter_au_df(df_dynamique, sigma=0.9)
df_gauss4 = appliquer_gaussian_filter_au_df(df_dynamique, sigma=0.4)
df_gauss5 = appliquer_gaussian_filter_au_df(df_dynamique, sigma=0.3)
# Execute the test function for multiple DataFrames
dfs_gauss = [df_gauss1, df_gauss2, df_gauss3, df_gauss4, df_gauss5, df_dynamique]
results_gauss = [test_estimate_pc_from_frame_blob_method(df) for df in dfs_gauss]
# Display the results
#for i, result in enumerate(results_gauss, 1):
    #print(f"Results {i}: {result}")
"""The best result was obtained with sigma = 0.5, so we will choose this value for the Gaussian filter."""
df_gauss = appliquer_gaussian_filter_au_df(df_dynamique, sigma=0.5)


"""### **Median filter**"""
df_median1 = appliquer_median_filter_au_df(df_dynamique, size=3)
df_median2 = appliquer_median_filter_au_df(df_dynamique, size=5)
df_median3 = appliquer_median_filter_au_df(df_dynamique, size=7)
df_median4 = appliquer_median_filter_au_df(df_dynamique, size=1)
# Execute the test function for multiple DataFrames
dfs_median = [df_median1, df_median2, df_median3, df_median4, df_dynamique]
results_median = [test_estimate_pc_from_frame_blob_method(df) for df in dfs_median]
# Display the results
#for i, result in enumerate(results_median, 1):
      #print(f"Results {i}: {result}")
"""The results are unsatisfactory, so the median filter will not be used. """


"""### **Savgol filter**"""
# Define the ranges of values for window_length and polyorder
window_lengths_to_test = [3, 5, 7, 9, 11]
polyorders_to_test = [1, 2, 3, 4]
# The target list of results to compare
target_result = [1, 2, 2, 1, 1]
# Loop to test each combination of window_length and polyorder
for window_length in window_lengths_to_test:
    for polyorder in polyorders_to_test:
        # Ensure that window_length is valid (odd and greater than polyorder)
        if window_length > polyorder and window_length % 2 == 1:
            # Apply the Savitzky-Golay filter with the current parameters
            df_savgol = appliquer_savgol_filter_au_df(df_dynamique, window_length=window_length, polyorder=polyorder)
            # Calculate the result using the method test_estimate_pc_from_frame_blob_method
            result = test_estimate_pc_from_frame_blob_method(df_savgol)
            # Print the result only if it matches the target_result
            #if result == target_result:
              #print(f"Matching result for window_length={window_length}, polyorder={polyorder}: {result}")
"""The three parameter pairs yield the correct value:
*   (window_length=5, polyorder=2)
*   (window_length=5, polyorder=3)
*   (window_length=9, polyorder=4) """


"""### **Bilateral filter**"""
# Define the ranges of values for sigma_color and sigma_spatial
sigma_colors_to_test = [0.01, 0.05, 0.1, 0.2]
sigma_spatials_to_test = [5, 10, 15, 20]
for sigma_color in sigma_colors_to_test:
    for sigma_spatial in sigma_spatials_to_test:
        # Apply the bilateral filter with the current parameters
        df_bilateral = appliquer_bilateral_filter_au_df(df_dynamique, sigma_color=sigma_color, sigma_spatial=sigma_spatial)
        # Calculate the result using the method test_estimate_pc_from_frame_blob_method
        result = test_estimate_pc_from_frame_blob_method(df_bilateral)
        # Print the result only if it matches target_result
        #print(f"Matching result for sigma_color={sigma_color}, sigma_spatial={sigma_spatial}: {result}")
"""The results are unsatisfactory, so the bilateral filter will not be used."""


"""# **Evaluation and metrics**"""
for i, df in enumerate(dfs4, start=1):
        df= appliquer_gaussian_filter_au_df(df, sigma=0.5)

predictions = test_and_calculate_metrics(dfs0)
print(len(predictions))
SO_evaluation(dfs0)
"""The results obtained across the entire dataset show no improvement. It is only effective on the data from `df_dynamique`. """


"""# **1. Morphological Filtering**"""
# Parameters to test
operations = ['erosion', 'dilation', 'opening', 'closing']
selem_sizes = [1, 2, 3, 4, 5]
# Apply the filter and evaluate
for operation in operations:
    for selem_size in selem_sizes:
        #print(f"Testing operation: {operation}, selem_size: {selem_size}")
        df_morph = appliquer_morphological_filter_au_df(df_dynamique, operation=operation, selem_size=selem_size)
        result = test_estimate_pc_from_frame_blob_method(df_morph)
        #print(f"Result: {result}")
"""Best result [1, 2, 1, 1, 1] obtained using the closing operation with selem_sizes of 1 or 2 
Morphological filtering is an image processing technique used to modify shapes and structures within an image. It includes basic operations like **erosion** (shrinking objects) and **dilation** (expanding objects), which can be combined into more complex operations like **opening** (erosion followed by dilation) and **closing** (dilation followed by erosion). The impact of these operations depends on the size of the **structuring element (selem)**, with smaller sizes making finer adjustments and larger sizes applying more drastic changes.
In this case, **closing** with a small `selem_size` of 1 or 2 produced the best results by effectively smoothing the image without losing important details."""


"""# **2. Global and Local Histogram Equalization**
Histogram Equalization is a technique used to improve the contrast in images by redistributing the intensity values so that the histogram of the output image is roughly uniform. This can be particularly useful for enhancing images with poor contrast."""
# Apply the histogram equalization and evaluate
df_eq = apply_histogram_equalization_to_df(df_dynamique)
# Execute the test function for the equalized DataFrame
result = test_estimate_pc_from_frame_blob_method(df_eq)
#print(f"Results after histogram equalization: {result}")
# Apply the histogram equalization CLAHE and evaluate
df_eq2 = apply_clahe_histogram_equalization_to_df(df_dynamique)
# Execute the test function for the equalized DataFrame
result = test_estimate_pc_from_frame_blob_method(df_eq2)
#print(f"Results after histogram equalization: {result}")
#display_images_in_row(df_eq)
#display_images_in_row(df_eq2)

"""Global vs. Local Histogram Equalization:
*   Histogram Equalization (HE) redistributes pixel values across the entire image, enhancing overall contrast by emphasizing differences in brightness between regions.
*   CLAHE (Contrast Limited Adaptive Histogram Equalization), on the other hand, applies equalization locally to different regions of the image. While this can improve contrast in low-contrast images, it may overemphasize local details (such as noise) and reduce the clarity of global contours.
In this case, the original image already has well-defined contours, so CLAHE have introduced too much local detail, making the contours less distinct due to the increased local noise.
**However, the results remain unsatisfactory in both cases, so neither method will be used.** """


"""# **3. Connected Component Analysis (CCA)**"""
# Apply the Connected Component Analysis and evaluate
df_cca = apply_connected_component_analysis_to_df(df_dynamique)
# Execute the test function for the DataFrame with CCA
result = test_estimate_pc_from_frame_blob_method(df_cca)
#print(f"Results after Connected Component Analysis: {result}")


"""# **4. Morphological Reconstruction**"""
# Apply morphological reconstruction and evaluate
df_reconstructed = apply_morphological_reconstruction_to_df(df_dynamique)
# Run the test function on the DataFrame with morphological reconstruction
result = test_estimate_pc_from_frame_blob_method(df_reconstructed)
#print(f"Results after Morphological Reconstruction: {result}")
#display_images_in_row(df_reconstructed)


"""# **5. Active Contour Model**"""
# Parameters to test
alpha_values = [0.01, 0.015, 0.02, 0.05]
beta_values = [0.01, 0.015, 0.02, 0.05]
# Test different combinations of parameters
results = {}
for alpha in alpha_values:
    for beta in beta_values:
        # Print the current alpha and beta values being tested
        # print(f"Testing alpha={alpha}, beta={beta}")
        # Apply the active contour model with the specified parameters
        df_active_contour = apply_active_contour_model_to_df(df_dynamique, alpha, beta)
        # Run the test function on the DataFrame with active contours
        result = test_estimate_pc_from_frame_blob_method(df_active_contour)
        results[(alpha, beta)] = result
        print(f"Results for alpha={alpha}, beta={beta}: {result}")
        # Use the existing function to display the images
        display_images_in_row2(df_active_contour)
# Display results of all tests
#print("All results:")
#for params, result in results.items():
#    print(f"Parameters: alpha={params[0]}, beta={params[1]} - Result: {result}")
"""The **Active Contour Model** is a technique used for edge detection in images. It adjusts a curve, or "active contour," to fit the boundaries of objects by minimizing an energy function that combines:
- **Internal Energy:** This penalizes irregularities in the contour and promotes smoothness. Parameters `alpha` and `beta` control this energy:
  - **`alpha` :** Regulates the smoothness of the contour. A higher `alpha` makes the contour smoother.
  - **`beta` :** Controls the rigidity of the contour with respect to its initial shape. A higher `beta` makes the contour more rigid.
- **External Energy:** Attracts the curve toward image features such as edges.
By adjusting `alpha` and `beta`, you can influence the flexibility and smoothness of the detected contour to better match objects in an image.
This solution is not retained because the contours obtained are too small and therefore not detected."""

"""# **6. Total Variation (TV) Denoising**"""
# Example usage of the denoising function
weight_values = [0.0001, 0.001, 10000]
for weight in weight_values:
    #print(f"Testing TV Denoising with weight: {weight}")
    df_tv_denoised = apply_tv_denoising_to_df(df_dynamique, weight=weight)
    # Execute the test function for the DataFrame with TV Denoising
    result = test_estimate_pc_from_frame_blob_method(df_tv_denoised)
    #print(f"Results after TV Denoising with weight {weight}: {result}")
    # Use the existing function to display the images
    #display_images_in_row(df_tv_denoised)
"""Total Variation (TV) Denoising is a noise reduction technique that preserves image edges by minimizing the total variation, which measures differences between neighboring pixel values. The weight parameter controls the denoising strength:
* Lower Weight: Less denoising, more detail preserved.
* Higher Weight: More denoising, but can blur details.
It balances noise reduction with edge preservation.
The result is unsatisfactory because the denoising is so strong that fine contours become invisible."""

"""
# CONCLUSION: Combining the successful techniques**
    * Gaussian filter with sigma = 0.5
    * Savgol filter with (window_length=5, polyorder=2) or (window_length=5, polyorder=3) or (window_length=9, polyorder=4)
    * Morphological filter using the closing operation with selem_sizes of 1 or 2
"""

# Filter rows from df4, df8, and df10 with specific 'time' values
df4_filtered = df4[df4['time'].isin(['2023-09-15 17:33:36+02:00', '2023-09-15 17:35:38+02:00', '2023-09-15 10:19:36+02:00'])]
df10_filtered = df10[df10['time'] == '2023-12-22 08:06:36+01:00']
df8_filtered = df8[df8['time'] == '2023-12-20 17:42:36+01:00']
# Concatenate the filtered DataFrames
filtered_dataframes = [df4_filtered, df10_filtered, df8_filtered]
df_dynamique = pd.concat(filtered_dataframes, ignore_index=True)
# Display images from the df_dynamique DataFrame
#display_images_in_row(df_dynamique)


# Usage:
df_savgol = appliquer_savgol_filter_au_df(df_dynamique, 9, 4)
df_filtered = appliquer_combined_filters_au_df(df_dynamique)
result = test_estimate_pc_from_frame_blob_method(df_savgol)
result2 = test_estimate_pc_from_frame_blob_method(df_filtered)
#display_images_in_row(df_filtered)

dfs_fin = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14]
for i, df in enumerate(dfs_fin, start=1):
        df= appliquer_combined_filters_au_df(df)

predictions = test_and_calculate_metrics(dfs_fin)
print(len(predictions))
SO_evaluation(dfs_fin)
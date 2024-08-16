"""# **Xin's Code**"""
# Import the pandas library
import pandas as pd
import numpy as np
import cv2
import pandas as pd
import ast
from scipy.stats import pearsonr
MODE = 1 #

# DEV_FLAG = True
# CICD_FLAG = False
C_DEV_FLAG = False

# Parameters
IR_FRAME_ROWS    = 24
IR_FRAME_COLUMNS = 32
ST_period = 11 #11 frames for temproal static-object removal
MAX_temprature = 3200 # if the pixel value higher than 3000, then it will be clipped
Similarity_score = 0.15 # if score lower than this value. then they are same 0.18
Interpolation_scale = 10 # interpolate image N time larger in both row and column
contour_corner = 2 # max movement of a detected contour to be defined as static
PIR_continue = 5 #if 4 frames get 0 pir then it is keep out
PIR_reset = 30*720 #3 days
cornerCorrectionMask = [[0.9 ,0.94,0.98,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.98,0.94,0.9 ,0.94,0.94,0.98,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.98,0.94,0.94,0.98,0.98,0.98,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.98,0.98,0.98,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.98,0.98,0.98,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.98,0.98,0.98,0.94,0.94,0.98,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.98,0.94,0.94,0.9 ,0.94,0.98,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.98,0.94,0.9 ]]

# Enable of different functions
CONTOUR_FILTERING = True
MAXIMUM_CLIPING = True
HEAT_AREA_SHRINKING = True
CHECK_AREA_SIZE = True
CHECK_PEAK_TEMP = True
CHECK_SHAPE = True

if MODE == 1:
    import matplotlib.pyplot as plt
import cv2
import pandas as pd
from skimage.filters import threshold_otsu, threshold_li
import math
import json
import datetime
import csv
import os
import numpy as np
from datetime import datetime

def global_initialization(PIR_continue):
    LM = np.zeros((24,32))+255
    CM = np.zeros((24,32))
    PIR_SL = np.zeros((PIR_continue))
    CR = 0
    return LM, CM, PIR_SL, CR

learningMask, confidenceMask, PIR_statusList, count_reset = global_initialization(PIR_continue)

def update_pir_list(status, statusList):
    if status <0:
        status =0
    statusList = np.delete(statusList, 0)
    statusList = np.append(statusList, [status], 0)
    return statusList

def checkMaxMeanDiff(input_array, diff_TH):
    max_T = np.max(input_array)
    mean_T = np.mean(input_array)
    check_T = max_T - mean_T
    # print(check_T)
    result_R = check_T < diff_TH # peak and mean difference is low
    return result_R

def maskUpdating(image, lMask, cMask):
  index0 = np.where(image<(np.mean(image)+80))#C 80 A60
  cMaskC = np.zeros((24,32))+100#.astype(np.uint8)
  cMaskC[index0] = 0
  cMask = (cMask+cMaskC)/2
  index1 = np.where(cMask > 95) #100% sure it is a keep out area
  lMask[index1] = 0
  return lMask, cMask

def maskCombination(img, mask):
  # change to 255 intensity and interpolate gray scale image.
  img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
  if C_DEV_FLAG:
      with open("Python code output - Normalisation.txt", "a") as file:
          np.savetxt(file, img, fmt='%d', delimiter=' ')
  mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

  if np.max(mask) == 0:
      return img

  kernelD = np.ones((3, 3), np.uint8)
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelD)
  mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX).astype(np.uint8)

  img = img * mask

  return img

def image_interpolation(img):
    columns = IR_FRAME_COLUMNS*Interpolation_scale
    rows = IR_FRAME_ROWS*Interpolation_scale
    img = cv2.resize(img, dsize=(columns, rows), interpolation=cv2.INTER_LANCZOS4)
    return img, columns, rows

def OTSU(input_image, apply_image, bias):
    otsu=threshold_otsu(input_image)
    sperated_img = cv2.threshold(apply_image, otsu+bias, 1, cv2.THRESH_BINARY)[1]
    return sperated_img

def get_forground_non0(mask, img):
    img = mask*img
    idx = np.nonzero(img)
    return img, img[idx]

def area_shrinking(mask, img):
    img, non0_value = get_forground_non0(mask, img)
    # foreground = mask*img
    # non0_fore = np.nonzero(foreground)
    mask = OTSU(non0_value, img, -10)
    return mask

def contour_detection(img):
    cntr = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntr = cntr[0] if len(cntr) == 2 else cntr[1]
    return cntr

def contour_properties(cntr):
    area = cv2.contourArea(cntr)
    minRect = cv2.minAreaRect(cntr)
    w = minRect[1][0]
    h = minRect[1][1]
    convex_hull = cv2.convexHull(cntr)
    convex_hull_area = cv2.contourArea(convex_hull)
    ratio_hull = area / (convex_hull_area+0.0001)
    return w, h, area, ratio_hull

def get_contour_pixels(cntr,img):
    mask = np.zeros([IR_FRAME_ROWS*Interpolation_scale,IR_FRAME_COLUMNS*Interpolation_scale])
    cv2.drawContours(mask,[cntr],0,255,-1)
    pixels = mask*img/255
    return pixels.astype(np.uint8)

def contour_shrinking(contour, image_inter):
    contour_pixels_img = get_contour_pixels(contour, image_inter)

    #send connected contour to otsu to do further shrinking and seperated the contours
    contour_pixels_non0 = contour_pixels_img[np.nonzero(contour_pixels_img)]
    seperated_contour_mask = OTSU(contour_pixels_non0, contour_pixels_img, 0)

    contour_list_sub = contour_detection(seperated_contour_mask)
    return contour_list_sub

def collect_current_contour(df, w, h,area, cntr, idx):
    aspect_ratio = min(w,h)/max(w,h)

    if aspect_ratio <0.4:
        return True, df, idx

    df.loc[idx] = [w, h, area, cntr]
    idx = idx + 1
    return False, df, idx

def contour_speration(contour_list, image_inter):
    index_count = 0
    df_currContours = pd.DataFrame(columns=['w', 'h', 'area', 'cntr'])

    for contour in contour_list:
        contour_w, contour_h, contour_area, contour_ratio_hull = contour_properties(contour)

        # discard thin contours that surranding keep out mask
        if contour_w==0 or contour_h == 0:
            continue

        # Separation of connected contours
        if contour_ratio_hull <0.93:
            contour_list_sub = contour_shrinking(contour, image_inter)

            for contour_sub in contour_list_sub:
                contour_w, contour_h, contour_area, contour_ratio_hull = contour_properties(contour_sub)

                # discard thin contours that surranding keep out mask
                if contour_w==0 or contour_h == 0:
                    continue

                continue_flag, df_currContours, index_count = collect_current_contour(df_currContours, contour_w, contour_h, contour_area, contour_sub, index_count)
                if continue_flag == True:
                    continue

        else:
            continue_flag, df_currContours, index_count = collect_current_contour(df_currContours, contour_w, contour_h, contour_area, contour, index_count)
            if continue_flag == True:
                continue
    return df_currContours

def showContours(img, contours0, color, thickness):
    for i in range(len(contours0)):
        cntr = contours0[i]
        cv2.drawContours(img, [cntr], 0, (0,0,0), thickness)

def CF_size(area):
    return area < 200 or area > (IR_FRAME_COLUMNS*Interpolation_scale*IR_FRAME_ROWS*Interpolation_scale)/3

def CF_peak_temp(row, img, mask):
    contour_pixels_img = get_contour_pixels(row['cntr'], img)
    forground_img, forground_non0_value = get_forground_non0(mask, img)
    centerTH = 0.6*(0.9*np.max(forground_non0_value)-np.median(forground_non0_value))+np.median(forground_non0_value)
    return np.max(contour_pixels_img) < centerTH

def CF_shape(row):
    area_ratio = row['area']/(row['w']*row['h'])
    return area_ratio <0.5

def contour_filtering(df, img, mask):
    drop_list = []
    for index, contour_row in df.iterrows():
        if CHECK_AREA_SIZE == True and CF_size(contour_row['area']):
            drop_list.append(index)
            continue

        if CHECK_PEAK_TEMP == True and CF_peak_temp(contour_row, img, mask):
            drop_list.append(index)
            continue

        if CHECK_SHAPE == True and CF_shape(contour_row):
            drop_list.append(index)
            continue

    df.drop(drop_list, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def alg_body(Imgarray, image, learningMask, confidenceMask, pir_status, PIR_statusList):

    # Apply keep in/out mask
    image_255 = maskCombination(image, learningMask)

    #needs to check again because image can be flat after adding the keep put mask
    if checkMaxMeanDiff(image_255, 80): #it was 110 but C got 80th
        if MODE == 1:
            # print('check_80')
            return confidenceMask, learningMask, 0, image, image, PIR_statusList
        elif MODE == 2:
            #save_dev_parameters(Imgarray, pir_status, PIR_statusList, learningMask, confidenceMask, np.array(-3), np.array(-3), 0)
            return 0
        else:
            return 0

    # ------------- Interpolation--------------------------------------
    image_inter, new_height, new_width = image_interpolation(image_255)
    if C_DEV_FLAG:
        with open("Python code output - Interpolation.txt", "a") as file:
            np.savetxt(file, image_inter, fmt='%d', delimiter=' ')

    # ------------- Heat scource seperation ---------------------------
    image_heat_mask = OTSU(image_inter, image_inter, -10)
    #if C_DEV_FLAG:
        #with open("Python code output - Thresholding2.txt", "a") as file:
            #np.savetxt(file, image_heat, fmt='%d', delimiter=' ')

    # ------------- TH calculation (aiming to shrink the size of contour)----
    if HEAT_AREA_SHRINKING == True:
        image_heat_mask = area_shrinking(image_heat_mask, image_inter)

    # ------------- Contour detection and counting---------------------
    contour_list = contour_detection(image_heat_mask)

    df_currContours = contour_speration (contour_list, image_inter)

    CountNum = len(df_currContours)
    if len(df_currContours) == 0:
        if MODE == 1:
            # print('no c')
            return confidenceMask, learningMask, 0, image, image, PIR_statusList
        elif MODE == 2:
            #save_dev_parameters(Imgarray, pir_status, PIR_statusList, learningMask, confidenceMask, np.array(-4), np.array(-4), 0)
            return 0
        else:
            return 0

    #if C_DEV_FLAG:
        #with open("Python code output - Contours.txt", "a") as file:
            #for firstDimension in contours:
                #for secondDimension in firstDimension:
                    #for thirdDimension in secondDimension:
                        #print(thirdDimension[0]," ",thirdDimension[1])
                        #file.write(str(thirdDimension[0])+" "+str(thirdDimension[1]) + "\n")

    if MODE == 1 or MODE == 2:
        label_img = image_inter.copy()
        showContours(label_img, df_currContours['cntr'], 'white',2)

    # ------------- Contour filtering ---------------------
    if CONTOUR_FILTERING == True:
        df_currContours = contour_filtering(df_currContours, image_inter, image_heat_mask)

        if C_DEV_FLAG:
            with open ("Python code output - Contours after filtering.txt", "a") as file:
                for firstDimension in range(len(df_currContours)):
                    for secondDimension in range(len(df_currContours['cntr'][firstDimension])):
                        np.savetxt(file, df_currContours['cntr'][firstDimension][secondDimension], fmt='%d', delimiter=' ')

    CountNum = len(df_currContours)

    if MODE == 1 or MODE == 2:
        contour_img = image_inter.copy()
        showContours(contour_img, df_currContours['cntr'], 'white',2)

    if MODE == 1:
        return confidenceMask, learningMask, CountNum, label_img, contour_img, PIR_statusList
    elif MODE == 2:
        #save_dev_parameters(Imgarray, pir_status, PIR_statusList, learningMask, confidenceMask, label_img, contour_img, CountNum)
        return CountNum
    else:
        return CountNum

MODE = 1

def estimate_pc_from_frame_blob_method(Imgarray,  pir_status):
    # for Keep out learning
    global learningMask
    global confidenceMask
    global count_reset
    global PIR_statusList

    PIR_statusList = update_pir_list(pir_status, PIR_statusList)

    if count_reset > PIR_reset:
        learningMask, confidenceMask, PIR_statusList, count_reset = global_initialization(PIR_continue)
    count_reset = count_reset +1


    if MAXIMUM_CLIPING == True:
        # ----- This is for maximum cliping,,, ,,
        index_high = np.where(Imgarray > MAX_temprature)
        Imgarray[index_high] = MAX_temprature

    #filter out the images that do not have heat source
    image_wc = np.array([np.round(Imgarray*cornerCorrectionMask)], np.int32)
    if C_DEV_FLAG:
        with open("Python code output - Masked input.txt", "a") as file:
            np.savetxt(file, image, fmt='%d', delimiter=' ')

    if checkMaxMeanDiff(image_wc, 500):#MLXC500 Orgin300
        if MODE == 1:
            return confidenceMask, learningMask, 0, np.zeros((24,32)), np.zeros((24,32)), PIR_statusList
        elif MODE == 2:
            #save_initial_values(initial_values_file, learningMask, confidenceMask, PIR_statusList, count_reset)
            #save_dev_parameters(Imgarray, pir_status, PIR_statusList, learningMask, confidenceMask, np.array(-1), np.array(-1), 0)
            return 0
        else:
            return 0

    # Reshape IR image from array to 24x32 image
    image = np.array([np.round(Imgarray)], np.int32)
    image = image.reshape(IR_FRAME_ROWS,IR_FRAME_COLUMNS)

    # ------------- keep in/out map updating------------
    #check if the current frame is a background frame
    if np.sum(PIR_statusList) == 0:
        learningMask, confidenceMask = maskUpdating(image, learningMask, confidenceMask)

        if MODE == 1:
            return confidenceMask, learningMask, 0, image, image, PIR_statusList
        elif MODE == 2:
            #save_initial_values(initial_values_file, learningMask, confidenceMask, PIR_statusList, count_reset)
            #save_dev_parameters(Imgarray, pir_status, PIR_statusList, learningMask, confidenceMask, np.array(-2), np.array(-2), 0)
            return 0
        else:
            return 0

    #if MODE == 2:
        #save_initial_values(initial_values_file, learningMask, confidenceMask, PIR_statusList, count_reset)


    output = alg_body(Imgarray, image, learningMask, confidenceMask, pir_status, PIR_statusList)
    return output

"""# **Définition des fonctions**"""

import matplotlib.pyplot as plt
import numpy as np
import ast

def display_images_in_row(df):
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

cornerCorrectionMask = np.array(cornerCorrectionMask).reshape(IR_FRAME_ROWS, IR_FRAME_COLUMNS)
# Test function to display the results

def test_estimate_pc_from_frame_blob_method(df):
    results = []
    for index in df.index:
        image_array = df.at[index, 'image array']
        if isinstance(image_array, str):
            image_array = ast.literal_eval(image_array)
        image_matrix = np.array(image_array).reshape(IR_FRAME_ROWS, IR_FRAME_COLUMNS)
        pir_status = df.at[index, 'occupancy']
        try:
            confidenceMask, learningMask, CountNum, label_img, contour_img, PIR_statusList = estimate_pc_from_frame_blob_method(image_matrix, pir_status)
            #print(f"Result for index {index}: {CountNum}")  # Added print for debugging
            results.append(CountNum)
        except ValueError as e:
            print(f"Error at index {index}: {e}")
            results.append(None) # Added a None result in case of error to keep the length consistent
    return results

# Function to test and calculate metrics for multiple dataframes
def test_and_calculate_metrics(dfs):
    predictions = {}
    for i, df in enumerate(dfs, start=1):
        res = test_estimate_pc_from_frame_blob_method(df)
        predictions[f'res_df{i}'] = res
    return predictions

def SO_evaluation(dataframes):
    datePlot = []
    accPlot_m = []
    accPlot_m2 = []
    accPlot_w = []
    accPlot_o = []
    acc_corr = []
    acc_corr2 = []

    predictions= test_and_calculate_metrics(dataframes)
    
    for i, df in enumerate(dataframes, 1):

        array_est = np.array(predictions[f'res_df{i}'])
        array_gt = np.array(df['occupancy'])
        count0 = 0
        for aaa in array_est:

            if aaa > 0:
                array_est[count0] = 1
            count0 = count0 + 1


        corr = np.corrcoef(array_est, array_gt)[0,1]
        acc_corr.append(corr)

        corr2 = np.corrcoef(array_est[180:540], array_gt[180:540])[0,1]
        acc_corr2.append(corr2)

        array_acc = np.zeros((len(array_est)))
        for i in range(0,len(array_est)):
            array_diff = abs(array_est[i]-array_gt[i])
            if array_diff == 0:
                array_acc[i] = 1

            if array_diff > 0:
                if array_gt[i] == 0:
                    array_acc[i] = 0
                if array_gt[i] < array_diff:
                    array_acc[i] = 0

                else:
                    array_acc[i] = 1 - (array_diff/array_gt[i])

        array_acc_overall = np.mean(array_acc)

        array_acc_overall2 = np.mean(array_acc[180:540])
        accPlot_m.append(array_acc_overall)
        accPlot_m2.append(array_acc_overall2)


        #print("==============================")
        print("People count:")
        print("Overall: ", array_acc_overall)
        #print("Overall2: ", array_acc_overall2)
        # print("No people: ", array_acc_no_people)
        # print("Only people: ", array_acc_only_people)
        #print("corr: ", corr)
        #print("corr2: ", corr2)
        #print("------------------------------")
        print(" ")

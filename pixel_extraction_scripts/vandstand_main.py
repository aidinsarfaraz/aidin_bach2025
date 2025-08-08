import cv2
import os
import ast
import pprint
import numpy as np
import pandas as pd
import pprint as pp
from params import *
from aux_non_zer_avg import average_non_zero

# print("Current working directory:", os.getcwd())

# # Params
# img_name = "028897"
# img_path = f"./img_separate/{img_name}_RIGHT.jpg"
# img_path_full = f"../../GEUS_data/3644593399_RIGHT/{img_name}_RIGHT.jpg"
# folder_path = "../../GEUS_data/3644593399_RIGHT/"
# save_to_folder = "SIFT/img_highlighted_pixels/"
# # folder_path2 : ""
# x_coord = 682
# center_y = 1685
# window_size = 5
# threshold = 35


# Func to identify sharpest increase in pixel intensity (i.e. shadow-water divide)

def find_transition_in_subarray(sub_array, threshold=50):
    """
    Finds the last index in a 1D array before a sharp increase in intensity.
    Returns the index of the 'last dark pixel' before the jump.

    Parameters:
        sub_array (np.ndarray): 1D array of grayscale values.
        threshold (int): Minimum difference to count as a sharp transition.

    Returns:
        int: Index within the sub_array of the last dark pixel before bright jump.
    """
    sub_array = sub_array.astype(np.int16)
    diffs = np.diff(sub_array)
    candidates = np.where(diffs > threshold)[0]
    print(f"diffs: {diffs}")
    print(f"Candidates: {candidates}")

    if len(candidates) == 0:
        return None  # No sharp transition found
    else:
        return candidates[0]  # First sharp jump in intensity


###########################################

# Complete sub-array and transition func

def completeTransitionPix(image_path, x_coord, center_y, img_name=img_name, window_size=window_size_params, threshold=threshold_params):

    # print(f"Processing {img_name}")
    # print(f"Threshold for operation: {threshold}")
    
    ######### Part 1: Generate subarray
    
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError("Image could not be read. Check the path.")

    height = img.shape[0]

    # Compute start and end index for y-range, ensuring bounds safety
    start_y = max(center_y - window_size, 0)
    end_y = min(center_y + window_size + 1, height)

    # Extract the vertical slice
    column = img[:, x_coord]
    sub_array = column[start_y:end_y]
    
    ###############################################################
    # Part 2: Find transition pixel position (y-coordinate)
    
    sub_array = sub_array.astype(np.int16)
    diffs = np.diff(sub_array)
    candidates = np.where(diffs > threshold)[0]
    # print(f"Sub array: {sub_array}")
    # print(f"diffs: {diffs}")
    # print(f"Candidates: {candidates}")

    if len(candidates) == 0:
        sub_array = sub_array.astype(np.int16).tolist()
        diffs = np.diff(sub_array).tolist()
        return [sub_array, diffs]  # No sharp transition found
    else:
        # First sharp jump in intensity
        real_y = center_y - window_size + candidates[0]
        return [int(real_y)]



###########################################

# Samme funktion som ovenfor, men udregner nu gennemsnit af 4 x-coords

def completeTransitionPixAvg(image_path, x_coords: list, center_ys:list, img_name=img_name, window_size=window_size_params, threshold=threshold_params):

    # Set up loval vars
    pivot_pixels = []
    
    ##########################################
    ####### Part 1: Generate subarray
    ##########################################
    
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError("Image could not be read. Check the path.")

    height = img.shape[0]

    # Compute start and end index for y-range, ensuring bounds safety
    start_ys = [center_ys[0] - window_size, 0,
                center_ys[1] - window_size, 0,
                center_ys[2] - window_size, 0,
                center_ys[3] - window_size, 0]
    end_ys = [min(center_ys[0] + window_size + 1, height),
              min(center_ys[1] + window_size + 1, height),
              min(center_ys[2] + window_size + 1, height),
              min(center_ys[3] + window_size + 1, height)]

    ##########################################
    # Part 2: Loop to extract vertical slice
    ##########################################

    for i in range(len(start_ys)):
        # Extract the vertical slice
        column = img[:, x_coords[i]]
        sub_array = column[start_ys[i]:end_ys[i]]
    
        # Convert and process slice (subarray)
        sub_array = sub_array.astype(np.int16)
        diffs = np.diff(sub_array)
        candidates = np.where(diffs > threshold)[0]
        # print(f"Sub array: {sub_array}")
        # print(f"diffs: {diffs}")
        # print(f"Candidates: {candidates}")

        if len(candidates) == 0:
            pivot_pixels.append(0)
        else:
            # First sharp jump in intensity
            real_y = center_ys[i] - window_size + candidates[0]
            pivot_pixels.append(int(real_y))
        
    # print(f"Pivot pixels: {pivot_pixels}")
    return (float(average_non_zero(pivot_pixels)), pivot_pixels)
    
##############################################################

# Process folder and apply func to all images in said folder

def process_folder(folder_path, window_size=window_size_params, pivot_pix_folder=pivot_pix_folder, save_csv=False):
    pivot_pixels = []
    unsuccessful_filenames = []
    counter = 0
    unsuccessful = 0

    for filename in os.listdir(folder_path):
        if filename.endswith("_RIGHT.jpg") or filename.endswith("_RIGHT.png"):
            filepath = os.path.join(folder_path, filename)

            # Apply pivot-finding function
            # print(f"Processing {filename}")
            pivot = completeTransitionPix(
                filepath,
                x_coord=x_coord1,
                center_y=center_y1,
                img_name=filepath,
                window_size=window_size
                )
            # print(f"Received from completeTransitionPix: {pivot}, Type: {type(pivot)}")

            if len(pivot) == 1:
                pivot_pixels.append([str(filename[:7]), pivot[0], [], []])
                counter += 1
            else:
                print(f"No transition found in: {filename}")
                unsuccessful_filenames.append(filename[:6])
                unsuccessful_subarray = pivot[0]
                unsuccessful_diff = pivot[1]
                pivot_pixels.append([str(filename[:7]), 0, unsuccessful_subarray, unsuccessful_diff])
                unsuccessful += 1
    
    # Count successful files
    print(f"Processed a total of {counter + unsuccessful} files.\tSuccesful/unsuccessful: {counter}/{unsuccessful}")
    # print(f"Unsuccessful diffs, type: {type(unsuccessful_diffs)}")

    
    # Save to CSV file
    if save_csv == True:
        with open(f"{folder_path}pivot_pixels.csv", mode="w", newline="") as f:
            df = pd.DataFrame(pivot_pixels, columns=["filename", "Pivot pixel", "Array of pixels", "Diffs"])  # Header
            df.to_csv(f"{pivot_pix_folder}transition_pixels_SEGS.csv", index=False)
            print(f"Saved CSV file to folder: {pivot_pix_folder}")

    # return pivot_pixels
    return unsuccessful_filenames

#######################################################

# Same as above but now calculates the average of 4 pivot pixels

def process_folder_avg(folder_path, x_coords_this: list, center_ys_this: list, window_size=window_size_params, pivot_pix_folder=pivot_pix_folder, save_csv=False):
    pivot_pixels = []
    unsuccessful_filenames = []
    counter = 0
    unsuccessful = 0

    for filename in os.listdir(folder_path):
        if filename.endswith("_RIGHT.jpg") or filename.endswith("_RIGHT.png"):
            filepath = os.path.join(folder_path, filename)

            # Apply pivot-finding function
            # print(f"Processing {filename}")
            pivot_tuple = completeTransitionPixAvg(
                filepath,
                x_coords = x_coords_this,
                center_ys = center_ys_this,
                img_name=filepath,
                window_size=window_size
                )
            # print(f"Received from completeTransitionPix: {pivot}, Type: {type(pivot)}")

            if pivot_tuple[0] is not 0:
                counter += 1
                pivot_pixels.append([str(filename[:7]), round(pivot_tuple[0], 2)])
            else:
                print(f"No transition found in: {filename}")
                unsuccessful += 1
                pivot_pixels.append(str(filename[:7]), pivot_tuple[0])



            # if len(pivot) == 1:
            #     pivot_pixels.append([str(filename[:7]), pivot[0], [], []])
            #     counter += 1
            # else:
            #     print(f"No transition found in: {filename}")
            #     unsuccessful_filenames.append(filename[:6])
            #     unsuccessful_subarray = pivot[0]
            #     unsuccessful_diff = pivot[1]
            #     pivot_pixels.append([str(filename[:7]), 0, unsuccessful_subarray, unsuccessful_diff])
            #     unsuccessful += 1
    
    # Count successful files
    print(f"Processed a total of {counter + unsuccessful} files.\nGenerated an avg. transition pixel in {counter} images ({unsuccessful} unsuccessful images).")

    
    # Save to CSV file
    if save_csv == True:
        with open(f"{folder_path}avg_pivot_pixels.csv", mode="w", newline="") as f:
            df = pd.DataFrame(pivot_pixels, columns=["filename", "Avg pivot pixel"])  # Header
            df.to_csv(f"{pivot_pix_folder}XX_avg_transition_pixels_SEGS.csv", index=False)
            print(f"Saved CSV file to folder: {pivot_pix_folder}")

    # return pivot_pixels
    return unsuccessful_filenames


#########################################

# Run programme

if __name__ == "__main__":

    print(f"\ncwd: {os.getcwd()}\n")

    # # Visualise single pic          VIRKER
    # get_pixel_intensity(img_name=img_name, img_path=img_path_full, save_img=True)

    # # Pivot idx, single image           VIRKER
    # pivot_idx = completeTransitionPix(image_path=img_path_full, x_coord=x_coord, center_y=center_y)
    # print(f"Received from completeTransitionPix(): {pivot_idx}")
    # if len(pivot_idx) == 1:
    #     # real_y = center_y - window_size + pivot_idx[0]
    #     print(f"Detected transition at y = {pivot_idx[0]}")
    # else:
    #     print(f"No sharp transition detected for threshold: {threshold}")

    # # Process full folder         FUNGERER
    # process_folder(folder_path_segs, save_csv=True)

    # # Test CompleteTransitionPixAvg
    # print(completeTransitionPixAvg(image_path=img_path_full,
    #                          x_coords=x_coords_params,
    #                          center_ys=center_ys_params,
    #                          ))
    
    # Test: Process full folder (averages)
    process_folder_avg(folder_path_segs,
                       x_coords_this=x_coords_params,
                       center_ys_this= center_ys_params,
                       window_size=25,
                       save_csv=True)
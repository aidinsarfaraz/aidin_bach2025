import cv2
import numpy as np
import os
from params import *

# # Local params
# save_to_folder_vis = "../SIFT/img_highlighted_subarray/"
img_name_this = "022300"
img_path_this = f"../../GEUS_data/3644593399_RIGHT/{img_name_this}_RIGHT.jpg"

def VisualiseSubarray(image_path, x_coord, center_y, img_name=img_name, window_size=window_size_params, threshold=threshold_params, save_img=False, save_csv=False):

    print(f"Processing {img_name}")
    # print(f"Threshold for operation: {threshold}")
    
    ######### Part 1: Generate subarray
    
    # Read the image in grayscale
    img_grey = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img_grey is None:
        raise FileNotFoundError("Image could not be read. Check the path.")

    height = img_grey.shape[0]

    x, y = x_coord, center_y
    pixel_intensity = img_grey[y, x]  # Note that the order is (row, column)
    color_grey = color = 0 if pixel_intensity > 127 else 255

    # Compute start and end index for y-range, ensuring bounds safety
    start_y = max(center_y - window_size, 0)
    end_y = min(center_y + window_size + 1, height)

    # Extract the vertical slice
    column = img_grey[:, x_coord]
    sub_array = column[start_y:end_y]
    
    ###############################################################
    # Part 2: Visualise

    # Defining the corners of the frame
    top_left = (x_coord - 1, start_y)
    bottom_right = (x_coord + 1, end_y - 1)

    # Draw the rectangle
    cv2.rectangle(img_grey, top_left, bottom_right, color_grey, thickness=1)

    # Save img to file
    if save_img == True:
        output_filename = os.path.join(save_to_folder_vis, f"{img_name}_RIGHT_subarray_highlight.png")
        cv2.imwrite(output_filename, img_grey)
        print(f"Saved image {img_name} to {output_filename}.")



#####################

def VisualiseMultipleSubarrays(image_path, x_coords, center_ys, img_name=img_name, window_size=window_size_params, threshold=threshold_params, save_img=False, save_csv=False):

    print(f"Processing {img_name}")
    # print(f"Threshold for operation: {threshold}")
    
    ######### Part 1: Generate subarray
    
    # Read the image in grayscale
    img_grey = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img_grey is None:
        raise FileNotFoundError("Image could not be read. Check the path.")

    height = img_grey.shape[0]

    for i in range(len(x_coords)):
        x, y = x_coords[i], center_ys[i]
        pixel_intensity = img_grey[y, x]  # Note that the order is (row, column)
        color_grey = color = 0 if pixel_intensity > 127 else 255

        # Compute start and end index for y-range, ensuring bounds safety
        start_y = max(center_ys[i] - window_size, 0)
        end_y = min(center_ys[i] + window_size + 1, height)

        # Extract the vertical slice
        column = img_grey[:, x_coords[i]]
        sub_array = column[start_y:end_y]
        
        ###############################################################
        # Part 2: Visualise

        # Defining the corners of the frame
        top_left = (x_coords[i] - 1, start_y)
        bottom_right = (x_coords[i] + 1, end_y - 1)

        # Draw the rectangle
        cv2.rectangle(img_grey, top_left, bottom_right, color_grey, thickness=1)

    # Save img to file
    if save_img == True:
        output_filename = os.path.join(save_to_folder_vis, f"{img_name}_RIGHT_subarray_highlight.png")
        cv2.imwrite(output_filename, img_grey)
        print(f"Saved image {img_name} to {output_filename}.")

###############################################
###############################################

# Run

if __name__ == "__main__":
    # Single subarray
    # VisualiseSubarray(image_path = img_path_full, x_coord=x_coord1, center_y=center_y1, save_img=True)
    
    # Multiple subarrays
    VisualiseMultipleSubarrays(image_path = img_path_this, x_coords= x_coords_params, center_ys=center_ys_params, save_img=True)
    
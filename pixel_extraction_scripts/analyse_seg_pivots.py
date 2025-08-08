from test_vandstand_1 import process_folder
from params import *
import cv2
import os
import numpy as np

save_csv = False

############################################
# Test reading of seg. image file
print(f"\ncwd: {os.getcwd()}\n")
file_path = folder_path_segs + img_name + "_RIGHT.png"
print(f"file_path: {file_path}\n")

seg_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

if seg_img is None:
    raise FileNotFoundError("Image could not be read. Check the path.")

# # Check if the x-coordinate is within the image width
if x_coord < 0 or x_coord >= seg_img.shape[1]:
    raise ValueError("x-coordinate is out of bounds.")

# Print shape of image
print(seg_img.shape)

# Access pixel intensities
pix_intensities = seg_img[:, x_coord]

# Write pix intensities to txt file
if save_csv == True:
    output_path = f"{pivot_pix_folder}/{img_name}_RIGHT_segs_pix_intensities" 
    with open(output_path, "w") as f:
        print(f"Pixel intensities for seg img file {img_path} at x-coordinate: {x_coord}")
        for y, intensity in enumerate(pix_intensities):
            # print(f"y = {y}: {intensity}")
            f.write(f"y = {y}:\t {intensity}\n")

    print(f"Segmentation pixel intensities written to file: {output_path}")

############################################



# process_folder(folder_path=folder_path_segs)


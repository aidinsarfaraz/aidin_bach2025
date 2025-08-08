import numpy as np
import cv2
import glob
import imutils
import os

# LOAD IMAGES ###############################

# Image in case
img_name = "028866"
# Check existence
print(f"Img_name {img_name} exists:", os.path.exists(f'./img_separate/{img_name}_LEFT.jpg'))

# Load image
img_path_L = f'img_separate/{img_name}_LEFT.jpg'
img_path_R = f'img_separate/{img_name}_RIGHT.jpg'

im_L = cv2.imread(img_path_L)
im_R = cv2.imread(img_path_R)
# Chek loading
if im_L is None:
    print("Error loading img_L")
if im_R is None:
    print("Error loading img_R")


# Create stitcher object
imageStitcher = cv2.Stitcher_create()

# Stitch images
error, stitchedImage = imageStitcher.stitch([im_L, im_R])

# Write to file
if not error:
    output_filename = os.path.join("./img_stitched/", f"{img_name}_stitched.png")
    cv2.imwrite(output_filename, stitchedImage)
    print(f"Stitched image '{img_name}' saved to {output_filename}.")
else:
    print(f"Error code: {error}")
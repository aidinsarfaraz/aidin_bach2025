import numpy as np
import cv2
import glob
import imutils
import os

# LOAD IMAGES ###############################

# Images in case
img_names = ["0" + str(i) for i in range(28880,28901)]
# print(img_names)

# # Check existence
# for name in img_names:
#     print(f"Img_name {name} exists:", os.path.exists(f'./img_separate/{name}_LEFT.jpg'))

# Create stitcher object
imageStitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

# Main loop
for img_name in img_names:
    # Load image
    img_path_L = f'img_separate/{img_name}_LEFT.jpg'
    img_path_R = f'img_separate/{img_name}_RIGHT.jpg'

    im_L = cv2.imread(img_path_L)
    im_R = cv2.imread(img_path_R)
    # Chek loading
    if im_L is None:
        print(f"Error loading img_L: {img_path_L}")
    if im_R is None:
        print(f"Error loading img_R: {img_path_R}")

    # Stitch images
    error, stitchedImage = imageStitcher.stitch([im_L, im_R])

    # Write to file
    if not error:
        output_filename = os.path.join("./img_stitched/", f"{img_name}_stitched.png")
        cv2.imwrite(output_filename, stitchedImage)
        print(f"Stitched image '{img_name}' saved to {output_filename}.")
    else:
        print(f"Failed to stitch image '{img_name}\t'Error code: {error}")
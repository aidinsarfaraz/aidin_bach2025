import numpy as np
import cv2
import glob
import imutils
import os
import time

# LOAD IMAGES ###############################

# Image in case
img_name = "028864"
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

########## GRAYSCALE ##########################

print("Performing greyscaling...")

im_L_grey = cv2.cvtColor(im_L, cv2.COLOR_BGR2GRAY)
im_R_grey = cv2.cvtColor(im_R, cv2.COLOR_BGR2GRAY)

# Convert grayscale images to 3-channel format
im_L_grey_3ch = cv2.cvtColor(im_L_grey, cv2.COLOR_GRAY2BGR)
im_R_grey_3ch = cv2.cvtColor(im_R_grey, cv2.COLOR_GRAY2BGR)

print("Greyscaling done.")

if im_L_grey is None:
    print("Error loading img_L_GREY")
if im_R_grey is None:
    print("Error loading img_R_GREY")

# Start timing the SIFT process
start_time = time.time()

# # Create stitcher object
imageStitcher = cv2.Stitcher_create()

###############################################
############## SIFT  ##########################################

# Create SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
keypoints_L, descriptors_L = sift.detectAndCompute(im_L_grey_3ch, None)
keypoints_R, descriptors_R = sift.detectAndCompute(im_R_grey_3ch, None)

# Create a matcher (e.g., FLANN or BFMatcher)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors
matches = bf.match(descriptors_L, descriptors_R)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Extract location of good matches
points_L = np.zeros((len(matches), 2), dtype=np.float32)
points_R = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points_L[i, :] = keypoints_L[match.queryIdx].pt
    points_R[i, :] = keypoints_R[match.trainIdx].pt

# Find homography
H, mask = cv2.findHomography(points_R, points_L, cv2.RANSAC)

# Warp the right image to the left image
height, width = im_L.shape[:2]
im_R_warped = cv2.warpPerspective(im_R_grey_3ch, H, (width, height))

# Combine images (simple blending)
stitchedImage = np.zeros((height, width, 3), dtype=np.uint8)

# Place the left image in the canvas
stitchedImage[0:height, 0:width] = im_L_grey_3ch

# Blend the warped right image into the canvas
for y in range(height):
    for x in range(width):
        if np.any(im_R_warped[y, x] > 0):  # Check if the pixel is not black
            stitchedImage[y, x] = im_R_warped[y, x]

output_filename = os.path.join("./img_stitched/", f"SIFT_{img_name}_stitched_GREY.png")
cv2.imwrite(output_filename, stitchedImage)

# Calculate the time taken
end_time = time.time()
stitching_time = end_time - start_time

# Print the time taken for stitching
print(f"Time taken for stitching attempt (SIFT): {stitching_time:.2f} seconds")

############## \SIFT ##########################################

###################################
############### O R B #############
# # Stitch images
# error, stitchedImage = imageStitcher.stitch([im_L_grey_3ch, im_R_grey_3ch])

# # Calculate the time taken
# end_time = time.time()
# stitching_time = end_time - start_time

# # Print the time taken for stitching
# print(f"Time taken for stitching attempt (ORB): {stitching_time:.2f} seconds")
      

# # Write to file
# if not error:
#     output_filename = os.path.join("./img_stitched/", f"{img_name}_stitched_GREY.png")
#     cv2.imwrite(output_filename, stitchedImage)
#     print(f"Stitched image '{img_name}_GREY' saved to {output_filename}.")
# else:
#     print(f"Error code: {error}")


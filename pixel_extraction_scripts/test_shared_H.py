import numpy as np
import cv2
import glob
import imutils
import os

# LOAD IMAGES ###############################

# Images in case
img_names = ["0" + str(i) for i in range(28880,28881)]
# print(img_names)

# # Check existence
# for name in img_names:
#     print(f"Img_name {name} exists:", os.path.exists(f'./img_separate/{name}_LEFT.jpg'))

# Directories
print(f"cwd: {os.getcwd()}")
input_dir = "img_separate"
output_dir = "img_shared_homography"

# Create ORB feature matcher
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Storage for shared H
H_shared = None

# Main loop
for i, img_name in enumerate(img_names):
    
    # Load image
    img_path_L = os.path.join(input_dir, f"{img_name}_LEFT.jpg")
    img_path_R = os.path.join(input_dir, f"{img_name}_RIGHT.jpg")

    im_L = cv2.imread(img_path_L)
    im_R = cv2.imread(img_path_R)
    # Chek loading
    if im_L is None:
        print(f"Error loading img_L: {img_path_L}")
        continue
    if im_R is None:
        print(f"Error loading img_R: {img_path_R}")
        continue

    # Get dimensions
    hL, wL = im_L.shape[:2]
    hR, wR = im_R.shape[:2]

    if H_shared is None:

        # Compute H from first pair only
        kp1, descrip1 = orb.detectAndCompute(im_L, None)
        kp2, descrip2 = orb.detectAndCompute(im_R, None)
        matches = bf.match(descrip1, descrip2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 4:
            print(f"Not enough matches to copmute H for image file: {img_name}.")
            continue

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H_shared, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H_shared is None:
            print(f"Failed to compute H in first par ({img_name}).")
            continue

    ####### END LOOP ###########

    # Compute canvas bounds
    corners_R = np.float32([[0, 0], [0, hR], [wR, hR], [wR, 0]]).reshape(-1, 1, 2)
    warped_corners_R = cv2.perspectiveTransform(corners_R, H_shared)

    corners_L = np.float32([[0, 0], [0, hL], [wL, hL], [wL, 0]]).reshape(-1, 1, 2)
    all_corners = np.concatenate((warped_corners_R, corners_L), axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Translation matrix to fit everything on positive canvas
    translation = [-x_min, -y_min]
    T = np.array([[1, 0, translation[0]],
                  [0, 1, translation[1]],
                  [0, 0, 1]])

    canvas_size = (x_max - x_min, y_max - y_min)

    # Warp RIGHT image with translation
    warped_R = cv2.warpPerspective(im_R, T @ H_shared, canvas_size)

    # Initialize panorama with warped RIGHT
    panorama = warped_R

    # Paste LEFT image into panorama with translation
    overlay = np.zeros_like(panorama)
    overlay[translation[1]:translation[1]+hL, translation[0]:translation[0]+wL] = im_L

    # Combine: avoid overwriting valid content from warped_R
    mask_panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY) > 0
    mask_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY) > 0
    final = panorama.copy()
    final[~mask_panorama & mask_overlay] = overlay[~mask_panorama & mask_overlay]

    # Save stitched image to file
    output_filename = os.path.join(output_dir, f"{img_name}_SHARED_H.png")
    cv2.imwrite(output_filename, panorama)
    print(f"Stitched image '{img_name}' saved to {output_filename}.")

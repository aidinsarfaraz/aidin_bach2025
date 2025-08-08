import cv2
import numpy as np

img_number = "028600"
path_img_L = f"../../GEUS_data/Aidin Samples/Tikee/Raw/3644593399/{img_number}_LEFT.jpg"
path_img_R = f"../../GEUS_data/Aidin Samples/Tikee/Raw/3644593399/{img_number}_RIGHT.jpg"

# Load images
img_L = cv2.imread(path_img_L, cv2.IMREAD_GRAYSCALE)
img_R = cv2.imread(path_img_R, cv2.IMREAD_GRAYSCALE)

# SIFT and ORB detectors
orb = cv2.ORB_create(nfeatures = 1000)

# Detect keypoints and descriptors
kp_L, des_L = orb.detectAndCompute(img_L, None)
kp_R, des_R = orb.detectAndCompute(img_R, None)

# Matcher (FLANN or BFMatcher)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(des_L, des_R)

# Sort by distance
matches = sorted(matches, key=lambda x: x.distance)

# Get top 3 matches
top_matches = matches[:20]

# # Draw matches
# img_matches = cv2.drawMatches(img_L, kp_L, img_R, kp_R, top_matches, None, flags=0)

# cv2.imwrite("top_matches_sift.png", img_matches)

pts1, pts2 = [], []

for m in top_matches:
    pt_L = kp_L[m.queryIdx].pt  # (x, y) in left image
    pts1.append(pt_L)
    pt_R = kp_R[m.trainIdx].pt  # (x, y) in right image
    pts2.append(pt_R)
    # print(f"Left: ({pt_L[0]:.2f}, {pt_L[1]:.2f}), Right: ({pt_R[0]:.2f}, {pt_R[1]:.2f})")


# Convert key points lists to NumPy arrays
pts1 = np.array(pts1, dtype=np.float32)
pts2 = np.array(pts2, dtype=np.float32)

# Estimate the fundamental matrix with RANSAC (also filters out outliers)
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

# Filter points to keep only inliers (good matches)
pts1_inliers = pts1[mask.ravel() == 1]
pts2_inliers = pts2[mask.ravel() == 1]

# Use the inlier points to estimate the rectification homographies
h, w = img_L.shape[:2]
_, H1, H2 = cv2.stereoRectifyUncalibrated(pts1_inliers, pts2_inliers, F, imgSize=(w, h))

# Warp images
img_L_rect = cv2.warpPerspective(img_L, H1, (w, h))
img_L_rect_cut = img_L_rect[:, :1000]
img_R_rect = cv2.warpPerspective(img_R, H2, (w, h))


# Save images
cv2.imwrite('left_rectified.png', img_L_rect)
cv2.imwrite('right_rectified.png', img_R_rect)

##################################################
########## PART 2: Running the same thing ########
##########      on rectified images       ########
##################################################

# Detect keypoints and descriptors
kp_L_rect, des_L_rect = orb.detectAndCompute(img_L_rect_cut, None)
kp_R_rect, des_R_rect = orb.detectAndCompute(img_R_rect, None)

# # Matcher (FLANN or BFMatcher)
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches_rect = bf.match(des_L_rect, des_R_rect)

# Sort by distance
matches_rect = sorted(matches_rect, key=lambda x: x.distance)

# Get top 3 matches
top_matches_rect = matches_rect[:1]

# Draw matches
img_matches_rect = cv2.drawMatches(img_L_rect_cut, kp_L_rect, img_R_rect, kp_R_rect, top_matches_rect, None, flags=0)

# Save image
cv2.imwrite("top_RECT_matches_sift_TOP1.png", img_matches_rect)

# Print matched key points' coordinates
for m in top_matches_rect:
    pt_L_rect = kp_L_rect[m.queryIdx].pt  # (x, y) in left image
    pt_R_rect = kp_R_rect[m.trainIdx].pt  # (x, y) in right image
    print(f"Left: ({pt_L_rect[0]:.2f}, {pt_L_rect[1]:.2f}), Right: ({pt_R_rect[0]:.2f}, {pt_R_rect[1]:.2f})")
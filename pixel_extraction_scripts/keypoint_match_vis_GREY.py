import cv2
import os

# Image in case
img_name = "028865"

# Verify image paths
print("Img1:", os.path.exists(f'./img_test/{img_name}_LEFT.jpg'))
print("Img2:", os.path.exists(f'./img_test/{img_name}_RIGHT.jpg'))

# Load images
img1_path = f'./img_test/{img_name}_LEFT.jpg'
img2_path = f'./img_test/{img_name}_RIGHT.jpg'

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

if img1 is None:
    print("Error loading img1")
if img2 is None:
    print("Error loading img2")

################# GREYSCALE ###############################
im_L_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
im_R_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Convert grayscale images to 3-channel format
im_L_grey_3ch = cv2.cvtColor(im_L_grey, cv2.COLOR_GRAY2BGR)
im_R_grey_3ch = cv2.cvtColor(im_R_grey, cv2.COLOR_GRAY2BGR)

# cv2.imshow("im_L_grey", im_L_grey)
# cv2.waitKey(0)
# cv2.imshow("im_R_grey", im_R_grey)
# cv2.waitKey(0)

###########################################################

# Create ORB detector
orb = cv2.ORB_create()

# Find keypoints and descriptors
keypoints1, descriptors1 = orb.detectAndCompute(im_L_grey_3ch, None)
keypoints2, descriptors2 = orb.detectAndCompute(im_R_grey_3ch, None)

# Create a BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(descriptors1, descriptors2)
print(f"Number of matches: {len(matches)}")

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw keypoints
img1_kp = cv2.drawKeypoints(im_L_grey, keypoints1, None, color=(0, 255, 0))
img2_kp = cv2.drawKeypoints(im_R_grey, keypoints2, None, color=(0, 255, 0))

# # Save keypoint visualizations
keypoint_output_dir = f"./img_vis/{img_name}"
os.makedirs(keypoint_output_dir, exist_ok=True)  # Create directory if it doesn't exist
cv2.imwrite(os.path.join(keypoint_output_dir, f"{img_name}_GREY_LEFT_keypoints.png"), img1_kp)
cv2.imwrite(os.path.join(keypoint_output_dir, f"{img_name}_GREY_RIGHT_keypoints.png"), img2_kp)

 # Create a visualization of the matches
img_matches = cv2.drawMatches(im_L_grey_3ch, keypoints1, im_R_grey_3ch, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


# Save matches visualization
cv2.imwrite(os.path.join(keypoint_output_dir, f"{img_name}_GREY_matches.png"), img_matches)
print(f"Keypoint and match visualizations saved for GREY version of {img_name}.")

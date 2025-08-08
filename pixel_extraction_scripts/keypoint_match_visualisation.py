import cv2
import os

# Image in case
img_name = "028874"

# Verify image paths
print("Img1:", os.path.exists(f'../img_test/{img_name}_LEFT.jpg'))
print("Img2:", os.path.exists(f'../img_test/{img_name}_RIGHT.jpg'))

# Load images
img1_path = f'./img_test/{img_name}_LEFT.jpg'
img2_path = f'./img_test/{img_name}_RIGHT.jpg'

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

if img1 is None:
    print("Error loading img1")
if img2 is None:
    print("Error loading img2")

# Create ORB detector
orb = cv2.ORB_create()

# Find keypoints and descriptors
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# Create a BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(descriptors1, descriptors2)
print(f"Number of matches: {len(matches)}")

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw keypoints
img1_kp = cv2.drawKeypoints(img1, keypoints1, None, color=(0, 255, 0))
img2_kp = cv2.drawKeypoints(img2, keypoints2, None, color=(0, 255, 0))

# # Save keypoint visualizations
# keypoint_output_dir = f"./img_vis/{img_name}"
# os.makedirs(keypoint_output_dir, exist_ok=True)  # Create directory if it doesn't exist
# cv2.imwrite(os.path.join(keypoint_output_dir, f"{img_name}_LEFT_keypoints.png"), img1_kp)
# cv2.imwrite(os.path.join(keypoint_output_dir, f"{img_name}_RIGHT_keypoints.png"), img2_kp)

# Draw matches
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[-10:], None, matchesThickness=3, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# # Save matches visualization
# cv2.imwrite(os.path.join(keypoint_output_dir, f"{img_name}_matches.png"), img_matches)
# print(f"Keypoint and match visualizations saved for {img_name}.")


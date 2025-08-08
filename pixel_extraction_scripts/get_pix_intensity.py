import cv2
import os
from params import *

# img_name = "028897"
img_name_shared = "028500"
# img_path = f"./img_separate/{img_name}_RIGHT.jpg"
# img_path_full = f"../../GEUS_data/3644593399_RIGHT/{img_name}_RIGHT.jpg"
img_path_shared_L = f"../../GEUS_data/Aidin Samples/Tikee/Raw/3644593399/{img_name_shared}_LEFT.jpg"
img_path_shared_R = f"../../GEUS_data/Aidin Samples/Tikee/Raw/3644593399/{img_name_shared}_RIGHT.jpg"
# folder_path = "../../GEUS_data/3644593399_RIGHT/"
save_to_folder = "../SIFT/img_highlighted_pixels/"
# x_coord = 682
# center_y = 1685
shared_pix_x_L, shared_pix_y_L = 4508, 1728
shared_pix_x_R, shared_pix_y_R = 455, 1732

def get_pixel_intensity(img_name: str, img_path: str, save_img=False):
    img_name = img_name
    img_path = img_path
    
    # Check if the files exist
    if not os.path.isfile(img_path):
        print(f"File not found: {img_path}")

    # Load image
    img = cv2.imread(img_path)
    img_grey = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Check if images are loaded correctly
    if img is None:
        print("Error loading img (colour)")
    if img_grey is None:
        print("Error loading img (grey)")

    ################### TEST ##############
    # Accessing individual pixel intensity value
    # print(type(img_grey))
    print(f"img_grey shape: {img_grey.shape}")

    x, y = shared_pix_x_L, shared_pix_y_L
    pixel_intensity = img_grey[y, x]  # Note that the order is (row, column)
    print(f"Pixel intensity at ({x}, {y}): {pixel_intensity}")

    # Highlight pixel
    circle_centre = (x, y)
    radius = 10
    color = (0, 255, 0)
    color_gray = color = 0 if pixel_intensity > 127 else 255
    thickness = 3

    # Apply circle
    cv2.circle(img_grey, circle_centre, radius, color_gray, thickness)

    # Save img to file
    if save_img == True:
        output_filename = os.path.join(save_to_folder, f"{img_name}_LEFT_highlight.png")
        cv2.imwrite(output_filename, img_grey)
        print(f"Saved image {img_name} to {output_filename}.\n")

    #######################################


if __name__ == "__main__":
    # print(f"\nCurrent pwd: {os.getcwd()}\n")
    get_pixel_intensity("028500", img_path=img_path_shared_L, save_img=True)
import cv2
import os

img_number = "028500"
path_img_L = f"../../GEUS_data/Aidin Samples/Tikee/Raw/3644593399/{img_number}_LEFT.jpg"
path_img_R = f"../../GEUS_data/Aidin Samples/Tikee/Raw/3644593399/{img_number}_RIGHT.jpg"

print(f"\npwd: {os.getcwd()}\n")

img_L = cv2.imread(path_img_L, cv2.IMREAD_GRAYSCALE)
img_R = cv2.imread(path_img_R, cv2.IMREAD_GRAYSCALE)

if img_L is None or img_R is None:
    raise FileNotFoundError("Images could not be read. Check the path.")

# Get info on height
img_height, img_width = img_L.shape[0], img_L.shape[1]
y_start = 2200
y_end = img_height

# --- 1. Define slices ---
# Horizontal overlap: last 1000 of left, first 1000 of right
left_overlap  = img_L[y_start:y_end, -1000:]
right_overlap = img_R[y_start:y_end, :1000]

# --- 2. Pick a patch near the bottom-right of the left overlap
template_size = 21  # Must be odd
half_patch = template_size // 2

# Coordinates relative to the slice
template_x = 950  # close to right edge of the 1000px wide slice
template_y = (y_end - y_start) - 50  # about 50px above bottom of slice

# bottom_right_x = top_left_x + 21
# bottom_right_y = top_left_y + 21

template = left_overlap[template_y - half_patch: template_y + half_patch + 1,
                        template_x - half_patch: template_x + half_patch + 1]


# --- 3. Match Template in Right Slice ---
result = cv2.matchTemplate(right_overlap, template, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(result)

# --- 4. Map Back to Full Image Coordinates ---

# LEFT image: location of the square patch
top_left_x_L = img_width - 1000 + (template_x - half_patch)
top_left_y_L = y_start + (template_y - half_patch)
bottom_right_x_L = top_left_x_L + template_size
bottom_right_y_L = top_left_y_L + template_size

# RIGHT image: top-left corner of matched region
matched_x_R = max_loc[0]
matched_y_R = max_loc[1]
matched_x_R_full = matched_x_R
matched_y_R_full = y_start + matched_y_R


# # Highlight square patch - LEFT IMAGE
# def HighlightSquarePatch(img):
#     # Draw rectangle where the template was taken from
#     cv2.rectangle(
#         img,
#         (top_left_x, top_left_y),
#         (bottom_right_x, bottom_right_y),
#         255,
#         2
#     )

#     # Circle at the center of the patch
#     center_x = top_left_x + 10
#     center_y = top_left_y + 10
#     cv2.circle(img, (center_x, center_y), 5, 255, -1)

#     # Save
#     output_filename = f"./{img_number}_L_template_patch.png"
#     cv2.imwrite(output_filename, img)
#     print(f"Saved LEFT image to {output_filename}.\n")


# # Highlight - RIGHT IMAGE
# def HighlightSharedPixWithCircle(img):
#     # Draw rectangle for matched region in right image
#     rect_top_left = (matched_x_R, matched_y_R)
#     rect_bottom_right = (matched_x_R + 21, matched_y_R + 21)
#     cv2.rectangle(img, rect_top_left, rect_bottom_right, 255, 2)

#     # Optional: circle at the center of the matched patch
#     circle_centre = (matched_x_R + 10, matched_y_R + 10)
#     cv2.circle(img, circle_centre, 5, 255, -1)

#     # Save
#     output_filename = f"./{img_number}_R_matched_patch.png"
#     cv2.imwrite(output_filename, img)
#     print(f"Saved RIGHT image to {output_filename}.\n")

def draw_patch_and_circle(img_L, img_R):
    # Copy for drawing
    img_L_draw = img_L.copy()
    img_R_draw = img_R.copy()

    # Draw rectangle on left image (template)
    cv2.rectangle(img_L_draw,
                  (top_left_x_L, top_left_y_L),
                  (bottom_right_x_L, bottom_right_y_L),
                  255, 2)

    # Draw circle at matched point on right image
    cv2.circle(img_R_draw,
               (matched_x_R_full, matched_y_R_full),
               radius=10, color=255, thickness=2)

    cv2.imwrite("./debug_L_patch.png", img_L_draw)
    cv2.imwrite("./debug_R_match.png", img_R_draw)

# # Print result
# print(f"Matched point in LEFT image:  ({top_left_x_L}, {top_left_y_L})")
# print(f"Matched point in RIGHT image: ({matched_x_R_full}, {matched_y_R_full})")
# print(f"Match score: {max_val:.3f}")

# # HighlightSquarePatch(img_L)
# # HighlightSharedPixWithCircle(img_R)

# draw_patch_and_circle(img_L, img_R)

##################################################
##################################################
##################################################


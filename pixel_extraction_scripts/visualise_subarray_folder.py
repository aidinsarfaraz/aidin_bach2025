from test_vandstand_1 import process_folder
from visualise_subarray import VisualiseSubarray
from params import *


def VisualiseSubarrayFolder(folder=folder_path_originals, save_folder=save_to_folder_vis):
    for filename in process_folder(folder):
        full_img_path = f"../../GEUS_data/3644593399_RIGHT/{filename}"
        VisualiseSubarray(
            image_path = full_img_path,
            img_name=filename,
            x_coord=x_coord,
            center_y=center_y,
            save_img=True
            )


if __name__ == "__main__":
    VisualiseSubarrayFolder()
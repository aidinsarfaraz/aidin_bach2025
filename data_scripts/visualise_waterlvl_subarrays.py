import pandas as pd
import matplotlib.pyplot as plt
import os

data_path_computed_pivots = "./pivot_pix_data/transition_pixels.csv"
data_path_segs = "./pivot_pix_data/transition_pixels_SEGS.csv"
avg_data_path_computed_pivots = "./pivot_pix_data/avg_transition_pixels_orig_images.csv"
avg_data_path_segs = "./pivot_pix_data/avg_transition_pixels_SEGS.csv"

def VisualiseWaterlevelPixel(data, title: str, save=False):

    df = pd.read_csv(data)
    df_sorted = df.sort_values(by=df.columns[0], ascending=True)

    fig, ax = plt.subplots(figsize=(20,6))
    ax.scatter(df_sorted["filename"], df_sorted["Avg pivot pixel"], marker='o', color='#5E19A8')
    ax.set_title(title)
    ax.set_xlabel("Img")
    ax.set_ylabel("Transition pixel")

    # Set y-axis limits
    ax.set_ylim(1670, 1697)

    # Rotate labels
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save img
    if save == True:
        plt.savefig("manuel_pivot_vandstand.png")

    plt.show()


print(f"cwd: {os.getcwd()}")

##################################################

if __name__ == "__main__":
    VisualiseWaterlevelPixel(avg_data_path_computed_pivots, "Original image transition pixels, single point (subarray computation)")
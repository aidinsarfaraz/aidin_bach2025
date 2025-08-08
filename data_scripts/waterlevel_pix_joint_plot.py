import pandas as pd
import matplotlib.pyplot as plt
from joint_waterlvl_read_data import PrepareDataJointWaterlvlStats



# Create plot/graph
def CreateJointWaterlevelPlot(computed_data, segs_data, manual_data):
    fig, ax = plt.subplots(figsize= (20,6))
    ax.scatter(computed_data["filename"], computed_data["Avg pivot pixel"], marker='o', color='#5E19A8', label="Computed transition pixels (original images)")
    ax.scatter(manual_data["filename"], manual_data["Avg pivot pixel"], marker = 'o', label="Manually notated transition pixels")
    ax.scatter(segs_data["filename"], segs_data["Avg pivot pixel"], marker='o', color='#19A83F', label="Computed transition pixels (segmentations)")

    ax.set_title("Waterlevel transition pixels (4 point average) - joint plot")
    ax.set_xlabel("Img")
    ax.set_ylabel("Transition pixel")

    ax.legend()

    # Set y-axis limitspi
    ax.set_ylim(1660, 1705)

    # Rotate labels
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.gca().invert_yaxis()

    plt.show()


####################################
####################################

if __name__ == "__main__":
    computed, segs, manual = PrepareDataJointWaterlvlStats()
    CreateJointWaterlevelPlot(computed, segs, manual)

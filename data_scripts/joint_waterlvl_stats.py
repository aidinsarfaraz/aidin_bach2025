import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from waterlevel_pix_joint_plot import PrepareDataJointWaterlvlStats

df_computed, df_segs, df_manual = PrepareDataJointWaterlvlStats()
dfs_all = [df_computed, df_segs, df_manual]
dfs_non_manual = [df_computed, df_segs]
dfs_str = ["Full images",
           "Segmentations",
           "Manually annotated"]
total_readings = [len(df_computed), len(df_segs), len(df_manual)]

# Helper func to count 0/NaNs in data sheets
def CountUnsuccessfulReadings(dataframe: pd.core.frame.DataFrame, column_name):
    if column_name in dataframe.columns:
        # Count empty strings and NaN values
        return dataframe[column_name].isna().sum() + (dataframe[column_name] == '').sum()
    else:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

# # Test of CountUnsuccessfulReadings() (VIRKER)
# print(f'Unsuccessful readings: {CountUnsuccessfulReadings(df_segs, "Avg pivot pixel")}')

########################################################
########################################################

# Func to create bar plot for readings success rate
def VisualiseSuccessRate(data: list, column, plot=True, CSV=False):
    unsuccessful_readings = []

    for dataframe in data:
        unsuccessful_readings.append(CountUnsuccessfulReadings(dataframe, column))
    # for i in range(len(unsuccessful_readings)):
    #     print(f"Data: {dfs_str[i]}\nUnsuccessful readings: {unsuccessful_readings[i]}\n")
    
    successful_readings = [s-u for u, s in zip(unsuccessful_readings, total_readings)]
    
    for i in range(len(dfs_all)):
        print(f"Successful readings for df: {dfs_str[i]}: {successful_readings[i]}/{total_readings[i]} ({unsuccessful_readings[i]} unsuccessful readings)")

    if plot == True:
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(successful_readings))

        plt.bar(x, successful_readings, color="blue", label="Successful readings")
        plt.bar(x, unsuccessful_readings, bottom=successful_readings, color="orange", label="Unsuccessful readings")

        plt.xlabel("Data")
        plt.ylabel("Readings")
        plt.title("Bar plot of Successful and Unsuccessful readings")
        plt.xticks(x, dfs_str)
        plt.legend()

        for i in range(len(successful_readings)):
            procent = round(successful_readings[i]/ total_readings[i] * 100, 2)
            full_text = str(successful_readings[i]) + f" ({procent})%"
            plt.text(i, successful_readings[i] + 1.5, full_text, ha='center', color='black')
            if unsuccessful_readings[i] != 0:
                plt.text(i, successful_readings[i] + unsuccessful_readings[i] + 1.5, str(total_readings[i]), ha='center', color='black')

        plt.show()


    return unsuccessful_readings

# # Test of VisualiseSuccessRate() (VIRKER)
# VisualiseSuccessRate(dfs_all, "Avg pivot pixel")

########################################################
########################################################

# Helper func to create joint data sheet (fill images, segs, manual)
# def CreateJointDataSheet(dataframes: list, save_CSV=False):
    df_computed_renamed = dataframes[0].rename(columns={"Avg pivot pixel": "Transition pixel (full image)"})
    df_segs_renamed = dataframes[1].rename(columns={"Avg pivot pixel": "Transition pixel (segs)"})
    if len(dataframes) == 3:
           df_manual_renamed = dataframes[2].rename(columns={"Avg pivot pixel": "Transition pixel (manual)"})

    if len(dataframes) == 3:
        merged_df = df_computed_renamed.merge(df_segs_renamed, on="filename").merge(df_manual_renamed[["filename", "Transition pixel (manual)"]], on="filename")
        merged_df_sorted = merged_df.sort_values(by="filename")
        merged_df_sorted["Dif. manual/full image"] = merged_df_sorted["Transition pixel (manual)"] - merged_df_sorted["Transition pixel (full image)"]
        merged_df_sorted["Dif. manual/segs"] = merged_df_sorted["Transition pixel (manual)"] - merged_df_sorted["Transition pixel (segs)"]
        print(merged_df_sorted.head())
        if save_CSV == True:
            merged_df_sorted.to_csv("./pivot_pix_data/joint_avg_waterlvl_all.csv", index=False)
            print("Wrote merged data (three sheets) to CSV file.")
    else:
        merged_df = df_computed_renamed.merge(df_segs_renamed, on="filename")
        merged_df_sorted = merged_df.sort_values(by="filename")
        merged_df_sorted["Dif. full image/segs"] = merged_df_sorted["Transition pixel (full image)"] - merged_df_sorted["Transition pixel (segs)"]
        if save_CSV == True:
            merged_df_sorted.to_csv("./pivot_pix_data/joint_avg_waterlvl_fullImage_segs.csv", index=False)
            print("Wrote merged data (two sheets) to CSV file.")

########################################################
########################################################

# Helper func to create joint data sheet (fill images, segs, manual)
def CreateJointDataSheet(dataframes: list, save_CSV=False):
    df_computed_renamed = dataframes[0].rename(columns={"Avg pivot pixel": "Transition pixel (full image)"})
    df_segs_renamed = dataframes[1].rename(columns={"Avg pivot pixel": "Transition pixel (segs)"})
    if len(dataframes) == 3:
           df_manual_renamed = dataframes[2].rename(columns={"Avg pivot pixel": "Transition pixel (manual)"})

    if len(dataframes) == 3:
        merged_df = df_computed_renamed.merge(df_segs_renamed, on="filename").merge(df_manual_renamed[["filename", "Transition pixel (manual)"]], on="filename")
        merged_df_sorted = merged_df.sort_values(by="filename")
        merged_df_sorted["Dif. manual/full image"] = round(merged_df_sorted["Transition pixel (manual)"] - merged_df_sorted["Transition pixel (full image)"], 2)
        merged_df_sorted["Dif. manual/segs"] = round(merged_df_sorted["Transition pixel (manual)"] - merged_df_sorted["Transition pixel (segs)"], 2)
        merged_df_sorted["Dif. fullImg/segs"] = round(merged_df_sorted["Transition pixel (full image)"] - merged_df_sorted["Transition pixel (segs)"], 2)
        print(merged_df_sorted.head())
        if save_CSV == True:
            merged_df_sorted.to_csv("./pivot_pix_data/XOX_joint_avg_waterlvl_all.csv", index=False)
            print("Wrote merged data (three sheets) to CSV file.")
    else:
        merged_df = df_computed_renamed.merge(df_segs_renamed, on="filename")
        merged_df_sorted = merged_df.sort_values(by="filename")
        merged_df_sorted["Dif. full image/segs"] = round(merged_df_sorted["Transition pixel (full image)"] - merged_df_sorted["Transition pixel (segs)"], 2)
        if save_CSV == True:
            merged_df_sorted.to_csv("./pivot_pix_data/joint_avg_waterlvl_fullImage_segs.csv", index=False)
            print("Wrote merged data (two sheets) to CSV file.")

# Test of CreateJointDataSheet()        (Virker)
# CreateJointDataSheet(dfs_all)

########################################################
########################################################

df_all_three = pd.read_csv("./pivot_pix_data/joint_avg_waterlvl_all.csv")
df_fullImage_segs = pd.read_csv("./pivot_pix_data/joint_avg_waterlvl_fullImage_segs.csv")

def PrintDifferentialStats():
    print(f"Accurate diff manual-fullimage:\t{df_all_three['Dif. manual/full image'].mean()}")
    print(f"ABS diff manual-fullimage:\t{abs(df_all_three['Dif. manual/full image']).mean()}")
    print()
    print(f"Accurate diff manual-segs:\t{df_all_three['Dif. manual/segs'].mean()}")
    print(f"ABS diff manual-segs:\t\t{abs(df_all_three['Dif. manual/segs']).mean()}")
    print()
    print(f"Images shared between fullImage and segs: {len(df_all_three)}")
    print(f"Accurate diff fullImage-segs:\t{df_all_three['Dif. fullImg/segs'].mean()}")
    print(f"ABS diff fullImage-segs:\t{abs(df_all_three['Dif. fullImg/segs']).mean()}")
    print()
    print(f"SD diffs manual-fullImg:\t{abs(df_all_three['Dif. manual/full image']).std()}")
    print(f"SD diffs manual-segs:\t\t{df_all_three['Dif. manual/segs'].std()}")
    print(f"SD diffs fullImg-segs:\t\t{df_all_three['Dif. fullImg/segs'].std()}")

    print(f'Unsuccessful readings: {CountUnsuccessfulReadings(df_all_three, "Dif. fullImg/segs")}')

PrintDifferentialStats()

########################################################
# Get unique values and their counts
counts_all = df_all_three["Dif. manual/segs"].value_counts().sort_index()
counts_fullImg_segs = df_all_three["Dif. fullImg/segs"].value_counts().sort_index()

def DifsBarPlotUniqueVals(difs_count):
    unique_vals = difs_count.index
    labels = [round(val, 2) for val in unique_vals]
    width = 0.5
    x = np.arange(len(unique_vals))
    
    fig, ax = plt.subplots(figsize=(16,12))
    ax.bar(x, difs_count.values, width, color="#AF2F2F")
    

    ax.axvline(x=11.55, color="#0A318C", linestyle='--', linewidth=2, label="Mean actual differential (-11.02)")

    # Adding labels and title
    ax.set_xlabel('Differential')
    ax.set_ylabel('Frequency')
    ax.set_title('Differentials: Full image vs. segmentations subarray')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)  # Set x-tick labels to unique values
    plt.xticks(rotation=90)

    max_count = difs_count.max()  # Get the maximum count for setting y-ticks
    ax.set_yticks(np.arange(0, max_count + 1, 1))

    ax.legend()
    plt.show()

# DifsBarPlotUniqueVals(counts_fullImg_segs)

############################################

def DifsBarPlotBuckets(difs_count):
    
    # Group into bins from 0 to max_count in steps of 5
    bins = np.arange(round(difs_count.index.min())-1, round(difs_count.index.max()) + 3, 3)  
    # labels = [f"({round(i)})-({round(i+4)})" for i in bins[:-1]]

    # Use np.histogram to count occurrences in each bin
    binned_counts = pd.cut(difs_count.index, bins=bins, right=False)
    bucket_counts = difs_count.groupby(binned_counts).sum()
    
    print()
    print(bucket_counts)

    # Set x positions based on the number of buckets
    x = np.arange(len(bucket_counts))   # x positions
    width = 0.5                         # Width of bars

    fig, ax = plt.subplots(figsize=(16, 12))

    # Create bars for each bucket
    ax.bar(x, bucket_counts, width, color="#AF2F2F")

    # Adding labels and title
    ax.set_xlabel('Difference in transition pixel y-coordinate')
    ax.set_ylabel('Frequency')
    ax.set_title('Binned differential: Full image-Segmentation transition pixel')
    ax.set_xticks(x)
    ax.set_xticklabels([f"({int(i.left)})-({int(i.right)})" for i in bucket_counts.index])

    # Set y-axis to show only specific whole numbers
    max_bucket_count = bucket_counts.max()  # Get the maximum count for setting y-ticks
    ax.set_yticks(np.arange(0, max_bucket_count + 1, 1))  # Set y-ticks from 0 to max_bucket_count with a step of 1

    # ax.legend()

    for i in range(len(bucket_counts)):
        procent = round(bucket_counts.iloc[i] / bucket_counts.sum() * 100, 1)
        full_text = str(bucket_counts.iloc[i]) + f" ({procent})%"
        plt.text(i, bucket_counts.iloc[i]+ 0.25, full_text, ha='center', color='black')

    plt.show()


# DifsBarPlotBuckets(counts_fullImg_segs)


import pandas as pd

# Read data files and create data frames
def PrepareDataJointWaterlvlStats():
    data_path_computed_pivots = "./pivot_pix_data/avg_transition_pixels_orig_images.csv"
    data_path_segs = "./pivot_pix_data/avg_transition_pixels_SEGS.csv"
    data_path_manual_waterlvl = "./pivot_pix_data/avg_vandstand_manuel.xlsx"

    df_computed = pd.read_csv(data_path_computed_pivots)
    df_segs = pd.read_csv(data_path_segs)
    df_manual = pd.read_excel(data_path_manual_waterlvl)

    # # Remove minus sign in and add underscore to filename
    # df_manual["Pivot pixel"] = df_manual["Pivot pixel"] * -1
    df_manual['filename'] = df_manual['filename'].apply(lambda x: f"0{x}")

    # Sort by filename (makes images appear chronologically)
    df_computed_sorted = df_computed.sort_values(by=df_computed.columns[0], ascending=True)
    df_segs_sorted = df_segs.sort_values(by=df_computed.columns[0], ascending=True)
    df_manual_sorted = df_manual.sort_values(by=df_manual.columns[0], ascending=True)

    return df_computed_sorted, df_segs_sorted, df_manual_sorted
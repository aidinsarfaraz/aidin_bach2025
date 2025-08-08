from visualise_waterlvl_subarrays import VisualiseWaterlevelPixel
import pandas as pd
import os
import matplotlib.pyplot as plt

manual_waterlvl_data_path = "./pivot_pix_data/vandstand_manuel.xlsx"


print(f"\ncurrentw pwd: {os.getcwd()}")

df = pd.read_excel(manual_waterlvl_data_path)
if df is None:
    print(f"Couldn't locate file: {manual_waterlvl_data_path}. Check file path.")
else:
    print(f"Successfully read file into dataframe.\n")

df['Pivot pixel'] = df['Pivot pixel'] * -1
df['filename'] = df['filename'].apply(lambda x: f"0{x}_")
df_sorted = df.sort_values(by=df.columns[0], ascending=True)
print(df.head())
fig, ax = plt.subplots(figsize=(20,6))
ax.scatter(df_sorted["filename"], df_sorted["Pivot pixel"], marker='o')
ax.set_title("Transition pixels (manual notation)")
ax.set_xlabel("Img")
ax.set_ylabel("Transition pixel")

# Set y-axis limits
ax.set_ylim(1674, 1697)

# Rotate labels
plt.xticks(rotation=45)
plt.tight_layout()

# Save img
plt.savefig("manuel_pivot_vandstand.png")

plt.show()